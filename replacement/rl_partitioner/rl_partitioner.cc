/**
 * @file rl_partitioner.cc
 * @brief 基于 Q-learning 和 Tile Coding 的 LLC 缓存分区策略实现
 */

#include "rl_partitioner.h"
#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iomanip>
#include "cache_stats.h"  // 访问全局统计信息 g_llc_stats

// ============================================================================
// 构造函数
// ============================================================================

rl_partitioner::rl_partitioner(CACHE* cache) 
    : rl_partitioner(cache, cache->NUM_SET, cache->NUM_WAY) 
{}

rl_partitioner::rl_partitioner(CACHE* cache, long sets, long ways) 
    : replacement(cache), 
      NUM_WAY(ways),
      last_used_cycles(static_cast<std::size_t>(sets * ways), 0),
      num_groups_(0),
      weights(TOTAL_WEIGHTS, 0.0),
      active_lr(LEARNING_RATE),
      active_epsilon(EPSILON),
      online_mode(false),
      last_update_cycle(0),
      rng(std::random_device{}()),
      uniform_dist(0.0, 1.0),
      last_accesses(MAX_CPUS_FOR_COMPETITION, 0),
      last_misses(MAX_CPUS_FOR_COMPETITION, 0),
      last_retired_instructions(MAX_CPUS_FOR_COMPETITION, 0),
      last_rl_global_cycle(0),
      baseline_initialized(false)
{
    replay_buffer.reserve(REPLAY_BUFFER_SIZE);
    std::cout << "[RL_Partitioner] 构造: NUM_WAY=" << NUM_WAY << std::endl;
}

// ============================================================================
// Replacement 接口函数实现
// ============================================================================

void rl_partitioner::initialize_replacement()
{
    // 1. 解析核心组配置
    parse_core_group_config();
    
    // 2. 初始化分区（各组均匀分配 Way）
    partition_ways_.resize(num_groups_);
    long base_ways = NUM_WAY / num_groups_;
    long remainder = NUM_WAY % num_groups_;
    for (int g = 0; g < num_groups_; g++) {
        partition_ways_[g] = base_ways + (g < remainder ? 1 : 0);
    }
    
    // 3. 初始化各核心组的状态向量
    last_norm_a_.assign(num_groups_, 0.0);
    last_norm_m_.assign(num_groups_, 0.0);
    last_norm_a_rest_.assign(num_groups_, 0.0);
    last_norm_m_rest_.assign(num_groups_, 0.0);
    last_partition_.resize(num_groups_);
    for (int g = 0; g < num_groups_; g++) last_partition_[g] = partition_ways_[g];
    last_actions_.assign(num_groups_, ACTION_KEEP);
    baseline_ipc_.assign(num_groups_, 0.0);
    baseline_ipc_rest_.assign(num_groups_, 0.0);
    
    // 4. 加载权重
    bool weights_loaded = false;
    
    const char* shm_path = std::getenv("RL_SHARED_WEIGHTS");
    if (shm_path != nullptr && std::strlen(shm_path) > 0) {
        if (shared_wt_.open(std::string(shm_path), TOTAL_WEIGHTS,
                            NUM_TILINGS, GRID_SIZE_A, GRID_SIZE_M,
                            static_cast<uint32_t>(MEMORY_SIZE), NUM_ACTIONS,
                            static_cast<int32_t>(NUM_WAY))) {
            shared_wt_.read_lock();
            shared_wt_.snapshot(weights);
            shared_wt_.read_unlock();
            weights_loaded = true;
            std::cout << "[RL_Partitioner] 已连接共享权重表: " << shm_path << std::endl;
        }
    }
    
    if (!weights_loaded) {
        const char* load_path_env = std::getenv("RL_WEIGHTS_LOAD");
        if (load_path_env != nullptr && std::strlen(load_path_env) > 0) {
            weights_loaded = load_weights(std::string(load_path_env));
        }
    }
    
    // 5. 在线模式检测
    const char* online_env = std::getenv("RL_ONLINE_MODE");
    if (weights_loaded && online_env != nullptr && std::string(online_env) == "1") {
        online_mode = true;
        active_lr = ONLINE_LEARNING_RATE;
        active_epsilon = ONLINE_EPSILON;
        std::cout << "[RL_Partitioner] 在线微调模式已启用" << std::endl;
    } else {
        online_mode = false;
        active_lr = LEARNING_RATE;
        active_epsilon = EPSILON;
        if (weights_loaded) {
            std::cout << "[RL_Partitioner] 已加载预训练权重，使用默认离线参数继续训练" << std::endl;
        }
    }
    
    // 6. 输出配置信息
    std::cout << "[RL_Partitioner] 替换策略初始化完成" << std::endl;
    std::cout << "  - 总 Way 数: " << NUM_WAY << std::endl;
    std::cout << "  - 核心组数: " << num_groups_ << std::endl;
    for (int g = 0; g < num_groups_; g++) {
        std::cout << "  - Group " << g << ": Way 数=" << partition_ways_[g]
                  << ", CPUs=[";
        for (size_t i = 0; i < group_cpus_[g].size(); i++) {
            if (i > 0) std::cout << ",";
            std::cout << group_cpus_[g][i];
        }
        std::cout << "]" << std::endl;
    }
    std::cout << "  - Tile Coding: NUM_TILINGS=" << NUM_TILINGS 
              << ", GRID_SIZE=" << GRID_SIZE_A << "x" << GRID_SIZE_M << std::endl;
    std::cout << "  - Q-Learning: LR=" << active_lr
              << ", GAMMA=" << GAMMA 
              << ", EPSILON=" << active_epsilon
              << (online_mode ? " (在线微调)" : " (离线训练)") << std::endl;
    std::cout << "  - Experience Replay: BUFFER=" << REPLAY_BUFFER_SIZE 
              << ", BATCH=" << REPLAY_BATCH_SIZE << std::endl;
}

long rl_partitioner::find_victim(uint32_t triggering_cpu, uint64_t instr_id, long set, 
                                  const champsim::cache_block* current_set, champsim::address ip,
                                  champsim::address full_addr, access_type type)
{
    maybe_update_rl();
    
    uint32_t group_id = get_group_id(triggering_cpu);
    
    // 根据核心组计算 Way 范围
    long partition_start = get_way_start(group_id);
    long partition_end = partition_start + partition_ways_[group_id];
    
    if (partition_start >= partition_end) {
        partition_start = 0;
        partition_end = NUM_WAY;
    }
    
    for (long w = partition_start; w < partition_end; w++) {
        if (!current_set[w].valid) {
            return w;
        }
    }
    
    auto begin = std::next(std::begin(last_used_cycles), set * NUM_WAY + partition_start);
    auto end = std::next(std::begin(last_used_cycles), set * NUM_WAY + partition_end);
    
    auto victim = std::min_element(begin, end);
    assert(begin <= victim);
    assert(victim < end);
    
    return partition_start + std::distance(begin, victim);
}

void rl_partitioner::update_replacement_state(uint32_t triggering_cpu, long set, long way, 
                                               champsim::address full_addr, champsim::address ip, 
                                               champsim::address victim_addr, access_type type, uint8_t hit)
{
    // 命中时更新 LRU 时间戳
    if (hit && access_type{type} != access_type::WRITE) {
        last_used_cycles.at(static_cast<std::size_t>(set * NUM_WAY + way)) = cycle++;
    }
}

void rl_partitioner::replacement_cache_fill(uint32_t triggering_cpu, long set, long way, 
                                             champsim::address full_addr, champsim::address ip, 
                                             champsim::address victim_addr, access_type type)
{
    // 填充时更新 LRU 时间戳
    last_used_cycles.at(static_cast<std::size_t>(set * NUM_WAY + way)) = cycle++;
}

void rl_partitioner::replacement_final_stats()
{
    std::cout << "\n[RL_Partitioner] 最终统计信息:" << std::endl;
    std::cout << "  - 模式: " << (online_mode ? "在线微调" : "离线训练") << std::endl;
    std::cout << "  - 核心组数: " << num_groups_ << std::endl;
    for (int g = 0; g < num_groups_; g++) {
        std::cout << "  - Group " << g << ": Way 数=" << partition_ways_[g]
                  << ", IPC 基线=" << std::fixed << std::setprecision(4) << baseline_ipc_[g]
                  << std::endl;
    }
    std::cout << "  - 实际参数: LR=" << active_lr << ", EPSILON=" << active_epsilon << std::endl;
    std::cout << "  - Experience Replay 缓冲区使用量: " << replay_buffer.size() 
              << " / " << REPLAY_BUFFER_SIZE << std::endl;
    std::cout << "  - 共享权重表: " << (shared_wt_.is_open() ? "已连接" : "未使用") << std::endl;
    
    if (shared_wt_.is_open()) {
        shared_wt_.read_lock();
        shared_wt_.snapshot(weights);
        shared_wt_.read_unlock();
        std::cout << "[RL_Partitioner] 已从共享内存同步最新权重" << std::endl;
    }
    
    const char* save_path_env = std::getenv("RL_WEIGHTS_SAVE");
    if (save_path_env != nullptr && std::strlen(save_path_env) > 0) {
        save_weights(std::string(save_path_env));
    } else {
        std::cout << "[RL_Partitioner] 未设置 RL_WEIGHTS_SAVE，权重未保存" << std::endl;
    }
}

// ============================================================================
// Tile Coding 相关函数实现
// ============================================================================

double rl_partitioner::log_normalize(uint64_t value) const
{
    // 对数变换: log(1 + value)
    double log_value = std::log(1.0 + static_cast<double>(value));
    // 归一化到 [0, 1] 区间
    double normalized = log_value / LOG_MAX_VALUE;
    // 截断到 [0, 1]
    return std::min(1.0, std::max(0.0, normalized));
}

size_t rl_partitioner::compute_tile_index(int tiling_id, double norm_a, double norm_m, long p) const
{
    // 对 A 和 M 维度应用 Tiling 偏移
    // 偏移量 = tiling_id / NUM_TILINGS / GRID_SIZE
    double offset_a = static_cast<double>(tiling_id) / NUM_TILINGS / GRID_SIZE_A;
    double offset_m = static_cast<double>(tiling_id) / NUM_TILINGS / GRID_SIZE_M;
    
    // 计算网格坐标（带偏移）
    int coord_a = static_cast<int>((norm_a + offset_a) * GRID_SIZE_A) % GRID_SIZE_A;
    int coord_m = static_cast<int>((norm_m + offset_m) * GRID_SIZE_M) % GRID_SIZE_M;
    
    // 确保坐标非负
    if (coord_a < 0) coord_a += GRID_SIZE_A;
    if (coord_m < 0) coord_m += GRID_SIZE_M;
    
    // P 维度不应用偏移，直接作为上下文
    // 不同的 P 对应完全独立的权重平面
    
    // 使用位混合哈希计算索引
    uint64_t hash_input = static_cast<uint64_t>(tiling_id);
    hash_input = hash_input * 31 + static_cast<uint64_t>(coord_a);
    hash_input = hash_input * 31 + static_cast<uint64_t>(coord_m);
    hash_input = hash_input * 31 + static_cast<uint64_t>(p);
    
    return static_cast<size_t>(hash_mix(hash_input) % MEMORY_SIZE);
}

std::vector<size_t> rl_partitioner::get_active_tiles(double norm_a, double norm_m, long p) const
{
    std::vector<size_t> tiles;
    tiles.reserve(NUM_TILINGS);
    
    for (int tiling_id = 0; tiling_id < NUM_TILINGS; tiling_id++) {
        tiles.push_back(compute_tile_index(tiling_id, norm_a, norm_m, p));
    }
    
    return tiles;
}

// ============================================================================
// Q-Learning 相关函数实现
// ============================================================================

double rl_partitioner::compute_q_single(const std::vector<size_t>& tiles, Action action) const
{
    double q_value = 0.0;
    size_t action_offset = static_cast<size_t>(action) * MEMORY_SIZE;
    
    for (size_t tile : tiles) {
        q_value += weights[action_offset + tile];
    }
    
    return q_value;
}

double rl_partitioner::compute_global_q(double norm_a1, double norm_m1, 
                                         double norm_a2, double norm_m2, 
                                         long p, Action action) const
{
    // Term 1: Group 1 视角
    // 状态: (A1, M1, P)
    // 动作: action
    std::vector<size_t> tiles1 = get_active_tiles(norm_a1, norm_m1, p);
    double q1 = compute_q_single(tiles1, action);
    
    // Term 2: Group 2 视角（镜像映射）
    // 状态: (A2, M2, N-P)  [P 变为 N-P]
    // 动作: flip(action)   [动作翻转]
    long p_mirrored = NUM_WAY - p;
    Action action_flipped = flip_action(action);
    std::vector<size_t> tiles2 = get_active_tiles(norm_a2, norm_m2, p_mirrored);
    double q2 = compute_q_single(tiles2, action_flipped);
    
    // 总 Q 值 = Q1 + Q2
    return q1 + q2;
}

rl_partitioner::Action rl_partitioner::flip_action(Action action) const
{
    // 动作翻转: INC <-> DEC, KEEP 不变
    switch (action) {
        case ACTION_INC:  return ACTION_DEC;
        case ACTION_DEC:  return ACTION_INC;
        case ACTION_KEEP: return ACTION_KEEP;
        default:          return ACTION_KEEP;
    }
}

rl_partitioner::Action rl_partitioner::select_action(double norm_a1, double norm_m1, 
                                                      double norm_a2, double norm_m2, long p)
{
    // ε-greedy 策略
    if (uniform_dist(rng) < active_epsilon) {
        // 探索：随机选择动作
        int random_action = static_cast<int>(uniform_dist(rng) * NUM_ACTIONS);
        return static_cast<Action>(std::min(random_action, NUM_ACTIONS - 1));
    } else {
        // 利用：选择最优动作
        return get_best_action(norm_a1, norm_m1, norm_a2, norm_m2, p);
    }
}

rl_partitioner::Action rl_partitioner::get_best_action(double norm_a1, double norm_m1, 
                                                        double norm_a2, double norm_m2, long p) const
{
    Action best_action = ACTION_KEEP;
    double best_q = compute_global_q(norm_a1, norm_m1, norm_a2, norm_m2, p, ACTION_KEEP);
    
    for (int a = 0; a < NUM_ACTIONS; a++) {
        Action action = static_cast<Action>(a);
        
        // 检查动作是否有效（边界检查）
        long new_p = p;
        if (action == ACTION_INC) new_p = p + 1;
        else if (action == ACTION_DEC) new_p = p - 1;
        
        // 确保 P 在 [1, N_ways - 1] 范围内
        if (new_p < 1 || new_p >= NUM_WAY) {
            continue;  // 跳过无效动作
        }
        
        double q = compute_global_q(norm_a1, norm_m1, norm_a2, norm_m2, p, action);
        if (q > best_q) {
            best_q = q;
            best_action = action;
        }
    }
    
    return best_action;
}

// ============================================================================
// 奖励计算
// ============================================================================

double rl_partitioner::compute_reward(double current_ipc, double baseline_ipc) const
{
    if (baseline_ipc <= 0) {
        return 0.0;
    }
    
    // 奖励 = β * (IPC - baseline) / baseline
    double reward = REWARD_BETA * (current_ipc - baseline_ipc) / baseline_ipc;
    
    // 截断到 [-REWARD_CLIP, REWARD_CLIP]
    return std::max(-REWARD_CLIP, std::min(REWARD_CLIP, reward));
}

void rl_partitioner::update_baseline(double& baseline, double current_ipc)
{
    // EWMA 更新: baseline = (1 - α) * baseline + α * current_ipc
    baseline = (1.0 - REWARD_ALPHA) * baseline + REWARD_ALPHA * current_ipc;
}

// ============================================================================
// 训练函数实现
// ============================================================================

void rl_partitioner::train(double norm_a1, double norm_m1, double norm_a2, double norm_m2, 
                            long p, Action action, double r1, double r2,
                            double norm_a1_next, double norm_m1_next, 
                            double norm_a2_next, double norm_m2_next, long p_next)
{
    // ===== 更新 1: Group 1 经验 =====
    // State: (A1, M1, P), Action: action, Reward: r1
    {
        std::vector<size_t> tiles = get_active_tiles(norm_a1, norm_m1, p);
        
        // 计算当前 Q 值
        double q_current = compute_q_single(tiles, action);
        
        // 计算目标 Q 值（使用全局 Q 值选择最佳动作）
        Action best_next = get_best_action(norm_a1_next, norm_m1_next, norm_a2_next, norm_m2_next, p_next);
        
        // 计算下一状态的 Q 值（Group 1 视角）
        std::vector<size_t> tiles_next = get_active_tiles(norm_a1_next, norm_m1_next, p_next);
        double q_next = compute_q_single(tiles_next, best_next);
        
        // TD 目标
        double td_target = r1 + GAMMA * q_next;
        
        // TD 误差
        double td_error = td_target - q_current;
        
        // 更新权重
        update_weights(tiles, action, td_error);
    }
    
    // ===== 更新 2: Group 2 经验（镜像映射）=====
    // State: (A2, M2, N-P), Action: flip(action), Reward: r2
    {
        long p_mirrored = NUM_WAY - p;
        Action action_flipped = flip_action(action);
        
        std::vector<size_t> tiles = get_active_tiles(norm_a2, norm_m2, p_mirrored);
        
        // 计算当前 Q 值
        double q_current = compute_q_single(tiles, action_flipped);
        
        // 计算目标 Q 值
        // 注意：下一状态也需要镜像映射
        long p_next_mirrored = NUM_WAY - p_next;
        Action best_next = get_best_action(norm_a1_next, norm_m1_next, norm_a2_next, norm_m2_next, p_next);
        Action best_next_flipped = flip_action(best_next);
        
        // 计算下一状态的 Q 值（Group 2 视角）
        std::vector<size_t> tiles_next = get_active_tiles(norm_a2_next, norm_m2_next, p_next_mirrored);
        double q_next = compute_q_single(tiles_next, best_next_flipped);
        
        // TD 目标
        double td_target = r2 + GAMMA * q_next;
        
        // TD 误差
        double td_error = td_target - q_current;
        
        // 更新权重
        update_weights(tiles, action_flipped, td_error);
    }
}

void rl_partitioner::update_weights(const std::vector<size_t>& tiles, Action action, double td_error)
{
    size_t action_offset = static_cast<size_t>(action) * MEMORY_SIZE;
    
    double step = active_lr * td_error / static_cast<double>(tiles.size());
    
    for (size_t tile : tiles) {
        size_t idx = action_offset + tile;
        weights[idx] += step;
        if (shared_wt_.is_open()) {
            pending_deltas_.emplace_back(idx, step);
        }
    }
}

// ============================================================================
// Experience Replay 实现
// ============================================================================

void rl_partitioner::store_experience(const Experience& exp)
{
    if (replay_buffer.size() < REPLAY_BUFFER_SIZE) {
        replay_buffer.push_back(exp);
    } else {
        replay_buffer[replay_write_pos] = exp;
    }
    replay_write_pos = (replay_write_pos + 1) % REPLAY_BUFFER_SIZE;
}

void rl_partitioner::train_single_experience(const Experience& exp)
{
    std::vector<size_t> tiles = get_active_tiles(exp.norm_a, exp.norm_m, exp.p);
    double q_current = compute_q_single(tiles, exp.action);
    
    std::vector<size_t> tiles_next = get_active_tiles(exp.norm_a_next, exp.norm_m_next, exp.p_next);
    
    Action best_next = ACTION_KEEP;
    double best_q = compute_q_single(tiles_next, ACTION_KEEP);
    
    for (int a = 0; a < NUM_ACTIONS; a++) {
        Action act = static_cast<Action>(a);
        long new_p = exp.p_next;
        if (act == ACTION_INC) new_p = exp.p_next + 1;
        else if (act == ACTION_DEC) new_p = exp.p_next - 1;
        
        if (new_p < 1 || new_p >= NUM_WAY) continue;
        
        double q = compute_q_single(tiles_next, act);
        if (q > best_q) {
            best_q = q;
            best_next = act;
        }
    }
    
    double td_target = exp.reward + GAMMA * best_q;
    double td_error = td_target - q_current;
    update_weights(tiles, exp.action, td_error);
}

void rl_partitioner::replay_experiences()
{
    if (replay_buffer.size() < REPLAY_BATCH_SIZE) return;
    
    std::uniform_int_distribution<size_t> index_dist(0, replay_buffer.size() - 1);
    
    for (size_t i = 0; i < REPLAY_BATCH_SIZE; i++) {
        size_t idx = index_dist(rng);
        train_single_experience(replay_buffer[idx]);
    }
}

// ============================================================================
// 权重持久化实现（离线预训练 + 在线微调）
// ============================================================================

#pragma pack(push, 1)
struct WeightsFileHeader {
    uint32_t magic;
    uint32_t version;
    int32_t  num_tilings;
    int32_t  grid_size_a;
    int32_t  grid_size_m;
    uint32_t memory_size;
    int32_t  num_actions;
    int32_t  num_way;
};
#pragma pack(pop)

bool rl_partitioner::save_weights(const std::string& path) const
{
    // 多进程环境下，使用文件锁防止并发写入同一文件
    std::string lock_path = path + ".lock";
    int lock_fd = ::open(lock_path.c_str(), O_RDWR | O_CREAT, 0666);
    if (lock_fd >= 0) {
        flock(lock_fd, LOCK_EX);
    }
    
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs.is_open()) {
        std::cerr << "[RL_Partitioner] 无法打开文件进行写入: " << path << std::endl;
        if (lock_fd >= 0) { flock(lock_fd, LOCK_UN); ::close(lock_fd); }
        return false;
    }
    
    WeightsFileHeader header{};
    header.magic       = WEIGHTS_FILE_MAGIC;
    header.version     = WEIGHTS_FILE_VERSION;
    header.num_tilings = NUM_TILINGS;
    header.grid_size_a = GRID_SIZE_A;
    header.grid_size_m = GRID_SIZE_M;
    header.memory_size = static_cast<uint32_t>(MEMORY_SIZE);
    header.num_actions = NUM_ACTIONS;
    header.num_way     = static_cast<int32_t>(NUM_WAY);
    
    ofs.write(reinterpret_cast<const char*>(&header), sizeof(header));
    ofs.write(reinterpret_cast<const char*>(weights.data()),
              static_cast<std::streamsize>(TOTAL_WEIGHTS * sizeof(double)));
    
    if (!ofs.good()) {
        std::cerr << "[RL_Partitioner] 权重文件写入失败: " << path << std::endl;
        if (lock_fd >= 0) { flock(lock_fd, LOCK_UN); ::close(lock_fd); }
        return false;
    }
    
    std::cout << "[RL_Partitioner] 权重已保存到: " << path
              << " (" << sizeof(header) + TOTAL_WEIGHTS * sizeof(double) << " bytes)" << std::endl;
    
    if (lock_fd >= 0) {
        flock(lock_fd, LOCK_UN);
        ::close(lock_fd);
    }
    return true;
}

bool rl_partitioner::load_weights(const std::string& path)
{
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs.is_open()) {
        std::cout << "[RL_Partitioner] 未找到权重文件: " << path
                  << "，从零开始训练" << std::endl;
        return false;
    }
    
    WeightsFileHeader header{};
    ifs.read(reinterpret_cast<char*>(&header), sizeof(header));
    if (!ifs.good()) {
        std::cerr << "[RL_Partitioner] 权重文件头读取失败" << std::endl;
        return false;
    }
    
    if (header.magic != WEIGHTS_FILE_MAGIC) {
        std::cerr << "[RL_Partitioner] 权重文件格式错误 (magic mismatch)" << std::endl;
        return false;
    }
    if (header.version != WEIGHTS_FILE_VERSION) {
        std::cerr << "[RL_Partitioner] 权重文件版本不匹配: 文件=" << header.version
                  << ", 期望=" << WEIGHTS_FILE_VERSION << std::endl;
        return false;
    }
    
    bool compatible = (header.num_tilings == NUM_TILINGS)
                   && (header.grid_size_a == GRID_SIZE_A)
                   && (header.grid_size_m == GRID_SIZE_M)
                   && (header.memory_size == static_cast<uint32_t>(MEMORY_SIZE))
                   && (header.num_actions == NUM_ACTIONS)
                   && (header.num_way == static_cast<int32_t>(NUM_WAY));
    
    if (!compatible) {
        std::cerr << "[RL_Partitioner] 超参数不匹配，拒绝加载权重" << std::endl;
        std::cerr << "  文件: TILINGS=" << header.num_tilings
                  << " GRID=" << header.grid_size_a << "x" << header.grid_size_m
                  << " MEM=" << header.memory_size
                  << " ACTIONS=" << header.num_actions
                  << " WAY=" << header.num_way << std::endl;
        std::cerr << "  当前: TILINGS=" << NUM_TILINGS
                  << " GRID=" << GRID_SIZE_A << "x" << GRID_SIZE_M
                  << " MEM=" << MEMORY_SIZE
                  << " ACTIONS=" << NUM_ACTIONS
                  << " WAY=" << NUM_WAY << std::endl;
        return false;
    }
    
    ifs.read(reinterpret_cast<char*>(weights.data()),
             static_cast<std::streamsize>(TOTAL_WEIGHTS * sizeof(double)));
    if (!ifs.good()) {
        std::cerr << "[RL_Partitioner] 权重数据读取失败，重置为零" << std::endl;
        std::fill(weights.begin(), weights.end(), 0.0);
        return false;
    }
    
    std::cout << "[RL_Partitioner] 成功加载预训练权重: " << path << std::endl;
    return true;
}

// ============================================================================
// 辅助函数实现
// ============================================================================

void rl_partitioner::apply_actions(const std::vector<Action>& actions)
{
    bool changed = false;
    long max_per_group = NUM_WAY - (num_groups_ - 1);
    
    for (int g = 0; g < num_groups_; g++) {
        long old_p = partition_ways_[g];
        switch (actions[g]) {
            case ACTION_INC:
                partition_ways_[g] = std::min(partition_ways_[g] + 1, max_per_group);
                break;
            case ACTION_DEC:
                partition_ways_[g] = std::max(partition_ways_[g] - 1, 1L);
                break;
            case ACTION_KEEP:
            default:
                break;
        }
        if (old_p != partition_ways_[g]) changed = true;
    }
    
    if (changed) {
        std::cout << "[RL_Partitioner] 分区调整:";
        for (int g = 0; g < num_groups_; g++) {
            std::cout << " G" << g << "=" << partition_ways_[g];
        }
        std::cout << std::endl;
    }
}

uint64_t rl_partitioner::hash_mix(uint64_t x)
{
    // MurmurHash3 的 finalizer
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return x;
}

uint32_t rl_partitioner::get_group_id(uint32_t cpu) const
{
    if (cpu < cpu_to_group_.size()) {
        return cpu_to_group_[cpu];
    }
    return 0;
}

void rl_partitioner::parse_core_group_config()
{
    cpu_to_group_.resize(NUM_CPUS, 0);
    
    const char* groups_env = std::getenv("RL_CORE_GROUPS");
    if (groups_env != nullptr && std::strlen(groups_env) > 0) {
        // 显式指定各 CPU 的核心组 ID，逗号分隔，例如 "0,0,1,1,2,2"
        std::string cfg(groups_env);
        size_t cpu_idx = 0;
        size_t pos = 0;
        int max_group = 0;
        
        while (pos < cfg.size() && cpu_idx < NUM_CPUS) {
            size_t next = cfg.find(',', pos);
            if (next == std::string::npos) next = cfg.size();
            int gid = std::stoi(cfg.substr(pos, next - pos));
            cpu_to_group_[cpu_idx] = static_cast<uint32_t>(gid);
            max_group = std::max(max_group, gid);
            cpu_idx++;
            pos = next + 1;
        }
        
        num_groups_ = max_group + 1;
        
        for (; cpu_idx < NUM_CPUS; cpu_idx++) {
            cpu_to_group_[cpu_idx] = static_cast<uint32_t>(num_groups_ - 1);
        }
    } else {
        // 通过 RL_NUM_GROUPS 指定组数，自动均匀分配；默认 2 组
        const char* num_groups_env = std::getenv("RL_NUM_GROUPS");
        if (num_groups_env != nullptr && std::strlen(num_groups_env) > 0) {
            num_groups_ = std::max(1, std::stoi(std::string(num_groups_env)));
        } else {
            num_groups_ = std::min(2, static_cast<int>(NUM_CPUS));
        }
        
        for (size_t i = 0; i < NUM_CPUS; i++) {
            cpu_to_group_[i] = static_cast<uint32_t>(
                static_cast<size_t>(i) * static_cast<size_t>(num_groups_) / NUM_CPUS);
        }
    }
    
    // 构建 group_cpus_ 反向映射
    group_cpus_.resize(num_groups_);
    for (size_t i = 0; i < NUM_CPUS; i++) {
        group_cpus_[cpu_to_group_[i]].push_back(static_cast<uint32_t>(i));
    }
    
    for (int g = 0; g < num_groups_; g++) {
        if (group_cpus_[g].empty()) {
            std::cerr << "[RL_Partitioner] 警告: Group " << g << " 没有分配任何 CPU" << std::endl;
        }
    }
    
    if (num_groups_ > NUM_WAY) {
        std::cerr << "[RL_Partitioner] 警告: 核心组数 (" << num_groups_
                  << ") 大于 Way 数 (" << NUM_WAY
                  << ")，将截断核心组数" << std::endl;
        num_groups_ = static_cast<int>(NUM_WAY);
        group_cpus_.clear();
        group_cpus_.resize(num_groups_);
        for (size_t i = 0; i < NUM_CPUS; i++) {
            if (cpu_to_group_[i] >= static_cast<uint32_t>(num_groups_)) {
                cpu_to_group_[i] = static_cast<uint32_t>(num_groups_ - 1);
            }
            group_cpus_[cpu_to_group_[i]].push_back(static_cast<uint32_t>(i));
        }
    }
}

long rl_partitioner::get_way_start(uint32_t group_id) const
{
    long start = 0;
    for (uint32_t g = 0; g < group_id && g < static_cast<uint32_t>(num_groups_); g++) {
        start += partition_ways_[g];
    }
    return start;
}

std::vector<rl_partitioner::Action> rl_partitioner::select_best_actions_constrained(
    const std::vector<std::array<double, NUM_ACTIONS>>& q_values) const
{
    int n = num_groups_;
    int offset = n;
    int range = 2 * n + 1;
    long max_per_group = NUM_WAY - (num_groups_ - 1);
    
    // dp[i][s+offset] = 考虑前 i 个核心组、delta 总和为 s 时的最大总 Q 值
    std::vector<std::vector<double>> dp(n + 1, std::vector<double>(range, -1e18));
    std::vector<std::vector<int>> choice(n + 1, std::vector<int>(range, -1));
    
    dp[0][offset] = 0.0;
    
    for (int i = 0; i < n; i++) {
        for (int s = -n; s <= n; s++) {
            if (dp[i][s + offset] <= -1e17) continue;
            
            for (int a = 0; a < NUM_ACTIONS; a++) {
                int delta = (a == ACTION_INC) ? 1 : ((a == ACTION_DEC) ? -1 : 0);
                
                long new_p = partition_ways_[i] + delta;
                if (new_p < 1 || new_p > max_per_group) continue;
                
                int new_s = s + delta;
                if (new_s < -n || new_s > n) continue;
                
                double new_q = dp[i][s + offset] + q_values[i][a];
                if (new_q > dp[i + 1][new_s + offset]) {
                    dp[i + 1][new_s + offset] = new_q;
                    choice[i + 1][new_s + offset] = a;
                }
            }
        }
    }
    
    std::vector<Action> actions(n, ACTION_KEEP);
    
    if (dp[n][offset] <= -1e17) {
        return actions;
    }
    
    // 回溯恢复各组动作
    int current_sum = 0;
    for (int i = n; i >= 1; i--) {
        int a = choice[i][current_sum + offset];
        actions[i - 1] = static_cast<Action>(a);
        int delta = (a == ACTION_INC) ? 1 : ((a == ACTION_DEC) ? -1 : 0);
        current_sum -= delta;
    }
    
    return actions;
}

std::vector<rl_partitioner::Action> rl_partitioner::select_random_actions_constrained()
{
    std::vector<Action> actions(num_groups_, ACTION_KEEP);
    long max_per_group = NUM_WAY - (num_groups_ - 1);
    
    // 构建可 INC / 可 DEC 的候选组列表
    std::vector<int> inc_candidates, dec_candidates;
    for (int g = 0; g < num_groups_; g++) {
        if (partition_ways_[g] + 1 <= max_per_group) inc_candidates.push_back(g);
        if (partition_ways_[g] - 1 >= 1) dec_candidates.push_back(g);
    }
    
    int max_k = std::min({num_groups_ / 2,
                          static_cast<int>(inc_candidates.size()),
                          static_cast<int>(dec_candidates.size())});
    
    if (max_k == 0) return actions;
    
    std::uniform_int_distribution<int> k_dist(0, max_k);
    int k = k_dist(rng);
    if (k == 0) return actions;
    
    // 随机选 k 个组执行 INC
    std::shuffle(inc_candidates.begin(), inc_candidates.end(), rng);
    
    std::vector<bool> used(num_groups_, false);
    int inc_count = 0;
    for (int g : inc_candidates) {
        if (inc_count >= k) break;
        actions[g] = ACTION_INC;
        used[g] = true;
        inc_count++;
    }
    
    // 从未被选中 INC 的组中，随机选 k 个执行 DEC
    std::vector<int> remaining_dec;
    for (int g : dec_candidates) {
        if (!used[g]) remaining_dec.push_back(g);
    }
    std::shuffle(remaining_dec.begin(), remaining_dec.end(), rng);
    
    int dec_count = 0;
    for (int g : remaining_dec) {
        if (dec_count >= k) break;
        actions[g] = ACTION_DEC;
        dec_count++;
    }
    
    if (dec_count < k) {
        return std::vector<Action>(num_groups_, ACTION_KEEP);
    }
    
    return actions;
}

void rl_partitioner::maybe_update_rl()
{
    // 单组无需分区
    if (num_groups_ <= 1) return;
    
    uint64_t current_global_cycle = g_llc_stats.global_cycle;
    
    if (current_global_cycle - last_update_cycle < UPDATE_INTERVAL) {
        return;
    }
    
    uint64_t delta_cycles = current_global_cycle - last_rl_global_cycle;
    if (delta_cycles == 0) {
        return;
    }
    
    last_update_cycle = current_global_cycle;
    
    // 从共享内存同步最新权重
    if (shared_wt_.is_open()) {
        shared_wt_.read_lock();
        shared_wt_.snapshot(weights);
        shared_wt_.read_unlock();
        pending_deltas_.clear();
    }
    
    // ===== 1. 按核心组收集统计信息 =====
    std::vector<uint64_t> group_A(num_groups_, 0);
    std::vector<uint64_t> group_M(num_groups_, 0);
    std::vector<uint64_t> group_delta_instrs(num_groups_, 0);
    
    for (size_t i = 0; i < NUM_CPUS; i++) {
        uint32_t g = cpu_to_group_[i];
        group_A[g] += g_llc_stats.accesses[i] - last_accesses[i];
        group_M[g] += g_llc_stats.misses[i] - last_misses[i];
        group_delta_instrs[g] += g_llc_stats.retired_instructions[i] - last_retired_instructions[i];
        
        last_accesses[i] = g_llc_stats.accesses[i];
        last_misses[i] = g_llc_stats.misses[i];
        last_retired_instructions[i] = g_llc_stats.retired_instructions[i];
    }
    
    last_rl_global_cycle = current_global_cycle;
    
    // ===== 2. 计算各组和全局总量 =====
    uint64_t total_A = 0, total_M = 0, total_delta_instrs = 0;
    for (int g = 0; g < num_groups_; g++) {
        total_A += group_A[g];
        total_M += group_M[g];
        total_delta_instrs += group_delta_instrs[g];
    }
    
    // ===== 3. 构造 "one-vs-rest" 状态特征 =====
    std::vector<double> norm_a(num_groups_), norm_m(num_groups_);
    std::vector<double> norm_a_rest(num_groups_), norm_m_rest(num_groups_);
    std::vector<double> group_ipc(num_groups_), ipc_rest(num_groups_);
    
    for (int g = 0; g < num_groups_; g++) {
        norm_a[g] = log_normalize(group_A[g]);
        norm_m[g] = log_normalize(group_M[g]);
        norm_a_rest[g] = log_normalize(total_A - group_A[g]);
        norm_m_rest[g] = log_normalize(total_M - group_M[g]);
        group_ipc[g] = static_cast<double>(group_delta_instrs[g]) / static_cast<double>(delta_cycles);
        ipc_rest[g] = static_cast<double>(total_delta_instrs - group_delta_instrs[g]) / static_cast<double>(delta_cycles);
    }
    
    // ===== 4. 初始化基线 =====
    if (!baseline_initialized) {
        bool all_positive = true;
        for (int g = 0; g < num_groups_; g++) {
            if (group_ipc[g] <= 0) { all_positive = false; break; }
        }
        if (all_positive) {
            for (int g = 0; g < num_groups_; g++) {
                baseline_ipc_[g] = group_ipc[g];
                baseline_ipc_rest_[g] = ipc_rest[g];
                last_norm_a_[g] = norm_a[g];
                last_norm_m_[g] = norm_m[g];
                last_norm_a_rest_[g] = norm_a_rest[g];
                last_norm_m_rest_[g] = norm_m_rest[g];
                last_partition_[g] = partition_ways_[g];
                last_actions_[g] = ACTION_KEEP;
            }
            baseline_initialized = true;
            std::cout << "[RL_Partitioner] 基线初始化:";
            for (int g = 0; g < num_groups_; g++) {
                std::cout << " IPC_G" << g << "=" << group_ipc[g];
            }
            std::cout << std::endl;
        }
        return;
    }
    
    // ===== 5. 计算各组奖励并更新基线 =====
    std::vector<double> r(num_groups_), r_rest(num_groups_);
    for (int g = 0; g < num_groups_; g++) {
        r[g] = compute_reward(group_ipc[g], baseline_ipc_[g]);
        r_rest[g] = compute_reward(ipc_rest[g], baseline_ipc_rest_[g]);
        update_baseline(baseline_ipc_[g], group_ipc[g]);
        update_baseline(baseline_ipc_rest_[g], ipc_rest[g]);
    }
    
    // ===== 6. 从各组 one-vs-rest 视角训练 =====
    for (int g = 0; g < num_groups_; g++) {
        train(last_norm_a_[g], last_norm_m_[g],
              last_norm_a_rest_[g], last_norm_m_rest_[g],
              last_partition_[g], last_actions_[g],
              r[g], r_rest[g],
              norm_a[g], norm_m[g],
              norm_a_rest[g], norm_m_rest[g],
              partition_ways_[g]);
        
        // 存储两个视角的经验（单视角统一格式）
        store_experience({last_norm_a_[g], last_norm_m_[g], last_partition_[g],
                          last_actions_[g], r[g],
                          norm_a[g], norm_m[g], partition_ways_[g]});
        store_experience({last_norm_a_rest_[g], last_norm_m_rest_[g],
                          NUM_WAY - last_partition_[g],
                          flip_action(last_actions_[g]), r_rest[g],
                          norm_a_rest[g], norm_m_rest[g],
                          NUM_WAY - partition_ways_[g]});
    }
    
    replay_experiences();
    
    // 将本轮权重增量同步回共享内存
    if (shared_wt_.is_open() && !pending_deltas_.empty()) {
        shared_wt_.write_lock();
        shared_wt_.apply_deltas(pending_deltas_);
        shared_wt_.write_unlock();
        pending_deltas_.clear();
    }
    
    // ===== 7. 约束动作选取：最大化总 Q 值，保证 #INC = #DEC =====
    std::vector<std::array<double, NUM_ACTIONS>> q_values(num_groups_);
    for (int g = 0; g < num_groups_; g++) {
        for (int a = 0; a < NUM_ACTIONS; a++) {
            q_values[g][a] = compute_global_q(
                norm_a[g], norm_m[g], norm_a_rest[g], norm_m_rest[g],
                partition_ways_[g], static_cast<Action>(a));
        }
    }
    
    std::vector<Action> actions;
    if (uniform_dist(rng) < active_epsilon) {
        actions = select_random_actions_constrained();
    } else {
        actions = select_best_actions_constrained(q_values);
    }
    
    // ===== 8. 应用动作 =====
    apply_actions(actions);
    
    // ===== 9. 保存当前状态 =====
    for (int g = 0; g < num_groups_; g++) {
        last_norm_a_[g] = norm_a[g];
        last_norm_m_[g] = norm_m[g];
        last_norm_a_rest_[g] = norm_a_rest[g];
        last_norm_m_rest_[g] = norm_m_rest[g];
        last_partition_[g] = partition_ways_[g];
        last_actions_[g] = actions[g];
    }
    
    // ===== 10. 调试输出 =====
    static uint64_t update_count = 0;
    if (++update_count % 10 == 0) {
        std::cout << "[RL_Partitioner] 更新 #" << update_count << " |";
        for (int g = 0; g < num_groups_; g++) {
            std::cout << " G" << g << ":[P=" << partition_ways_[g]
                      << " A=" << group_A[g] << " M=" << group_M[g]
                      << " IPC=" << std::fixed << std::setprecision(3) << group_ipc[g]
                      << " R=" << std::setprecision(2) << r[g] << "]";
        }
        std::cout << std::endl;
    }
}
