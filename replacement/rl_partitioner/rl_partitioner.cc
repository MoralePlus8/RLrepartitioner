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
      partition_p(ways / 2),  // 初始平均分配
      weights(TOTAL_WEIGHTS, 0.0),  // 初始化权重为 0
      active_lr(LEARNING_RATE),
      active_epsilon(EPSILON),
      online_mode(false),
      last_A1(0), last_A2(0), last_M1(0), last_M2(0),
      last_P(ways / 2),
      last_action(ACTION_KEEP),
      baseline_ipc_g1(0.0),
      baseline_ipc_g2(0.0),
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
    std::cout << "[RL_Partitioner] 初始化: NUM_WAY=" << NUM_WAY 
              << ", 初始分区 P=" << partition_p << std::endl;
}

// ============================================================================
// Replacement 接口函数实现
// ============================================================================

void rl_partitioner::initialize_replacement()
{
    partition_p = NUM_WAY / 2;
    last_P = partition_p;
    
    // 检测环境变量，尝试加载预训练权重
    const char* load_path_env = std::getenv("RL_WEIGHTS_LOAD");
    bool weights_loaded = false;
    if (load_path_env != nullptr && std::strlen(load_path_env) > 0) {
        weights_loaded = load_weights(std::string(load_path_env));
    }
    
    // 检测在线模式：加载了预训练权重 且 设置了 RL_ONLINE_MODE=1
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
    
    std::cout << "[RL_Partitioner] 替换策略初始化完成" << std::endl;
    std::cout << "  - 总 Way 数: " << NUM_WAY << std::endl;
    std::cout << "  - Group 1 Way 数: " << partition_p << std::endl;
    std::cout << "  - Group 2 Way 数: " << (NUM_WAY - partition_p) << std::endl;
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
    // 周期性检查是否需要 RL 更新
    maybe_update_rl();
    
    // 确定 CPU 所属的组
    uint32_t group_id = get_group_id(triggering_cpu);
    
    // 根据分区确定搜索范围
    long partition_start, partition_end;
    if (group_id == 0) {
        // Group 1: Way [0, partition_p)
        partition_start = 0;
        partition_end = partition_p;
    } else {
        // Group 2: Way [partition_p, NUM_WAY)
        partition_start = partition_p;
        partition_end = NUM_WAY;
    }
    
    // 确保分区范围有效
    if (partition_start >= partition_end) {
        // 如果分区为空，使用整个 cache
        partition_start = 0;
        partition_end = NUM_WAY;
    }
    
    // 首先在分区内查找无效 way
    for (long w = partition_start; w < partition_end; w++) {
        if (!current_set[w].valid) {
            return w;
        }
    }
    
    // 使用 LRU 策略在分区内选择 victim
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
    std::cout << "  - 最终分区 P (Group 1 Way 数): " << partition_p << std::endl;
    std::cout << "  - Group 2 Way 数: " << (NUM_WAY - partition_p) << std::endl;
    std::cout << "  - 最终 IPC 基线 Group 1: " << std::fixed << std::setprecision(4) << baseline_ipc_g1 << std::endl;
    std::cout << "  - 最终 IPC 基线 Group 2: " << std::fixed << std::setprecision(4) << baseline_ipc_g2 << std::endl;
    std::cout << "  - 实际参数: LR=" << active_lr << ", EPSILON=" << active_epsilon << std::endl;
    std::cout << "  - Experience Replay 缓冲区使用量: " << replay_buffer.size() 
              << " / " << REPLAY_BUFFER_SIZE << std::endl;
    
    // 检查环境变量，保存权重
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
    
    // 梯度下降：w = w + α * td_error / |tiles|
    double step = active_lr * td_error / static_cast<double>(tiles.size());
    
    for (size_t tile : tiles) {
        weights[action_offset + tile] += step;
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
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs.is_open()) {
        std::cerr << "[RL_Partitioner] 无法打开文件进行写入: " << path << std::endl;
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
        return false;
    }
    
    std::cout << "[RL_Partitioner] 权重已保存到: " << path
              << " (" << sizeof(header) + TOTAL_WEIGHTS * sizeof(double) << " bytes)" << std::endl;
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

void rl_partitioner::apply_action(Action action)
{
    long old_p = partition_p;
    
    switch (action) {
        case ACTION_INC:
            partition_p = std::min(partition_p + 1, NUM_WAY - 1);
            break;
        case ACTION_DEC:
            partition_p = std::max(partition_p - 1, 1L);
            break;
        case ACTION_KEEP:
        default:
            // 不改变
            break;
    }
    
    if (old_p != partition_p) {
        std::cout << "[RL_Partitioner] 分区调整: P " << old_p << " -> " << partition_p 
                  << " (动作: " << (action == ACTION_INC ? "INC" : (action == ACTION_DEC ? "DEC" : "KEEP")) << ")" 
                  << std::endl;
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
    // 将 CPU 分为两组：
    // Group 0: CPU 0, 2, 4, ...
    // Group 1: CPU 1, 3, 5, ...
    // 或者简单地根据 CPU 数量平分：
    // 如果有 N 个 CPU，前 N/2 个属于 Group 0，后 N/2 个属于 Group 1
    
    // 这里采用简单的平分策略
    if (NUM_CPUS <= 1) {
        return 0;
    }
    return (cpu < NUM_CPUS / 2) ? 0 : 1;
}

void rl_partitioner::maybe_update_rl()
{
    // 获取当前全局 cycle（从 g_llc_stats 获取，由 champsim.cc 每周期更新）
    uint64_t current_global_cycle = g_llc_stats.global_cycle;
    
    // 检查是否达到更新间隔
    if (current_global_cycle - last_update_cycle < UPDATE_INTERVAL) {
        return;
    }
    
    // 计算周期增量（用于 IPC 计算）
    uint64_t delta_cycles = current_global_cycle - last_rl_global_cycle;
    if (delta_cycles == 0) {
        return;  // 避免除零
    }
    
    // 更新时间戳
    last_update_cycle = current_global_cycle;
    
    // 收集当前统计信息
    // Group 1: CPU 0 到 NUM_CPUS/2 - 1
    // Group 2: CPU NUM_CPUS/2 到 NUM_CPUS - 1
    
    uint64_t A1 = 0, A2 = 0;  // LLC Access
    uint64_t M1 = 0, M2 = 0;  // LLC Miss
    uint64_t delta_instrs1 = 0, delta_instrs2 = 0;  // 指令增量
    
    size_t half_cpus = (NUM_CPUS + 1) / 2;
    
    // 累加 Group 1 的统计
    for (size_t i = 0; i < half_cpus && i < NUM_CPUS; i++) {
        // 计算 LLC 访问和缺失增量
        uint64_t delta_access = g_llc_stats.accesses[i] - last_accesses[i];
        uint64_t delta_miss = g_llc_stats.misses[i] - last_misses[i];
        
        A1 += delta_access;
        M1 += delta_miss;
        
        // 计算指令增量（用于实时 IPC 计算）
        delta_instrs1 += g_llc_stats.retired_instructions[i] - last_retired_instructions[i];
        
        // 更新历史值
        last_accesses[i] = g_llc_stats.accesses[i];
        last_misses[i] = g_llc_stats.misses[i];
        last_retired_instructions[i] = g_llc_stats.retired_instructions[i];
    }
    
    // 累加 Group 2 的统计
    for (size_t i = half_cpus; i < NUM_CPUS; i++) {
        // 计算 LLC 访问和缺失增量
        uint64_t delta_access = g_llc_stats.accesses[i] - last_accesses[i];
        uint64_t delta_miss = g_llc_stats.misses[i] - last_misses[i];
        
        A2 += delta_access;
        M2 += delta_miss;
        
        // 计算指令增量（用于实时 IPC 计算）
        delta_instrs2 += g_llc_stats.retired_instructions[i] - last_retired_instructions[i];
        
        // 更新历史值
        last_accesses[i] = g_llc_stats.accesses[i];
        last_misses[i] = g_llc_stats.misses[i];
        last_retired_instructions[i] = g_llc_stats.retired_instructions[i];
    }
    
    // 更新上次 RL 全局 cycle
    last_rl_global_cycle = current_global_cycle;
    
    // 计算实时 IPC（基于当前 RL 周期内的增量，而非 heartbeat_ipc）
    // 这确保了每次 RL 更新都使用该周期内的真实性能数据
    double ipc1 = static_cast<double>(delta_instrs1) / static_cast<double>(delta_cycles);
    double ipc2 = static_cast<double>(delta_instrs2) / static_cast<double>(delta_cycles);
    
    // 归一化状态特征
    double norm_a1 = log_normalize(A1);
    double norm_m1 = log_normalize(M1);
    double norm_a2 = log_normalize(A2);
    double norm_m2 = log_normalize(M2);
    
    // 初始化基线
    if (!baseline_initialized) {
        if (ipc1 > 0 && ipc2 > 0) {
            baseline_ipc_g1 = ipc1;
            baseline_ipc_g2 = ipc2;
            baseline_initialized = true;
            
            // 初始化上一状态
            last_A1 = norm_a1;
            last_M1 = norm_m1;
            last_A2 = norm_a2;
            last_M2 = norm_m2;
            last_P = partition_p;
            last_action = ACTION_KEEP;
            
            std::cout << "[RL_Partitioner] 基线初始化: IPC_G1=" << ipc1 << ", IPC_G2=" << ipc2 << std::endl;
        }
        return;
    }
    
    // 计算奖励
    double r1 = compute_reward(ipc1, baseline_ipc_g1);
    double r2 = compute_reward(ipc2, baseline_ipc_g2);
    
    // 更新基线
    update_baseline(baseline_ipc_g1, ipc1);
    update_baseline(baseline_ipc_g2, ipc2);
    
    // 执行训练（双重更新）
    train(last_A1, last_M1, last_A2, last_M2, last_P, last_action, r1, r2,
          norm_a1, norm_m1, norm_a2, norm_m2, partition_p);
    
    // 将两个视角的经验存入回放缓冲区（统一为单视角格式）
    store_experience({last_A1, last_M1, last_P, last_action, r1,
                      norm_a1, norm_m1, partition_p});
    store_experience({last_A2, last_M2, NUM_WAY - last_P, flip_action(last_action), r2,
                      norm_a2, norm_m2, NUM_WAY - partition_p});
    
    // 从缓冲区随机采样 mini-batch 进行经验回放
    replay_experiences();
    
    // 选择下一个动作
    Action next_action = select_action(norm_a1, norm_m1, norm_a2, norm_m2, partition_p);
    
    // 应用动作
    apply_action(next_action);
    
    // 保存当前状态
    last_A1 = norm_a1;
    last_M1 = norm_m1;
    last_A2 = norm_a2;
    last_M2 = norm_m2;
    last_P = partition_p;
    last_action = next_action;
    
    // 输出调试信息（可选）
    static uint64_t update_count = 0;
    if (++update_count % 10 == 0) {  // 每 10 次更新输出一次
        std::cout << "[RL_Partitioner] 更新 #" << update_count 
                  << " | P=" << partition_p 
                  << " | A1=" << A1 << " M1=" << M1 
                  << " | A2=" << A2 << " M2=" << M2
                  << " | IPC1=" << std::fixed << std::setprecision(3) << ipc1 
                  << " IPC2=" << ipc2
                  << " | R1=" << std::setprecision(2) << r1 
                  << " R2=" << r2
                  << std::endl;
    }
}
