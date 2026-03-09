/**
 * @file rl_partitioner.h
 * @brief 基于 Q-learning 和 Tile Coding 的 LLC 缓存分区策略
 * 
 * 核心设计：采用 "单权重表 + 对称状态映射 + 动作翻转" 策略。
 * 支持 N 个核心组的动态 LLC Way 分配（N ≥ 2），基于 "one-vs-rest" 思想
 * 将每个核心组与其余所有核心组组成二元状态，复用两组模型进行决策。
 */

#ifndef REPLACEMENT_RL_PARTITIONER_H
#define REPLACEMENT_RL_PARTITIONER_H

#include <array>
#include <vector>
#include <string>
#include <cstdint>
#include <random>
#include <cmath>

#include "cache.h"
#include "modules.h"
#include "champsim.h"
#include "shared_weight_table.h"

/**
 * @class rl_partitioner
 * @brief 基于强化学习的 LLC 缓存分区替换策略
 * 
 * 支持 N 个核心组的 LLC Way 动态分配。对于每个核心组 S_i，
 * 将其余 N-1 个核心组视为整体，构造 "one-vs-rest" 五元组状态空间：
 *   (A_i, A_rest, M_i, M_rest, P_i)
 * 在每个决策周期，通过约束优化选取使总 Q 值最大的动作组合，
 * 保证各核心组 Way 总数不变（#INC = #DEC）。
 */
class rl_partitioner : public champsim::modules::replacement
{
public:
    // ===== 动作空间定义 =====
    // 动作定义为当前核心组的 Way 变化量
    enum Action {
        ACTION_DEC = 0,   // 该核心组 Way 数减 1
        ACTION_KEEP = 1,  // 该核心组 Way 数不变
        ACTION_INC = 2    // 该核心组 Way 数加 1
    };
    static constexpr int NUM_ACTIONS = 3;

    // ===== Tile Coding 参数 =====
    static constexpr int NUM_TILINGS = 8;        // Tiling 层数
    static constexpr int GRID_SIZE_A = 10;       // A 维度网格大小
    static constexpr int GRID_SIZE_M = 10;       // M 维度网格大小
    static constexpr double LOG_MAX_VALUE = 12.0; // 对数变换后的最大值（用于归一化）
    
    // ===== Q-Learning 参数 =====
    static constexpr double LEARNING_RATE = 0.05;  // 学习率 alpha
    static constexpr double GAMMA = 0.95;          // 折扣因子
    static constexpr double EPSILON = 0.1;         // ε-greedy 探索率
    
    // ===== 奖励机制参数 =====
    static constexpr double REWARD_ALPHA = 0.1;    // EWMA 基线更新系数
    static constexpr double REWARD_BETA = 10.0;    // 奖励缩放因子
    static constexpr double REWARD_CLIP = 5.0;     // 奖励截断范围 [-REWARD_CLIP, REWARD_CLIP]
    
    // ===== 权重表参数 =====
    static constexpr size_t MEMORY_SIZE = 4096;    // 每个动作的权重向量大小
    static constexpr size_t TOTAL_WEIGHTS = MEMORY_SIZE * NUM_ACTIONS;  // 总权重数
    
    // ===== 更新周期参数 =====
    static constexpr uint64_t UPDATE_INTERVAL = 500000;  // 每隔多少 cycle 进行一次 RL 更新
    
    // ===== Experience Replay 参数 =====
    static constexpr size_t REPLAY_BUFFER_SIZE = 1024;  // 环形回放缓冲区容量
    static constexpr size_t REPLAY_BATCH_SIZE = 16;     // 每次回放采样的 mini-batch 大小
    
    // ===== 离线/在线模式参数 =====
    static constexpr double ONLINE_LEARNING_RATE = 0.02; // 在线微调学习率
    static constexpr double ONLINE_EPSILON = 0.05;       // 在线微调探索率
    
    // ===== 权重文件格式 =====
    static constexpr uint32_t WEIGHTS_FILE_MAGIC = 0x524C5057;  // "RLPW"
    static constexpr uint32_t WEIGHTS_FILE_VERSION = 1;

    // ===== 经验结构体（单视角统一格式，利用对称性）=====
    struct Experience {
        double norm_a;
        double norm_m;
        long p;
        Action action;
        double reward;
        double norm_a_next;
        double norm_m_next;
        long p_next;
    };

private:
    // ===== 缓存配置 =====
    long NUM_WAY;                                   // LLC 总 Way 数
    std::vector<uint64_t> last_used_cycles;        // LRU 时间戳
    uint64_t cycle = 0;                            // 内部周期计数器
    
    // ===== 多核心组配置 =====
    int num_groups_;                                   // 核心组数量
    std::vector<uint32_t> cpu_to_group_;               // CPU ID → 核心组 ID 映射
    std::vector<std::vector<uint32_t>> group_cpus_;    // 核心组 ID → CPU 列表
    
    // ===== 分区状态 =====
    std::vector<long> partition_ways_;                  // 各核心组分配的 Way 数量
    
    // ===== Q-Learning 权重表 =====
    // 单一权重表，两个组共用
    // 索引方式：weights[action * MEMORY_SIZE + tile_index]
    std::vector<double> weights;
    
    // ===== 运行时参数（支持在线模式自动调整）=====
    double active_lr;          // 实际使用的学习率
    double active_epsilon;     // 实际使用的探索率
    bool online_mode;          // 是否处于在线微调模式
    
    // ===== 状态缓存（上一时刻各核心组的 "one-vs-rest" 状态）=====
    std::vector<double> last_norm_a_;              // 各核心组上一时刻的归一化 Access
    std::vector<double> last_norm_m_;              // 各核心组上一时刻的归一化 Miss
    std::vector<double> last_norm_a_rest_;         // 各核心组对应 "rest" 的归一化 Access
    std::vector<double> last_norm_m_rest_;         // 各核心组对应 "rest" 的归一化 Miss
    std::vector<long> last_partition_;             // 各核心组上一时刻的 Way 分配
    std::vector<Action> last_actions_;             // 各核心组上一时刻的动作
    
    // ===== IPC 基线（EWMA）=====
    std::vector<double> baseline_ipc_;             // 各核心组的 IPC 基线
    std::vector<double> baseline_ipc_rest_;        // 各核心组对应 "rest" 的 IPC 基线
    
    // ===== 上一次 RL 更新的全局 cycle =====
    uint64_t last_update_cycle;
    
    // ===== 随机数生成器 =====
    std::mt19937 rng;
    std::uniform_real_distribution<double> uniform_dist;
    
    // ===== 统计信息（用于增量计算）=====
    std::vector<uint64_t> last_accesses;          // 上次更新时的 LLC Access
    std::vector<uint64_t> last_misses;            // 上次更新时的 LLC Miss
    
    // ===== IPC 增量计算所需的统计信息 =====
    // 这些字段用于在每个 RL 周期内独立计算 IPC，避免依赖 heartbeat_ipc 导致的延迟
    std::vector<uint64_t> last_retired_instructions;  // 上次 RL 更新时各 CPU 的已退休指令数
    uint64_t last_rl_global_cycle;                    // 上次 RL 更新时的全局 cycle
    
    // ===== 是否已初始化基线 =====
    bool baseline_initialized;
    
    // ===== Experience Replay 缓冲区 =====
    std::vector<Experience> replay_buffer;
    size_t replay_write_pos = 0;
    
    // ===== 共享内存权重表（多进程并行训练）=====
    SharedWeightTable shared_wt_;
    std::vector<std::pair<size_t, double>> pending_deltas_;

public:
    // ===== 构造函数 =====
    explicit rl_partitioner(CACHE* cache);
    rl_partitioner(CACHE* cache, long sets, long ways);
    
    // ===== Replacement 接口函数 =====
    void initialize_replacement();
    long find_victim(uint32_t triggering_cpu, uint64_t instr_id, long set, 
                    const champsim::cache_block* current_set, champsim::address ip,
                    champsim::address full_addr, access_type type);
    void update_replacement_state(uint32_t triggering_cpu, long set, long way, 
                                  champsim::address full_addr, champsim::address ip, 
                                  champsim::address victim_addr, access_type type, uint8_t hit);
    void replacement_cache_fill(uint32_t triggering_cpu, long set, long way, 
                                champsim::address full_addr, champsim::address ip, 
                                champsim::address victim_addr, access_type type);
    void replacement_final_stats();

private:
    // ===== Tile Coding 相关函数 =====
    
    /**
     * @brief 对数变换并归一化
     * @param value 原始值（Access 或 Miss 数）
     * @return 归一化后的值，范围 [0, 1]
     */
    double log_normalize(uint64_t value) const;
    
    /**
     * @brief 计算单个 Tiling 层的哈希索引
     * @param tiling_id Tiling 层编号
     * @param norm_a 归一化后的 A（Access）
     * @param norm_m 归一化后的 M（Miss）
     * @param p 当前分区值
     * @return 哈希索引
     */
    size_t compute_tile_index(int tiling_id, double norm_a, double norm_m, long p) const;
    
    /**
     * @brief 获取所有激活的 Tile 索引
     * @param norm_a 归一化后的 A
     * @param norm_m 归一化后的 M
     * @param p 当前分区值
     * @return 激活的 Tile 索引向量
     */
    std::vector<size_t> get_active_tiles(double norm_a, double norm_m, long p) const;
    
    // ===== Q-Learning 相关函数 =====
    
    /**
     * @brief 计算单个视角的 Q 值（权重之和）
     * @param tiles 激活的 Tile 索引
     * @param action 动作
     * @return Q 值
     */
    double compute_q_single(const std::vector<size_t>& tiles, Action action) const;
    
    /**
     * @brief 计算全局 Q 值（两个视角之和）
     * @param norm_a1, norm_m1 Group 1 的状态
     * @param norm_a2, norm_m2 Group 2 的状态
     * @param p 当前分区
     * @param action 动作
     * @return 全局 Q 值
     */
    double compute_global_q(double norm_a1, double norm_m1, 
                            double norm_a2, double norm_m2, 
                            long p, Action action) const;
    
    /**
     * @brief 翻转动作（用于对称映射）
     * @param action 原动作
     * @return 翻转后的动作
     */
    Action flip_action(Action action) const;
    
    /**
     * @brief 使用 ε-greedy 策略选择动作
     * @param norm_a1, norm_m1 Group 1 的状态
     * @param norm_a2, norm_m2 Group 2 的状态
     * @param p 当前分区
     * @return 选择的动作
     */
    Action select_action(double norm_a1, double norm_m1, 
                         double norm_a2, double norm_m2, long p);
    
    /**
     * @brief 获取最优动作（用于计算 TD target）
     * @return 最优动作
     */
    Action get_best_action(double norm_a1, double norm_m1, 
                           double norm_a2, double norm_m2, long p) const;
    
    // ===== 奖励计算 =====
    
    /**
     * @brief 计算相对于基线的奖励
     * @param current_ipc 当前 IPC
     * @param baseline_ipc 基线 IPC
     * @return 奖励值（已截断）
     */
    double compute_reward(double current_ipc, double baseline_ipc) const;
    
    /**
     * @brief 更新 EWMA 基线
     * @param baseline 基线引用
     * @param current_ipc 当前 IPC
     */
    void update_baseline(double& baseline, double current_ipc);
    
    // ===== 训练函数 =====
    
    /**
     * @brief 执行一次 Q-learning 更新（双重更新策略）
     * @param s1 Group 1 的当前状态 (A1, M1, P)
     * @param s2 Group 2 的当前状态 (A2, M2, N-P)
     * @param action 执行的动作
     * @param r1, r2 两个组的奖励
     * @param s1_next, s2_next 下一状态
     */
    void train(double norm_a1, double norm_m1, double norm_a2, double norm_m2, 
               long p, Action action, double r1, double r2,
               double norm_a1_next, double norm_m1_next, 
               double norm_a2_next, double norm_m2_next, long p_next);
    
    /**
     * @brief 更新权重（梯度下降）
     * @param tiles 激活的 Tiles
     * @param action 动作
     * @param td_error TD 误差
     */
    void update_weights(const std::vector<size_t>& tiles, Action action, double td_error);
    
    // ===== Experience Replay 相关函数 =====
    
    /**
     * @brief 将一条经验存入环形回放缓冲区
     */
    void store_experience(const Experience& exp);
    
    /**
     * @brief 从缓冲区随机采样 mini-batch 并执行训练
     */
    void replay_experiences();
    
    /**
     * @brief 对单条经验执行 Q-learning 更新（方案 A：单视角 Q 值选 best_next）
     */
    void train_single_experience(const Experience& exp);
    
    // ===== 权重持久化（离线预训练 + 在线微调）=====
    
    /**
     * @brief 将权重表保存到二进制文件（含超参数校验头）
     * @return 是否保存成功
     */
    bool save_weights(const std::string& path) const;
    
    /**
     * @brief 从二进制文件加载预训练权重（校验超参数一致性）
     * @return 是否加载成功
     */
    bool load_weights(const std::string& path);
    
    // ===== 辅助函数 =====
    
    /**
     * @brief 解析核心组配置（环境变量 RL_CORE_GROUPS / RL_NUM_GROUPS）
     */
    void parse_core_group_config();
    
    /**
     * @brief 获取核心组的 Way 起始索引
     * @param group_id 核心组 ID
     * @return 起始 Way 索引
     */
    long get_way_start(uint32_t group_id) const;
    
    /**
     * @brief 使用动态规划选取满足约束的最优动作组合
     * 
     * 约束：各核心组动作导致的 Way 变化之和为 0（#INC = #DEC）。
     * @param q_values q_values[i][a] = 核心组 i 执行动作 a 的 Q 值
     * @return 各核心组的最优动作
     */
    std::vector<Action> select_best_actions_constrained(
        const std::vector<std::array<double, NUM_ACTIONS>>& q_values) const;
    
    /**
     * @brief 生成满足约束的随机动作组合（用于 ε-greedy 探索）
     * @return 各核心组的随机动作
     */
    std::vector<Action> select_random_actions_constrained();
    
    /**
     * @brief 执行多组分区动作
     * @param actions 各核心组的动作向量
     */
    void apply_actions(const std::vector<Action>& actions);
    
    /**
     * @brief 简单的位混合哈希函数
     * @param x 输入值
     * @return 哈希值
     */
    static uint64_t hash_mix(uint64_t x);
    
    /**
     * @brief 获取 CPU 所属的核心组 ID
     * @param cpu CPU 编号
     * @return 核心组 ID
     */
    uint32_t get_group_id(uint32_t cpu) const;
    
    /**
     * @brief 周期性地检查是否需要执行 RL 更新
     */
    void maybe_update_rl();
};

#endif // REPLACEMENT_RL_PARTITIONER_H
