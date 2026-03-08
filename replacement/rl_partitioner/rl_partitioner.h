/**
 * @file rl_partitioner.h
 * @brief 基于 Q-learning 和 Tile Coding 的 LLC 缓存分区策略
 * 
 * 核心设计：采用 "单权重表 + 对称状态映射 + 动作翻转" 策略，
 * 处理两个核心组（Group 1 和 Group 2）之间的 LLC Way 分配问题。
 */

#ifndef REPLACEMENT_RL_PARTITIONER_H
#define REPLACEMENT_RL_PARTITIONER_H

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
 * 使用 Q-learning + Tile Coding 来动态调整 Group 1 和 Group 2 的 LLC Way 分配。
 * 状态空间：(A1, A2, M1, M2, P)
 *   - A1, A2: 两个组的 LLC Access 数
 *   - M1, M2: 两个组的 LLC Miss 数
 *   - P: 分配给 Group 1 的 Way 数量
 */
class rl_partitioner : public champsim::modules::replacement
{
public:
    // ===== 动作空间定义 =====
    // 动作定义为 Group 1 的 Way 变化量
    enum Action {
        ACTION_DEC = 0,   // P 减 1 (Group 1 减, Group 2 增)
        ACTION_KEEP = 1,  // P 不变
        ACTION_INC = 2    // P 加 1 (Group 1 增, Group 2 减)
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
    
    // ===== 分区状态 =====
    long partition_p;                              // 分配给 Group 1 的 Way 数量
    
    // ===== Q-Learning 权重表 =====
    // 单一权重表，两个组共用
    // 索引方式：weights[action * MEMORY_SIZE + tile_index]
    std::vector<double> weights;
    
    // ===== 运行时参数（支持在线模式自动调整）=====
    double active_lr;          // 实际使用的学习率
    double active_epsilon;     // 实际使用的探索率
    bool online_mode;          // 是否处于在线微调模式
    
    // ===== 状态缓存（上一时刻的状态）=====
    double last_A1, last_A2, last_M1, last_M2;     // 上一时刻的 Access 和 Miss
    long last_P;                                   // 上一时刻的分区设置
    Action last_action;                            // 上一时刻的动作
    
    // ===== IPC 基线（EWMA）=====
    double baseline_ipc_g1;                        // Group 1 的 IPC 基线
    double baseline_ipc_g2;                        // Group 2 的 IPC 基线
    
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
     * @brief 执行分区动作
     * @param action 动作
     */
    void apply_action(Action action);
    
    /**
     * @brief 简单的位混合哈希函数
     * @param x 输入值
     * @return 哈希值
     */
    static uint64_t hash_mix(uint64_t x);
    
    /**
     * @brief 获取 CPU 所属的组 ID
     * @param cpu CPU 编号
     * @return 组 ID (0 或 1)
     */
    uint32_t get_group_id(uint32_t cpu) const;
    
    /**
     * @brief 周期性地检查是否需要执行 RL 更新
     */
    void maybe_update_rl();
};

#endif // REPLACEMENT_RL_PARTITIONER_H
