/**
 * @file ucp.h
 * @brief Utility-Based Cache Partitioning (UCP) — Qureshi & Patt, MICRO 2006
 *
 * 核心组件：
 *   1. UMON (Utility Monitor)：采样 ATD + 命中计数器，构建效用曲线
 *   2. Lookahead 贪心分区算法：基于边际效用最大化分配 Way
 *
 * 分区执行：各 CPU 只在自己分配的 Way 范围内做 LRU 替换。
 */

#ifndef REPLACEMENT_UCP_H
#define REPLACEMENT_UCP_H

#include <cstdint>
#include <vector>

#include "cache.h"
#include "modules.h"
#include "champsim.h"

class ucp : public champsim::modules::replacement
{
    // ===== ATD 采样参数 =====
    static constexpr int SAMPLED_SET_DIVISOR = 32;  // 采样比例 1/32
    static constexpr uint64_t SAMPLE_PRIME = 443;   // 用于采样 set 选择的质数

    // ===== 分区更新周期 =====
    static constexpr uint64_t REPARTITION_INTERVAL = 5000000;  // 每 5M cycle 重新分区

    // ===== 缓存配置 =====
    long NUM_SET;
    long NUM_WAY;
    uint64_t cycle = 0;

    // ===== LRU 替换用时间戳 =====
    std::vector<uint64_t> last_used_cycles;

    // ===== 分区状态：partition_ways[cpu] = 分配给该 CPU 的 way 数 =====
    std::vector<long> partition_ways;

    // ===== UMON: Auxiliary Tag Directory =====
    // ATD 条目
    struct ATDEntry {
        bool valid = false;
        uint64_t tag = 0;
    };

    // 每个 CPU 的 ATD 存储：atd[cpu][local_sample_idx][way]
    // local_sample_idx 是采样 set 的本地索引（0-based 连续编号）
    std::vector<std::vector<std::vector<ATDEntry>>> atd;

    // ATD 的 LRU 顺序：atd_lru[cpu][local_sample_idx][way] = 时间戳
    std::vector<std::vector<std::vector<uint64_t>>> atd_lru;
    uint64_t atd_cycle = 0;

    // 采样 set 列表
    std::vector<long> sampled_sets;
    // 全局 set → 本地采样索引 的映射（-1 表示非采样 set）
    std::vector<long> set_to_sample_idx;

    // 命中计数器：hit_counters[cpu][way_position]
    // hit_counters[cpu][k] = 当访问命中 ATD 的 LRU 位置 k 时累加
    std::vector<std::vector<uint64_t>> hit_counters;

    // ===== 分区更新追踪 =====
    uint64_t last_repartition_cycle = 0;

public:
    explicit ucp(CACHE* cache);
    ucp(CACHE* cache, long sets, long ways);

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
    bool is_sampled_set(long set) const;
    void atd_access(uint32_t cpu, long sample_idx, uint64_t tag);
    void build_utility_and_repartition();
    void lookahead_partition(const std::vector<std::vector<uint64_t>>& utility);
    long get_way_start(uint32_t cpu) const;
    void maybe_repartition();
};

#endif // REPLACEMENT_UCP_H
