/**
 * @file ucp.cc
 * @brief Utility-Based Cache Partitioning (UCP) 实现
 *
 * 参考论文：Qureshi & Patt, "Utility-Based Cache Partitioning:
 * A Low-Overhead, High-Performance, Runtime Mechanism to Partition
 * Shared Caches", MICRO 2006.
 */

#include "ucp.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <numeric>
#include "cache_stats.h"

// ============================================================================
// 构造函数
// ============================================================================

ucp::ucp(CACHE* cache) : ucp(cache, cache->NUM_SET, cache->NUM_WAY) {}

ucp::ucp(CACHE* cache, long sets, long ways)
    : replacement(cache),
      NUM_SET(sets),
      NUM_WAY(ways),
      last_used_cycles(static_cast<std::size_t>(sets * ways), 0),
      partition_ways(NUM_CPUS, 0)
{
    std::cout << "[UCP] 构造: NUM_SET=" << NUM_SET
              << ", NUM_WAY=" << NUM_WAY << std::endl;
}

// ============================================================================
// 初始化
// ============================================================================

void ucp::initialize_replacement()
{
    // 1. 均匀分配 Way
    long base_ways = NUM_WAY / static_cast<long>(NUM_CPUS);
    long remainder = NUM_WAY % static_cast<long>(NUM_CPUS);
    for (std::size_t i = 0; i < NUM_CPUS; i++) {
        partition_ways[i] = base_ways + (static_cast<long>(i) < remainder ? 1 : 0);
    }

    // 2. 构建采样 set 列表
    sampled_sets.clear();
    set_to_sample_idx.assign(NUM_SET, -1);
    for (long s = 0; s < NUM_SET; s++) {
        if ((static_cast<uint64_t>(s) % SAMPLE_PRIME) % SAMPLED_SET_DIVISOR == 0) {
            set_to_sample_idx[s] = static_cast<long>(sampled_sets.size());
            sampled_sets.push_back(s);
        }
    }
    long num_samples = static_cast<long>(sampled_sets.size());

    // 3. 初始化 ATD 和 LRU 时间戳
    atd.resize(NUM_CPUS);
    atd_lru.resize(NUM_CPUS);
    for (std::size_t cpu = 0; cpu < NUM_CPUS; cpu++) {
        atd[cpu].assign(num_samples, std::vector<ATDEntry>(NUM_WAY));
        atd_lru[cpu].assign(num_samples, std::vector<uint64_t>(NUM_WAY, 0));
    }

    // 4. 初始化命中计数器
    hit_counters.assign(NUM_CPUS, std::vector<uint64_t>(NUM_WAY, 0));

    last_repartition_cycle = 0;

    std::cout << "[UCP] 替换策略初始化完成" << std::endl;
    std::cout << "  - 总 Way 数: " << NUM_WAY << std::endl;
    std::cout << "  - CPU 数: " << NUM_CPUS << std::endl;
    std::cout << "  - 采样 Set 数: " << num_samples
              << " / " << NUM_SET
              << " (1/" << SAMPLED_SET_DIVISOR << ")" << std::endl;
    std::cout << "  - 分区更新周期: " << REPARTITION_INTERVAL << " cycles" << std::endl;
    std::cout << "  - 初始分区: ";
    for (std::size_t i = 0; i < NUM_CPUS; i++) {
        std::cout << "CPU" << i << "=" << partition_ways[i] << " ";
    }
    std::cout << std::endl;
}

// ============================================================================
// 采样 set 判断
// ============================================================================

bool ucp::is_sampled_set(long set) const
{
    return set >= 0 && set < NUM_SET && set_to_sample_idx[set] >= 0;
}

// ============================================================================
// ATD 访问：在采样 set 的辅助标签目录中模拟 LRU 访问
// ============================================================================

void ucp::atd_access(uint32_t cpu, long sample_idx, uint64_t tag)
{
    auto& entries = atd[cpu][sample_idx];
    auto& lru_ts = atd_lru[cpu][sample_idx];

    // 查找 tag 是否命中 ATD
    long hit_way = -1;
    for (long w = 0; w < NUM_WAY; w++) {
        if (entries[w].valid && entries[w].tag == tag) {
            hit_way = w;
            break;
        }
    }

    if (hit_way >= 0) {
        // ATD 命中：计算该条目在 LRU 栈中的位置（0 = MRU, NUM_WAY-1 = LRU）
        uint64_t hit_ts = lru_ts[hit_way];
        long stack_position = 0;
        for (long w = 0; w < NUM_WAY; w++) {
            if (w != hit_way && entries[w].valid && lru_ts[w] > hit_ts) {
                stack_position++;
            }
        }
        // stack_position 表示"如果只有 (stack_position+1) 个 way 就能命中"
        if (stack_position < NUM_WAY) {
            hit_counters[cpu][stack_position]++;
        }
        // 更新为 MRU
        lru_ts[hit_way] = ++atd_cycle;
    } else {
        // ATD 未命中：找到无效或 LRU 条目替换
        long victim_way = -1;
        uint64_t min_ts = UINT64_MAX;

        for (long w = 0; w < NUM_WAY; w++) {
            if (!entries[w].valid) {
                victim_way = w;
                break;
            }
            if (lru_ts[w] < min_ts) {
                min_ts = lru_ts[w];
                victim_way = w;
            }
        }

        entries[victim_way].valid = true;
        entries[victim_way].tag = tag;
        lru_ts[victim_way] = ++atd_cycle;
    }
}

// ============================================================================
// 构建效用曲线并重新分区
// ============================================================================

void ucp::build_utility_and_repartition()
{
    // utility[cpu][w] = 分配 (w+1) 个 way 时的预期命中数
    // utility[cpu][w] = sum(hit_counters[cpu][0..w])
    std::vector<std::vector<uint64_t>> utility(NUM_CPUS, std::vector<uint64_t>(NUM_WAY, 0));

    for (std::size_t cpu = 0; cpu < NUM_CPUS; cpu++) {
        utility[cpu][0] = hit_counters[cpu][0];
        for (long w = 1; w < NUM_WAY; w++) {
            utility[cpu][w] = utility[cpu][w - 1] + hit_counters[cpu][w];
        }
    }

    // 打印效用曲线摘要
    std::cout << "[UCP] 效用曲线 (采样命中数):" << std::endl;
    for (std::size_t cpu = 0; cpu < NUM_CPUS; cpu++) {
        std::cout << "  CPU" << cpu << ": U(1)=" << utility[cpu][0];
        if (NUM_WAY > 1)
            std::cout << " U(" << NUM_WAY / 2 << ")=" << utility[cpu][NUM_WAY / 2 - 1];
        std::cout << " U(" << NUM_WAY << ")=" << utility[cpu][NUM_WAY - 1]
                  << std::endl;
    }

    lookahead_partition(utility);

    // 清零命中计数器
    for (std::size_t cpu = 0; cpu < NUM_CPUS; cpu++) {
        std::fill(hit_counters[cpu].begin(), hit_counters[cpu].end(), 0);
    }
}

// ============================================================================
// Lookahead 贪心分区算法
// ============================================================================

void ucp::lookahead_partition(const std::vector<std::vector<uint64_t>>& utility)
{
    // 每个 CPU 至少分配 1 个 way
    std::vector<long> alloc(NUM_CPUS, 1);
    long remaining = NUM_WAY - static_cast<long>(NUM_CPUS);

    // 贪心分配剩余 way：每次分配给边际效用最大的 CPU
    while (remaining > 0) {
        long best_cpu = -1;
        uint64_t best_marginal = 0;

        for (std::size_t cpu = 0; cpu < NUM_CPUS; cpu++) {
            if (alloc[cpu] >= NUM_WAY) continue;
            // 边际效用 = U(alloc+1) - U(alloc)
            uint64_t marginal = utility[cpu][alloc[cpu]] - utility[cpu][alloc[cpu] - 1];
            if (best_cpu < 0 || marginal > best_marginal) {
                best_marginal = marginal;
                best_cpu = static_cast<long>(cpu);
            }
        }

        if (best_cpu < 0) break;
        alloc[best_cpu]++;
        remaining--;
    }

    // 更新分区
    std::cout << "[UCP] 重新分区: ";
    for (std::size_t i = 0; i < NUM_CPUS; i++) {
        partition_ways[i] = alloc[i];
        std::cout << "CPU" << i << "=" << alloc[i] << " ";
    }
    std::cout << std::endl;
}

// ============================================================================
// 获取 CPU 的 Way 起始索引（连续分区）
// ============================================================================

long ucp::get_way_start(uint32_t cpu) const
{
    long start = 0;
    for (uint32_t i = 0; i < cpu; i++) {
        start += partition_ways[i];
    }
    return start;
}

// ============================================================================
// 周期性检查是否需要重新分区
// ============================================================================

void ucp::maybe_repartition()
{
    uint64_t global_cycle = g_llc_stats.global_cycle;
    if (global_cycle - last_repartition_cycle >= REPARTITION_INTERVAL) {
        last_repartition_cycle = global_cycle;
        build_utility_and_repartition();
    }
}

// ============================================================================
// find_victim：在 CPU 的分区范围内选择 LRU victim
// ============================================================================

long ucp::find_victim(uint32_t triggering_cpu, uint64_t instr_id, long set,
                      const champsim::cache_block* current_set, champsim::address ip,
                      champsim::address full_addr, access_type type)
{
    maybe_repartition();

    long partition_start = get_way_start(triggering_cpu);
    long partition_end = partition_start + partition_ways[triggering_cpu];

    // 安全检查：分区范围异常时退化为全 set 搜索
    if (partition_start >= partition_end || partition_end > NUM_WAY) {
        partition_start = 0;
        partition_end = NUM_WAY;
    }

    // 优先选择无效 way
    for (long w = partition_start; w < partition_end; w++) {
        if (!current_set[w].valid) {
            return w;
        }
    }

    // 分区内 LRU 替换
    auto begin = std::next(std::begin(last_used_cycles), set * NUM_WAY + partition_start);
    auto end = std::next(std::begin(last_used_cycles), set * NUM_WAY + partition_end);

    auto victim = std::min_element(begin, end);
    assert(begin <= victim);
    assert(victim < end);

    return partition_start + std::distance(begin, victim);
}

// ============================================================================
// update_replacement_state：命中时更新 LRU + ATD
// ============================================================================

void ucp::update_replacement_state(uint32_t triggering_cpu, long set, long way,
                                   champsim::address full_addr, champsim::address ip,
                                   champsim::address victim_addr, access_type type, uint8_t hit)
{
    if (hit && access_type{type} != access_type::WRITE) {
        last_used_cycles.at(static_cast<std::size_t>(set * NUM_WAY + way)) = cycle++;
    }

    // 对采样 set 更新 ATD
    if (is_sampled_set(set)) {
        long sample_idx = set_to_sample_idx[set];
        uint64_t tag = full_addr.to<uint64_t>() >> 6;  // 去掉块内偏移
        atd_access(triggering_cpu, sample_idx, tag);
    }
}

// ============================================================================
// replacement_cache_fill：填充时更新 LRU + ATD
// ============================================================================

void ucp::replacement_cache_fill(uint32_t triggering_cpu, long set, long way,
                                 champsim::address full_addr, champsim::address ip,
                                 champsim::address victim_addr, access_type type)
{
    last_used_cycles.at(static_cast<std::size_t>(set * NUM_WAY + way)) = cycle++;

    // 对采样 set 也更新 ATD（miss 路径）
    if (is_sampled_set(set)) {
        long sample_idx = set_to_sample_idx[set];
        uint64_t tag = full_addr.to<uint64_t>() >> 6;
        atd_access(triggering_cpu, sample_idx, tag);
    }
}

// ============================================================================
// replacement_final_stats：输出最终统计
// ============================================================================

void ucp::replacement_final_stats()
{
    std::cout << "\n[UCP] 最终统计信息:" << std::endl;
    std::cout << "  - CPU 数: " << NUM_CPUS << std::endl;
    std::cout << "  - 最终分区: ";
    for (std::size_t i = 0; i < NUM_CPUS; i++) {
        std::cout << "CPU" << i << "=" << partition_ways[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "  - 采样 Set 数: " << sampled_sets.size() << std::endl;
    std::cout << "  - 分区更新周期: " << REPARTITION_INTERVAL << " cycles" << std::endl;
}
