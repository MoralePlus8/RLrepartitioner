#include "partition.h"
#include <algorithm>
#include <cassert>
#include <iostream>
#include "champsim.h"

partition::partition(CACHE* cache) : partition(cache, cache->NUM_SET, cache->NUM_WAY) {}

partition::partition(CACHE* cache, long sets, long ways) 
    : replacement(cache), NUM_WAY(ways), 
      last_used_cycles(static_cast<std::size_t>(sets * ways), 0),
      partition_left_margins(NUM_CPUS + 1, 0)  // 初始化分区边界向量
{}

void partition::initialize_replacement(){
    // 平均分配每个CPU的way数量
    long init_partition_ways = NUM_WAY / static_cast<long>(NUM_CPUS);
    partition_left_margins[0] = 0;
    for (std::size_t i = 1; i < NUM_CPUS; i++) {
      partition_left_margins[i] = static_cast<long>(i) * init_partition_ways;
    }
    partition_left_margins[NUM_CPUS] = NUM_WAY;
    std::cout << "partition_strategy_initialized" << std::endl;
}
  

long partition::find_victim(uint32_t triggering_cpu, uint64_t instr_id, long set, const champsim::cache_block* current_set, champsim::address ip,
                            champsim::address full_addr, access_type type)
{
  // 计算当前CPU分区的起始和结束位置
  long partition_start = partition_left_margins[triggering_cpu];
  long partition_end = partition_left_margins[triggering_cpu + 1];
  
  // 首先在分区内查找无效way（解决初始填充阶段跨分区问题）
  for (long w = partition_start; w < partition_end; w++) {
    if (!current_set[w].valid) {
      return w;
    }
  }
  
  // 如果分区内没有无效way，使用LRU策略在分区内选择victim
  auto begin = std::next(std::begin(last_used_cycles), set * NUM_WAY + partition_start);
  auto end = std::next(std::begin(last_used_cycles), set * NUM_WAY + partition_end);

  // 在分区内找到LRU（最小cycle值）的way
  auto victim = std::min_element(begin, end);
  assert(begin <= victim);
  assert(victim < end);
  // 返回实际的 way 索引，而不是相对于分区起始位置的偏移量
  return partition_start + std::distance(begin, victim);
}

void partition::replacement_cache_fill(uint32_t triggering_cpu, long set, long way, champsim::address full_addr, champsim::address ip, champsim::address victim_addr,
                                      access_type type)
{
  // Mark the way as being used on the current cycle
  last_used_cycles.at((std::size_t)(set * NUM_WAY + way)) = cycle++;
}

void partition::update_replacement_state(uint32_t triggering_cpu, long set, long way, champsim::address full_addr, champsim::address ip,
                                        champsim::address victim_addr, access_type type, uint8_t hit)
{
  // Mark the way as being used on the current cycle
  if (hit && access_type{type} != access_type::WRITE)
  last_used_cycles.at((std::size_t)(set * NUM_WAY + way)) = cycle++;
}