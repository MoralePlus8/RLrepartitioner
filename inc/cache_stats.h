#ifndef CACHE_STATS_H
#define CACHE_STATS_H

#include <cstdint>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "channel.h"
#include "event_counter.h"

// Maximum number of CPUs supported for competition tracking
constexpr std::size_t MAX_CPUS_FOR_COMPETITION = 16;

struct cache_stats {
  std::string name;
  // prefetch stats
  uint64_t pf_requested = 0;
  uint64_t pf_issued = 0;
  uint64_t pf_useful = 0;
  uint64_t pf_useless = 0;
  uint64_t pf_fill = 0;

  champsim::stats::event_counter<std::pair<access_type, std::remove_cv_t<decltype(NUM_CPUS)>>> hits = {};
  champsim::stats::event_counter<std::pair<access_type, std::remove_cv_t<decltype(NUM_CPUS)>>> misses = {};
  champsim::stats::event_counter<std::pair<access_type, std::remove_cv_t<decltype(NUM_CPUS)>>> mshr_merge = {};
  champsim::stats::event_counter<std::pair<access_type, std::remove_cv_t<decltype(NUM_CPUS)>>> mshr_return = {};

  long total_miss_latency_cycles{};

  // Cache competition stats for LLC (only used when NUM_CPUS > 1)
  // evictions_caused[i] = number of cache lines belonging to OTHER cores that CPU i evicted
  // evicted_by_others[i] = number of cache lines belonging to CPU i that were evicted by OTHER cores
  std::vector<uint64_t> evictions_caused = std::vector<uint64_t>(MAX_CPUS_FOR_COMPETITION, 0);
  std::vector<uint64_t> evicted_by_others = std::vector<uint64_t>(MAX_CPUS_FOR_COMPETITION, 0);
};

cache_stats operator-(cache_stats lhs, cache_stats rhs);

#endif
