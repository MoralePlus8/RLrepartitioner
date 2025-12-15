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

// Global LLC competition stats structure for access from CPU heartbeat
struct llc_competition_stats {
  std::vector<uint64_t> evictions_caused = std::vector<uint64_t>(MAX_CPUS_FOR_COMPETITION, 0);
  std::vector<uint64_t> evicted_by_others = std::vector<uint64_t>(MAX_CPUS_FOR_COMPETITION, 0);
  // Last heartbeat values for computing per-period stats
  std::vector<uint64_t> last_heartbeat_evictions_caused = std::vector<uint64_t>(MAX_CPUS_FOR_COMPETITION, 0);
  std::vector<uint64_t> last_heartbeat_evicted_by_others = std::vector<uint64_t>(MAX_CPUS_FOR_COMPETITION, 0);
};

// Global LLC competition stats instance (defined in cache.cc)
extern llc_competition_stats g_llc_competition;

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
};

cache_stats operator-(cache_stats lhs, cache_stats rhs);

#endif
