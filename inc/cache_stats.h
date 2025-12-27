#ifndef CACHE_STATS_H
#define CACHE_STATS_H

#include <cstdint>
#include <fstream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "channel.h"
#include "event_counter.h"

// Maximum number of CPUs supported for competition tracking
constexpr std::size_t MAX_CPUS_FOR_COMPETITION = 16;

// CSV export file path (can be set before simulation starts)
extern std::string g_llc_stats_csv_path;
// Flag to track if CSV header has been written
extern bool g_llc_stats_csv_header_written;

// Global LLC stats structure for access from CPU heartbeat
struct llc_stats {
  // Cache competition stats
  std::vector<uint64_t> evictions_caused = std::vector<uint64_t>(MAX_CPUS_FOR_COMPETITION, 0);
  std::vector<uint64_t> evicted_by_others = std::vector<uint64_t>(MAX_CPUS_FOR_COMPETITION, 0);
  // Last heartbeat values for computing per-period competition stats
  std::vector<uint64_t> last_heartbeat_evictions_caused = std::vector<uint64_t>(MAX_CPUS_FOR_COMPETITION, 0);
  std::vector<uint64_t> last_heartbeat_evicted_by_others = std::vector<uint64_t>(MAX_CPUS_FOR_COMPETITION, 0);

  // LLC access stats per CPU
  std::vector<uint64_t> accesses = std::vector<uint64_t>(MAX_CPUS_FOR_COMPETITION, 0);  // Total LLC accesses per CPU
  std::vector<uint64_t> misses = std::vector<uint64_t>(MAX_CPUS_FOR_COMPETITION, 0);    // LLC misses per CPU
  // Last heartbeat values for computing per-period access stats
  std::vector<uint64_t> last_heartbeat_accesses = std::vector<uint64_t>(MAX_CPUS_FOR_COMPETITION, 0);
  std::vector<uint64_t> last_heartbeat_misses = std::vector<uint64_t>(MAX_CPUS_FOR_COMPETITION, 0);

  // Cache line lifetime statistics (cycles until eviction)
  std::vector<uint64_t> total_lifetime_cycles = std::vector<uint64_t>(MAX_CPUS_FOR_COMPETITION, 0);  // Sum of all lifetime cycles per CPU
  std::vector<uint64_t> eviction_count = std::vector<uint64_t>(MAX_CPUS_FOR_COMPETITION, 0);          // Number of evictions per CPU (被驱逐的次数)
  // Last heartbeat values for computing per-period lifetime stats
  std::vector<uint64_t> last_heartbeat_total_lifetime_cycles = std::vector<uint64_t>(MAX_CPUS_FOR_COMPETITION, 0);
  std::vector<uint64_t> last_heartbeat_eviction_count = std::vector<uint64_t>(MAX_CPUS_FOR_COMPETITION, 0);

  // Total evictions caused by each CPU (includes both self and other cores' cache lines)
  std::vector<uint64_t> total_evictions_caused = std::vector<uint64_t>(MAX_CPUS_FOR_COMPETITION, 0);  // 每个核心引起的所有驱逐次数
  std::vector<uint64_t> last_heartbeat_total_evictions_caused = std::vector<uint64_t>(MAX_CPUS_FOR_COMPETITION, 0);

  // Way occupancy statistics - tracks how many ways each CPU occupies
  // These are snapshots that can be updated periodically
  std::vector<uint64_t> way_occupancy_samples = std::vector<uint64_t>(MAX_CPUS_FOR_COMPETITION, 0);  // Sum of way counts in samples
  uint64_t way_occupancy_sample_count = 0;  // Number of samples taken
  // Last heartbeat values for computing per-period way occupancy
  std::vector<uint64_t> last_heartbeat_way_occupancy_samples = std::vector<uint64_t>(MAX_CPUS_FOR_COMPETITION, 0);
  uint64_t last_heartbeat_way_occupancy_sample_count = 0;

  // Interim lifetime statistics - computed at each heartbeat by traversing all cache lines
  // This measures how long current cache lines have been residing without being evicted
  // Unlike eviction-based lifetime, this captures cache lines that are still in cache
  std::vector<uint64_t> heartbeat_interim_lifetime_sum = std::vector<uint64_t>(MAX_CPUS_FOR_COMPETITION, 0);  // Sum of interim lifetimes at last heartbeat
  std::vector<uint64_t> heartbeat_interim_line_count = std::vector<uint64_t>(MAX_CPUS_FOR_COMPETITION, 0);    // Number of cache lines at last heartbeat

  // Fill count statistics for Little's Law calculation
  // Little's Law: W = L / λ, where W = avg lifetime, L = avg occupancy, λ = arrival rate
  // λ = fill_count / period_cycles, so W = L × period_cycles / fill_count
  std::vector<uint64_t> fill_count = std::vector<uint64_t>(MAX_CPUS_FOR_COMPETITION, 0);  // Total cache line fills per CPU
  std::vector<uint64_t> last_heartbeat_fill_count = std::vector<uint64_t>(MAX_CPUS_FOR_COMPETITION, 0);
};

// Global LLC stats instance (defined in cache.cc)
extern llc_stats g_llc_stats;

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
