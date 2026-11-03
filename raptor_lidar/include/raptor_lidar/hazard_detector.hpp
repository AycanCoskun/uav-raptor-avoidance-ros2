// Copyright 2026 Aycan Coskun
// SPDX-License-Identifier: AGPL-3.0-only

#ifndef RAPTOR_LIDAR__HAZARD_DETECTOR_HPP_
#define RAPTOR_LIDAR__HAZARD_DETECTOR_HPP_

#include <cstddef>
#include <cstdint>
#include <deque>
#include <limits>
#include <utility>
#include <vector>

namespace raptor_lidar
{

// Sector identifiers. The values match the SECTOR_* constants in
// raptor_interfaces/msg/LidarHazard.msg.
enum class Sector : uint8_t
{
  kNone = 0,
  kFront = 1,
  kFrontRight = 2,
  kRight = 3,
  kBack = 4,
  kLeft = 5,
  kFrontLeft = 6,
};

// Human-readable sector name, e.g. "FRONT_LEFT".
const char * sector_name(Sector sector);

// Map a LaserScan beam angle (radians, counter-clockwise from +X, the front of the drone)
// to one of six 60-degree sectors. The sector table is the one used by
// lidar_live_writer_v2.py, which worked with RPLIDAR angles (degrees, clockwise from the front):
//   FRONT [330, 30), FRONT_RIGHT [30, 90), RIGHT [90, 150),
//   BACK [150, 210), LEFT [210, 270), FRONT_LEFT [270, 330).
Sector sector_for_angle(double angle_rad);

// Sliding-window N-of-M confirmation filter. Returns true once the window holds M flags and
// at least N of them are set.
class NofMFilter
{
public:
  NofMFilter(std::size_t n, std::size_t m);

  bool update(bool hazard);
  void reset();

private:
  std::size_t n_;
  std::size_t m_;
  std::deque<bool> window_;
  std::size_t count_{0};
};

// Default values are the final values from the paper (converted to metres).
struct HazardConfig
{
  double hazard_distance_m{1.2};    // scan is a hazard if its closest return is nearer than this
  double sector_distance_m{1.2};    // sector is blocked if its closest return is nearer than this
  std::size_t confirm_n{2};         // N of the N-of-M filter
  std::size_t confirm_m{3};         // M of the N-of-M filter
  std::size_t approach_window{10};  // number of scans used for the approach rate (1 s at 10 Hz)
};

struct HazardResult
{
  bool hazard{false};
  bool confirmed{false};
  float min_range_m{std::numeric_limits<float>::infinity()};  // +inf when nothing is in range
  float approach_rate_mps{0.0F};                               // positive = object closing in
  Sector sector_blocked{Sector::kNone};
};

// ROS-independent port of lidar_live_writer_v2.py: per-scan hazard flag, N-of-M confirmation,
// closest blocked sector and approach rate. The approach rate used to be computed in
// state_machine_v3.py; it now lives here so every LidarHazard message carries it.
class HazardDetector
{
public:
  explicit HazardDetector(const HazardConfig & config);

  // Process one scan. `ranges[i]` is measured at `angle_min + i * angle_increment`.
  // Returns outside [range_min, range_max], NaN and +/-inf are ignored.
  HazardResult update(
    const std::vector<float> & ranges, double angle_min, double angle_increment,
    double range_min, double range_max, double stamp_s);

  const HazardConfig & config() const;

private:
  double approach_rate(double stamp_s, double min_range_m);

  HazardConfig config_;
  NofMFilter filter_;
  std::deque<std::pair<double, double>> history_;  // (stamp_s, min_range_m)
};

}  // namespace raptor_lidar

#endif  // RAPTOR_LIDAR__HAZARD_DETECTOR_HPP_
