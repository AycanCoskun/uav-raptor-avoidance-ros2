// Copyright 2026 Aycan Coskun
// SPDX-License-Identifier: AGPL-3.0-only

#include "raptor_lidar/hazard_detector.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace raptor_lidar
{

namespace
{

constexpr double kPi = 3.14159265358979323846;
constexpr double kSectorWidthDeg = 60.0;
constexpr std::size_t kNumSectors = 6;
// Minimum time span for the approach rate, same guard as state_machine_v3.py.
constexpr double kMinRateSpanS = 0.01;

}  // namespace

const char * sector_name(Sector sector)
{
  switch (sector) {
    case Sector::kFront:
      return "FRONT";
    case Sector::kFrontRight:
      return "FRONT_RIGHT";
    case Sector::kRight:
      return "RIGHT";
    case Sector::kBack:
      return "BACK";
    case Sector::kLeft:
      return "LEFT";
    case Sector::kFrontLeft:
      return "FRONT_LEFT";
    case Sector::kNone:
    default:
      return "NONE";
  }
}

Sector sector_for_angle(double angle_rad)
{
  // ROS angles are counter-clockwise, the original sector table is clockwise.
  double cw_deg = std::fmod(-angle_rad * 180.0 / kPi, 360.0);
  if (cw_deg < 0.0) {
    cw_deg += 360.0;
  }
  // Shift by half a sector so that FRONT [330, 30) starts at 0.
  const double shifted = std::fmod(cw_deg + kSectorWidthDeg / 2.0, 360.0);
  const std::size_t index =
    std::min(static_cast<std::size_t>(shifted / kSectorWidthDeg), kNumSectors - 1);
  return static_cast<Sector>(index + 1);
}

NofMFilter::NofMFilter(std::size_t n, std::size_t m)
: n_(n), m_(m)
{
  if (n_ == 0 || n_ > m_) {
    throw std::invalid_argument("N-of-M filter requires 1 <= N <= M");
  }
}

bool NofMFilter::update(bool hazard)
{
  window_.push_back(hazard);
  if (hazard) {
    ++count_;
  }
  if (window_.size() > m_) {
    if (window_.front()) {
      --count_;
    }
    window_.pop_front();
  }
  return window_.size() == m_ && count_ >= n_;
}

void NofMFilter::reset()
{
  window_.clear();
  count_ = 0;
}

HazardDetector::HazardDetector(const HazardConfig & config)
: config_(config), filter_(config.confirm_n, config.confirm_m)
{
  if (config_.hazard_distance_m <= 0.0 || config_.sector_distance_m <= 0.0) {
    throw std::invalid_argument("hazard and sector distances must be positive");
  }
  if (config_.approach_window < 2) {
    throw std::invalid_argument("approach_window must be at least 2 scans");
  }
}

const HazardConfig & HazardDetector::config() const
{
  return config_;
}

HazardResult HazardDetector::update(
  const std::vector<float> & ranges, double angle_min, double angle_increment,
  double range_min, double range_max, double stamp_s)
{
  constexpr double kInf = std::numeric_limits<double>::infinity();
  std::array<double, kNumSectors> sector_min;  // index = Sector value - 1
  sector_min.fill(kInf);
  double min_range = kInf;

  for (std::size_t i = 0; i < ranges.size(); ++i) {
    const double range = ranges[i];
    if (!std::isfinite(range) || range < range_min || range > range_max) {
      continue;  // no return, or outside the sensor limits
    }
    min_range = std::min(min_range, range);
    const Sector sector = sector_for_angle(angle_min + static_cast<double>(i) * angle_increment);
    double & sector_range = sector_min[static_cast<std::size_t>(sector) - 1];
    sector_range = std::min(sector_range, range);
  }

  HazardResult result;
  result.hazard = min_range < config_.hazard_distance_m;
  result.confirmed = filter_.update(result.hazard);
  result.min_range_m = static_cast<float>(min_range);
  result.approach_rate_mps = static_cast<float>(approach_rate(stamp_s, min_range));

  // Closest blocked sector; ties keep the first sector in FRONT..FRONT_LEFT order.
  double closest = config_.sector_distance_m;
  for (std::size_t i = 0; i < sector_min.size(); ++i) {
    if (sector_min[i] < closest) {
      closest = sector_min[i];
      result.sector_blocked = static_cast<Sector>(i + 1);
    }
  }
  return result;
}

double HazardDetector::approach_rate(double stamp_s, double min_range_m)
{
  // R(t) = (d(t - dt) - d(t)) / dt over the last `approach_window` scans (paper, Eq. 6).
  if (!std::isfinite(min_range_m)) {
    history_.clear();  // nothing in range, so there is no object to track
    return 0.0;
  }
  if (!history_.empty() && stamp_s < history_.back().first) {
    history_.clear();  // time jumped back (simulation reset)
  }
  history_.emplace_back(stamp_s, min_range_m);
  while (history_.size() > config_.approach_window) {
    history_.pop_front();
  }
  if (history_.size() < 2) {
    return 0.0;
  }
  const double span = history_.back().first - history_.front().first;
  if (span <= kMinRateSpanS) {
    return 0.0;
  }
  return (history_.front().second - history_.back().second) / span;
}

}  // namespace raptor_lidar
