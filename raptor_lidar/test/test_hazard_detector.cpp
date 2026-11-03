// Copyright 2026 Aycan Coskun
// SPDX-License-Identifier: AGPL-3.0-only

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

#include "raptor_lidar/hazard_detector.hpp"

namespace
{

using raptor_lidar::HazardConfig;
using raptor_lidar::HazardDetector;
using raptor_lidar::HazardResult;
using raptor_lidar::NofMFilter;
using raptor_lidar::Sector;
using raptor_lidar::sector_for_angle;

constexpr double kPi = 3.14159265358979323846;
constexpr double kDeg = kPi / 180.0;
constexpr float kInf = std::numeric_limits<float>::infinity();

// Same beam layout as the Gazebo lidar: 360 beams, 1 degree apart, starting at -180 degrees.
constexpr std::size_t kBeams = 360;
constexpr double kAngleMin = -kPi;
constexpr double kAngleIncrement = kDeg;
constexpr double kRangeMin = 0.2;
constexpr double kRangeMax = 12.0;

std::vector<float> empty_scan()
{
  return std::vector<float>(kBeams, kInf);
}

// Index of the beam pointing at `angle_deg` (-180 <= angle_deg < 180).
std::size_t beam(double angle_deg)
{
  return static_cast<std::size_t>(std::lround(angle_deg + 180.0));
}

HazardResult feed(HazardDetector & detector, const std::vector<float> & ranges, double stamp_s)
{
  return detector.update(ranges, kAngleMin, kAngleIncrement, kRangeMin, kRangeMax, stamp_s);
}

}  // namespace

TEST(SectorForAngle, SectorCentres)
{
  EXPECT_EQ(sector_for_angle(0.0), Sector::kFront);
  EXPECT_EQ(sector_for_angle(60.0 * kDeg), Sector::kFrontLeft);
  EXPECT_EQ(sector_for_angle(120.0 * kDeg), Sector::kLeft);
  EXPECT_EQ(sector_for_angle(180.0 * kDeg), Sector::kBack);
  EXPECT_EQ(sector_for_angle(-120.0 * kDeg), Sector::kRight);
  EXPECT_EQ(sector_for_angle(-60.0 * kDeg), Sector::kFrontRight);
}

TEST(SectorForAngle, ConvertsClockwiseRplidarTable)
{
  // ROS angles are counter-clockwise, so the right-hand side has negative angles.
  EXPECT_EQ(sector_for_angle(15.0 * kDeg), Sector::kFront);
  EXPECT_EQ(sector_for_angle(-15.0 * kDeg), Sector::kFront);
  EXPECT_EQ(sector_for_angle(-89.0 * kDeg), Sector::kFrontRight);
  EXPECT_EQ(sector_for_angle(-91.0 * kDeg), Sector::kRight);
  EXPECT_EQ(sector_for_angle(89.0 * kDeg), Sector::kFrontLeft);
  EXPECT_EQ(sector_for_angle(91.0 * kDeg), Sector::kLeft);
}

TEST(SectorForAngle, WrapsAround)
{
  EXPECT_EQ(sector_for_angle(-kPi), Sector::kBack);
  EXPECT_EQ(sector_for_angle(2.0 * kPi), Sector::kFront);
  EXPECT_EQ(sector_for_angle(-2.0 * kPi + 60.0 * kDeg), Sector::kFrontLeft);
}

TEST(NofMFilter, WaitsForFullWindow)
{
  NofMFilter filter(2, 3);
  EXPECT_FALSE(filter.update(true));
  EXPECT_FALSE(filter.update(true));  // 2 of 2, window not full yet
  EXPECT_TRUE(filter.update(false));  // 2 of 3
}

TEST(NofMFilter, TwoOfThreeRejectsSingleScanSpike)
{
  NofMFilter filter(2, 3);
  for (int i = 0; i < 3; ++i) {
    filter.update(false);
  }
  EXPECT_FALSE(filter.update(true));   // [0 0 1] single-scan spike
  EXPECT_TRUE(filter.update(true));    // [0 1 1]
  EXPECT_TRUE(filter.update(false));   // [1 1 0] old hazard flags still in the window
  EXPECT_FALSE(filter.update(false));  // [1 0 0]
}

TEST(NofMFilter, ResetClearsWindow)
{
  NofMFilter filter(2, 3);
  for (int i = 0; i < 3; ++i) {
    filter.update(true);
  }
  filter.reset();
  EXPECT_FALSE(filter.update(true));
}

TEST(NofMFilter, RejectsInvalidConfig)
{
  EXPECT_THROW({NofMFilter filter(0, 3);}, std::invalid_argument);
  EXPECT_THROW({NofMFilter filter(4, 3);}, std::invalid_argument);
}

TEST(HazardDetector, EmptyScanIsClear)
{
  HazardDetector detector{HazardConfig{}};
  const HazardResult result = feed(detector, empty_scan(), 0.0);
  EXPECT_FALSE(result.hazard);
  EXPECT_FALSE(result.confirmed);
  EXPECT_TRUE(std::isinf(result.min_range_m));
  EXPECT_FLOAT_EQ(result.approach_rate_mps, 0.0F);
  EXPECT_EQ(result.sector_blocked, Sector::kNone);
}

TEST(HazardDetector, ObjectBeyondHazardDistanceIsNotHazard)
{
  // The test pillar in the Gazebo world is about 3.75 m in front of the drone.
  HazardDetector detector{HazardConfig{}};
  std::vector<float> scan = empty_scan();
  scan[beam(0.0)] = 3.75F;
  const HazardResult result = feed(detector, scan, 0.0);
  EXPECT_FALSE(result.hazard);
  EXPECT_FLOAT_EQ(result.min_range_m, 3.75F);
  EXPECT_EQ(result.sector_blocked, Sector::kNone);
}

TEST(HazardDetector, ObstacleAheadIsConfirmedAfterTwoOfThree)
{
  HazardDetector detector{HazardConfig{}};
  std::vector<float> scan = empty_scan();
  feed(detector, scan, 0.0);

  scan[beam(0.0)] = 1.0F;
  HazardResult result = feed(detector, scan, 0.1);
  EXPECT_TRUE(result.hazard);
  EXPECT_FALSE(result.confirmed);
  EXPECT_FLOAT_EQ(result.min_range_m, 1.0F);
  EXPECT_EQ(result.sector_blocked, Sector::kFront);

  result = feed(detector, scan, 0.2);
  EXPECT_TRUE(result.hazard);
  EXPECT_TRUE(result.confirmed);
}

TEST(HazardDetector, ClosestBlockedSectorWins)
{
  HazardDetector detector{HazardConfig{}};
  std::vector<float> scan = empty_scan();
  scan[beam(0.0)] = 2.0F;     // FRONT, beyond the sector distance
  scan[beam(120.0)] = 1.0F;   // LEFT
  scan[beam(-180.0)] = 0.5F;  // BACK
  const HazardResult result = feed(detector, scan, 0.0);
  EXPECT_TRUE(result.hazard);
  EXPECT_FLOAT_EQ(result.min_range_m, 0.5F);
  EXPECT_EQ(result.sector_blocked, Sector::kBack);
}

TEST(HazardDetector, IgnoresInvalidReturns)
{
  HazardDetector detector{HazardConfig{}};
  std::vector<float> scan = empty_scan();
  scan[beam(0.0)] = 0.1F;  // below range_min, e.g. the drone's own frame
  scan[beam(10.0)] = std::numeric_limits<float>::quiet_NaN();
  scan[beam(20.0)] = 15.0F;  // above range_max
  scan[beam(30.0)] = -kInf;
  const HazardResult result = feed(detector, scan, 0.0);
  EXPECT_FALSE(result.hazard);
  EXPECT_TRUE(std::isinf(result.min_range_m));
  EXPECT_EQ(result.sector_blocked, Sector::kNone);
}

TEST(HazardDetector, ApproachRateIsPositiveWhenClosing)
{
  HazardDetector detector{HazardConfig{}};
  std::vector<float> scan = empty_scan();
  HazardResult result;
  for (int i = 0; i < 5; ++i) {
    scan[beam(0.0)] = static_cast<float>(3.0 - 0.1 * i);  // closing at 1 m/s
    result = feed(detector, scan, 0.1 * i);
  }
  EXPECT_NEAR(result.approach_rate_mps, 1.0, 1e-3);
}

TEST(HazardDetector, ApproachRateIsNegativeWhenMovingAway)
{
  HazardDetector detector{HazardConfig{}};
  std::vector<float> scan = empty_scan();
  HazardResult result;
  for (int i = 0; i < 5; ++i) {
    scan[beam(0.0)] = static_cast<float>(1.0 + 0.05 * i);  // moving away at 0.5 m/s
    result = feed(detector, scan, 0.1 * i);
  }
  EXPECT_NEAR(result.approach_rate_mps, -0.5, 1e-3);
}

TEST(HazardDetector, ApproachRateUsesSlidingWindow)
{
  HazardConfig config;
  config.approach_window = 3;
  HazardDetector detector{config};
  std::vector<float> scan = empty_scan();
  const float ranges[] = {3.0F, 2.0F, 2.0F, 2.0F, 2.0F};
  HazardResult result;
  for (int i = 0; i < 5; ++i) {
    scan[beam(0.0)] = ranges[i];
    result = feed(detector, scan, 0.1 * i);
  }
  EXPECT_FLOAT_EQ(result.approach_rate_mps, 0.0F);  // the 3.0 m sample has left the window
}

TEST(HazardDetector, ApproachRateResetsWhenTargetIsLost)
{
  HazardDetector detector{HazardConfig{}};
  std::vector<float> scan = empty_scan();
  scan[beam(0.0)] = 3.0F;
  feed(detector, scan, 0.0);
  scan[beam(0.0)] = 2.0F;
  EXPECT_GT(feed(detector, scan, 0.1).approach_rate_mps, 0.0F);

  EXPECT_FLOAT_EQ(feed(detector, empty_scan(), 0.2).approach_rate_mps, 0.0F);
  scan[beam(0.0)] = 1.0F;
  EXPECT_FLOAT_EQ(feed(detector, scan, 0.3).approach_rate_mps, 0.0F);  // one sample only
}

TEST(HazardDetector, ApproachRateResetsWhenTimeJumpsBack)
{
  HazardDetector detector{HazardConfig{}};
  std::vector<float> scan = empty_scan();
  scan[beam(0.0)] = 3.0F;
  feed(detector, scan, 10.0);
  scan[beam(0.0)] = 2.0F;
  EXPECT_FLOAT_EQ(feed(detector, scan, 0.0).approach_rate_mps, 0.0F);
}

TEST(HazardDetector, RejectsInvalidConfig)
{
  HazardConfig bad_window;
  bad_window.approach_window = 1;
  EXPECT_THROW({HazardDetector detector{bad_window};}, std::invalid_argument);

  HazardConfig bad_distance;
  bad_distance.hazard_distance_m = 0.0;
  EXPECT_THROW({HazardDetector detector{bad_distance};}, std::invalid_argument);

  HazardConfig bad_filter;
  bad_filter.confirm_n = 4;
  EXPECT_THROW({HazardDetector detector{bad_filter};}, std::invalid_argument);
}
