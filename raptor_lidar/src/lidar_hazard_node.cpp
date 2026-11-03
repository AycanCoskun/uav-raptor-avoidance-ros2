// Copyright 2026 Aycan Coskun
// SPDX-License-Identifier: AGPL-3.0-only

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"

#include "raptor_interfaces/msg/lidar_hazard.hpp"
#include "raptor_lidar/hazard_detector.hpp"

namespace raptor_lidar
{

using LidarHazard = raptor_interfaces::msg::LidarHazard;

// The Sector enum must stay in sync with the message constants.
static_assert(static_cast<uint8_t>(Sector::kNone) == LidarHazard::SECTOR_NONE, "sector mismatch");
static_assert(static_cast<uint8_t>(Sector::kFront) == LidarHazard::SECTOR_FRONT, "sector mismatch");
static_assert(
  static_cast<uint8_t>(Sector::kFrontRight) == LidarHazard::SECTOR_FRONT_RIGHT, "sector mismatch");
static_assert(static_cast<uint8_t>(Sector::kRight) == LidarHazard::SECTOR_RIGHT, "sector mismatch");
static_assert(static_cast<uint8_t>(Sector::kBack) == LidarHazard::SECTOR_BACK, "sector mismatch");
static_assert(static_cast<uint8_t>(Sector::kLeft) == LidarHazard::SECTOR_LEFT, "sector mismatch");
static_assert(
  static_cast<uint8_t>(Sector::kFrontLeft) == LidarHazard::SECTOR_FRONT_LEFT, "sector mismatch");

// Subscribes to a 2D LaserScan and publishes one LidarHazard message per scan
// (replaces the lidar_live.csv file of the original system).
class LidarHazardNode : public rclcpp::Node
{
public:
  explicit LidarHazardNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions())
  : rclcpp::Node("lidar_hazard_node", options),
    detector_(load_config())
  {
    publisher_ = create_publisher<LidarHazard>("lidar/hazard", 10);
    subscription_ = create_subscription<sensor_msgs::msg::LaserScan>(
      "scan", rclcpp::SensorDataQoS(),
      [this](const sensor_msgs::msg::LaserScan::ConstSharedPtr msg) {on_scan(*msg);});

    const HazardConfig & config = detector_.config();
    RCLCPP_INFO(
      get_logger(),
      "hazard_distance=%.2f m, sector_distance=%.2f m, N=%zu, M=%zu, approach_window=%zu scans",
      config.hazard_distance_m, config.sector_distance_m, config.confirm_n, config.confirm_m,
      config.approach_window);
  }

private:
  HazardConfig load_config()
  {
    HazardConfig config;
    config.hazard_distance_m = declare_parameter("hazard_distance_m", config.hazard_distance_m);
    config.sector_distance_m = declare_parameter("sector_distance_m", config.sector_distance_m);
    config.confirm_n = declare_count("confirm_n", config.confirm_n);
    config.confirm_m = declare_count("confirm_m", config.confirm_m);
    config.approach_window = declare_count("approach_window", config.approach_window);
    return config;
  }

  std::size_t declare_count(const std::string & name, std::size_t default_value)
  {
    const int64_t value = declare_parameter(name, static_cast<int64_t>(default_value));
    if (value < 1) {
      throw std::invalid_argument("parameter '" + name + "' must be >= 1");
    }
    return static_cast<std::size_t>(value);
  }

  void on_scan(const sensor_msgs::msg::LaserScan & scan)
  {
    const double stamp_s = rclcpp::Time(scan.header.stamp).seconds();
    const HazardResult result = detector_.update(
      scan.ranges, scan.angle_min, scan.angle_increment, scan.range_min, scan.range_max, stamp_s);

    LidarHazard msg;
    msg.header = scan.header;
    msg.hazard = result.hazard;
    msg.confirmed = result.confirmed;
    msg.min_range_m = result.min_range_m;
    msg.approach_rate_mps = result.approach_rate_mps;
    msg.sector_blocked = static_cast<uint8_t>(result.sector_blocked);
    publisher_->publish(msg);

    RCLCPP_INFO_THROTTLE(
      get_logger(), *get_clock(), 2000,
      "min_range=%.2f m sector=%s hazard=%d confirmed=%d approach_rate=%.2f m/s",
      result.min_range_m, sector_name(result.sector_blocked), result.hazard, result.confirmed,
      result.approach_rate_mps);
  }

  HazardDetector detector_;
  rclcpp::Publisher<LidarHazard>::SharedPtr publisher_;
  rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr subscription_;
};

}  // namespace raptor_lidar

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<raptor_lidar::LidarHazardNode>());
  rclcpp::shutdown();
  return 0;
}
