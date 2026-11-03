# LiDAR hazard result, one message per scan (replaces lidar_live.csv).
uint8 SECTOR_NONE=0
uint8 SECTOR_FRONT=1
uint8 SECTOR_FRONT_RIGHT=2
uint8 SECTOR_RIGHT=3
uint8 SECTOR_BACK=4
uint8 SECTOR_LEFT=5
uint8 SECTOR_FRONT_LEFT=6

std_msgs/Header header
bool hazard                # min range in this scan < hazard_distance_m
bool confirmed             # N-of-M filter output
float32 min_range_m        # closest valid return in the scan
float32 approach_rate_mps  # positive = object closing in
uint8 sector_blocked       # closest blocked sector (SECTOR_* constant)
