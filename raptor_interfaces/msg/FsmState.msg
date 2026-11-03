# State machine output, one message per tick (replaces state_log_v3.csv).
std_msgs/Header header
string state            # IDLE, OBSTACLE_AVOID, EARLY_WARNING, THREAT_CONFIRMED, ESCAPE, RTL
string layer_active     # NONE, OBSTACLE, RAPTOR
float32 fusion_score
float32 proximity_score
string command          # last MAVLink command sent through MAVROS
string note
