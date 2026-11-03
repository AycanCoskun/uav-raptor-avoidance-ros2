# Bird detection result, one message per camera frame (replaces diag_live.csv).
std_msgs/Header header
bool hazard            # target class detected with conf >= conf_threshold in this frame
bool confirmed         # N-of-M filter output
float32 max_conf       # highest target-class confidence (0..1)
float32 cx_norm        # bbox centre x, normalised (0 = left, 1 = right)
float32 cy_norm        # bbox centre y, normalised (0 = top, 1 = bottom)
float32 inference_ms   # model inference + decode time
