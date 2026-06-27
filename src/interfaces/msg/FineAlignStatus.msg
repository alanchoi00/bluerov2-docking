# Fine alignment phase and error telemetry, consumed by the docking FSM.
uint8 ALIGNING = 0
uint8 SEATED = 1
uint8 BLOCKED = 2  # dock pose not trusted (stale or no TF); holding

std_msgs/Header header

uint8 phase

float64 range_to_dock_m
float64 lateral_error_m
float64 vertical_error_m
float64 yaw_error_rad

bool aligned           # lateral/vertical/yaw within align tolerance (surge-enable)
bool seated            # terminal, debounced -> FSM reads this
bool dock_healthy
