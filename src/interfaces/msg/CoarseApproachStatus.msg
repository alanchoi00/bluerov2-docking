# Coarse approach phase and error telemetry, consumed by the state machine.
uint8 APPROACHING = 0
uint8 AT_STANDOFF = 1
uint8 BLOCKED = 2  # dock pose not trusted (stale or no TF); holding

std_msgs/Header header

uint8 phase

float64 range_to_standoff_m
float64 axis_offset_m
float64 vertical_error_m
float64 heading_error_rad

bool within_position_tol
bool within_heading_tol
bool dock_healthy
bool ready_for_handoff
