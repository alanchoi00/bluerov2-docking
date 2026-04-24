# Dock pose filter health status
# 0=WARMING_UP, 1=HEALTHY, 2=DEGRADED, 3=STALE
uint8 WARMING_UP = 0
uint8 HEALTHY = 1
uint8 DEGRADED = 2
uint8 STALE = 3

std_msgs/Header header
uint8 status
float64 seconds_since_last_update
float64 position_std_m
