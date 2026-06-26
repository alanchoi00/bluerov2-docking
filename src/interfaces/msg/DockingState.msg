# Current docking FSM state. Doubles as the active-phase gate signal that
# phase controllers subscribe to (publish cmd_vel only when state matches).
uint8 COARSE = 0
uint8 FINE = 1
uint8 DOCKED = 2

std_msgs/Header header

uint8 state
string label
