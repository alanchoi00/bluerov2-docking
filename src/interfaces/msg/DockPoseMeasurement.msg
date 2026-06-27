# Per-frame fused dock-pose measurement from aruco_fusion.
#
# Carries the fused pose+covariance plus how many markers backed this fix.
# The dock_pose_filter uses num_markers to refuse initialization on an
# under-determined (e.g. single-marker) measurement, which is subject to the
# planar PnP flip/tilt ambiguity.

std_msgs/Header header
geometry_msgs/PoseWithCovariance pose
uint8 num_markers
uint16[] marker_ids
