# Route containerized GL apps (gazebo, rviz2) onto the NVIDIA eGPU via PRIME offload.
#
# SOURCE this in the terminal you launch the sim from (do not execute it):
#   source .devcontainer/nvidia/egpu-render.sh        # offload ON  (use the eGPU)
#   source .devcontainer/nvidia/egpu-render.sh off     # offload OFF (host default GL)
#
# The vars only affect processes started from the shell that sourced this, which is
# why executing the file (a subshell) would do nothing. The provider index NVIDIA-G1
# is specific to the eGPU topology; recheck with `xrandr --listproviders` if it moves.

if [ "${1:-on}" = "off" ]; then
  unset __NV_PRIME_RENDER_OFFLOAD __NV_PRIME_RENDER_OFFLOAD_PROVIDER \
        __GLX_VENDOR_LIBRARY_NAME __VK_LAYER_NV_optimus
  echo "eGPU render offload OFF (host default GL path)"
else
  export __NV_PRIME_RENDER_OFFLOAD=1
  export __NV_PRIME_RENDER_OFFLOAD_PROVIDER=NVIDIA-G1
  export __GLX_VENDOR_LIBRARY_NAME=nvidia
  export __VK_LAYER_NV_optimus=NVIDIA_only
  echo "eGPU render offload ON (provider NVIDIA-G1)"
fi
