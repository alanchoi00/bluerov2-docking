from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='perception',
            executable='led_mock_publisher',
            name='led_mock_publisher',
            parameters=[{
                'noise_stddev_m': 0.01,
                'detection_distance_m': 10.0,
            }],
            output='screen',
        ),
    ])
