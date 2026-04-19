from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='perception',
            executable='led_mock_publisher',
            name='led_mock_publisher',
            parameters=[{
                'noise_stddev': 0.01,
                'detection_distance': 10.0,
            }],
            output='screen',
        ),
    ])
