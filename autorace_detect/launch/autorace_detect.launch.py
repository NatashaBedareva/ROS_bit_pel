from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    follow_path = Node(
            package='autorace_detect',
            executable='follow_path',
            name='follow_path',
            output='screen',
            emulate_tty=True,
        )
    return LaunchDescription([
        follow_path
    ])