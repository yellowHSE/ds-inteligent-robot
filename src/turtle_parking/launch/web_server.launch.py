from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument

def generate_launch_description():
    port = LaunchConfiguration('port', default='8080')
    return LaunchDescription([
        DeclareLaunchArgument('port', default_value='8080'),
        Node(
            package='turtle_parking',
            executable='web_server',
            name='web_server',
            parameters=[{'port': port}],
            output='screen'
        )
    ])
