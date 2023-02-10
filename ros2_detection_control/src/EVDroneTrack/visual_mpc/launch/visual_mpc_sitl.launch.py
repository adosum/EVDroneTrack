from launch_ros.actions import Node
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('command_namespace', default_value='CommandCenter',
                              description='Command Center Namespace'),
        DeclareLaunchArgument('leader_namespace', default_value='Drone1',
                              description='Leader Drone Namespace'),
        DeclareLaunchArgument('follower_namespace', default_value='Drone2',
                              description='Follower Drone Namespace'),
        DeclareLaunchArgument('trajectory', default_value='single_drone_tracking_2',
                              description='Leader Drone Trajectory'),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([get_package_share_directory('ls2n_drone_simulation'),
                                          '/drones_sitl.launch.py']),
            launch_arguments={'param_file': 'two_drone_params.yaml',
                              'gz_world': 'empty'}.items()
        ),
        Node(package='ls2n_drone_command_center', executable='fake_joystick',
             output='screen', namespace=LaunchConfiguration('command_namespace'),
        ),
        Node(package='ls2n_drone_command_center', executable='trajectory_publisher',
             output='screen', namespace=LaunchConfiguration('leader_namespace'),
             parameters=[{'trajectory': LaunchConfiguration('trajectory')}]
        ),
        Node(package='visual_mpc', executable='joystick_comm_splitter',
             output='screen', parameters=[
                 {'leader_namespace': LaunchConfiguration('leader_namespace')},
                 {'follower_namespace': LaunchConfiguration('follower_namespace')},
                 {'command_namespace': LaunchConfiguration('command_namespace')}]
        ),
        Node(package='visual_mpc', executable='visual_simulator',
             output='screen', parameters=[
                 {'target_namespace': LaunchConfiguration('leader_namespace')},
                 {'drone_namespace': LaunchConfiguration('follower_namespace')},
                 {'command_namespace': LaunchConfiguration('command_namespace')}]

        ),
        Node(package='visual_mpc', executable='visual_mpc_node',
             output='screen', namespace=LaunchConfiguration('follower_namespace'),
        ),
    ])
