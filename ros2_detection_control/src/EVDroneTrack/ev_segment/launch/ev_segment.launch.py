import os
import launch
import launch_ros.actions
from launch.actions import IncludeLaunchDescription
from ament_index_python.packages import get_package_share_directory
from launch.launch_description_sources import PythonLaunchDescriptionSource


config = os.path.join(
    get_package_share_directory('ev_segment'),
    'config',
    'ev_config.yaml'
)

def generate_launch_description():
    ros_share_dir = get_package_share_directory('ev_segment')

    node = launch_ros.actions.Node(
        package="ev_segment", executable="ev_seg",
        parameters=[config],
    )

    image_view = launch_ros.actions.Node(
        package="rqt_image_view", executable="rqt_image_view",
        remappings=[("image", "event_frame")]
    )

    return launch.LaunchDescription([
        node,
        image_view
    ])
