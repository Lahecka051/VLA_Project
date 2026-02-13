from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
import os
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    ldlidar_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('ldlidar_stl_ros2'),
                         'launch', 'stl27l.launch.py')
        )
    )

    return LaunchDescription([
        ldlidar_launch,
        Node(package='jetson_detection', executable='camera_yolo_node', name='camera_yolo_node'),
        Node(package='jetson_detection', executable='imu_node', name='imu_node'),
        Node(package='jetson_detection', executable='fusion_node', name='fusion_node'),
        Node(package='jetson_detection', executable='viewer_node', name='viewer_node'),
    ])
