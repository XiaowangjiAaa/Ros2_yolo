from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([

        # 1️⃣ 图像中转节点
        Node(
            package='yolo_seg_ros2',
            executable='image_relay_node',
            name='image_relay_node',
            output='screen',
            parameters=[{
                'rgb_input': '/ascamera/camera_publisher/rgb0/image',
                'depth_input': '/ascamera/camera_publisher/depth0/image_raw',
                'rgb_output': '/rgb_relay',
                'depth_output': '/depth_relay',
            }]
        ),

        # 2️⃣ YOLO 检测节点
        Node(
            package='yolo_seg_ros2',
            executable='yolo_seg_crack_node',
            name='yolo_seg_crack_node',
            output='screen',
            parameters=[{
                'input_topic': '/rgb_relay',
                'depth_topic': '/depth_relay',
                'output_topic': '/yolo_result',
                'model_path': 'yolo26n-seg.pt',
                'conf_threshold': 0.25,
                'imgsz': 640,
                'process_fps': 5.0,
            }]
        ),
    ])