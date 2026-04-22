from setuptools import setup
import os

package_name = 'yolo_seg_ros2'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/yolo_seg.launch.py']),
        ('share/' + package_name, ['yolo26n-seg.pt']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='your_name',
    maintainer_email='your_email@example.com',
    description='ROS2 YOLO segmentation node for RGB detection with depth distance',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'image_relay_node = yolo_seg_ros2.image_relay_node:main',
            'yolo_seg_crack_node = yolo_seg_ros2.yolo_seg_crack_node:main',
        ],
    },
)