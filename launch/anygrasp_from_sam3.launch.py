from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='anygrasp_sam3_ros2',
            executable='anygrasp_from_topic_node',
            name='anygrasp_from_topic_node',
            output='screen',
            parameters=[{
                'sdk_root': '/home/jwg/anygrasp_sdk/grasp_detection',
                'checkpoint_path': '/home/jwg/anygrasp_sdk/ckpt/checkpoint_detection.tar',
                'target_cloud_topic': '/yolo/target_pc',
                'object_cloud_topic': '/yolo/object_pc',
                'use_object_cloud_as_input': False,
                'score_threshold': 0.10,
                'max_publish_grasps': 30,
                'dense_grasp': False,
                'apply_object_mask': True,
                'collision_detection': True,
                'run_once': False,
            }]
        )
    ])
