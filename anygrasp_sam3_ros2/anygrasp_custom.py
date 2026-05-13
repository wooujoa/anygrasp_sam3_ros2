#!/usr/bin/env python3
import os
import sys
import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

# NumPy compatibility patch for legacy AnyGrasp / SDK code
for _name, _value in {
    'float': float,
    'int': int,
    'complex': complex,
}.items():
    if not hasattr(np, _name):
        setattr(np, _name, _value)
try:
    np.bool
except Exception:
    np.bool = bool
try:
    np.object
except Exception:
    np.object = object

from scipy.spatial.transform import Rotation as R

import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from std_msgs.msg import ColorRGBA, Float32
from geometry_msgs.msg import Pose, PoseArray, PoseStamped, Point, PointStamped, Vector3Stamped
from sensor_msgs.msg import PointCloud2, CameraInfo
from sensor_msgs_py import point_cloud2
from visualization_msgs.msg import Marker, MarkerArray
import tf2_ros


@dataclass
class RankedGrasp:
    score: float
    width: float
    translation: np.ndarray
    rotation_matrix: np.ndarray
    rank: float = 0.0
    mask_ok: bool = False
    opening_axis: Optional[np.ndarray] = None
    approach_axis: Optional[np.ndarray] = None
    body_axis: Optional[np.ndarray] = None
    body_align: float = 0.0
    opening_perp: float = 0.0
    approach_perp: float = 0.0
    width_match: float = 0.0
    radial_score: float = 0.0
    mid_score: float = 0.0
    bg_clearance: float = float('inf')
    bg_collision_count: int = 0
    okrobot_theta: float = 0.0
    okrobot_score: float = 0.0
    horizontal_score: float = 0.0
    arm_side_dot: float = 0.0
    arm_side_ok: bool = True
    top_down_dot: float = 0.0
    top_down_score: float = 0.0
    target_dist: float = float('inf')
    target_ok: bool = False
    local_span: float = 0.0
    local_width_limit: float = 0.0
    width_gate_ok: bool = True


class AnyGraspFromTopicNode(Node):
    def __init__(self) -> None:
        super().__init__('anygrasp_from_topic_node')

        # SDK
        self.declare_parameter('sdk_root', '/home/jwg/anygrasp_sdk/grasp_detection')
        self.declare_parameter('checkpoint_path', '/home/jwg/anygrasp_sdk/ckpt/checkpoint_detection.tar')
        self.declare_parameter('max_gripper_width', 0.15)
        self.declare_parameter('gripper_height', 0.03)
        self.declare_parameter('top_down_grasp', False)
        self.declare_parameter('debug', False)
        self.declare_parameter('dense_grasp', False)
        self.declare_parameter('apply_object_mask', True)
        self.declare_parameter('collision_detection', True)

        # inputs
        self.declare_parameter('scene_cloud_topic', '/sam3/full_scene_pc')
        self.declare_parameter('target_cloud_topic', '/yolo/target_pc')
        self.declare_parameter('object_cloud_topic', '/yolo/object_pc')
        self.declare_parameter('background_cloud_topic', '/yolo/background_pc')
        self.declare_parameter('target_mask_topic', '/sam3/target_mask')
        self.declare_parameter('camera_info_topic', '/camera_r/camera_r/aligned_depth_to_color/camera_info')
        self.declare_parameter('use_scene_cloud', True)
        self.declare_parameter('run_once', False)
        self.declare_parameter('min_points', 150)
        self.declare_parameter('voxel_size', 0.004)
        self.declare_parameter('crop_margin_x', 0.02)
        self.declare_parameter('crop_margin_y', 0.02)
        self.declare_parameter('crop_margin_z', 0.02)

        # scoring / filtering
        self.declare_parameter('score_threshold', 0.05)
        self.declare_parameter('max_publish_grasps', 50)
        self.declare_parameter('mask_filter_margin_px', 16)
        self.declare_parameter('local_radius', 0.035)
        self.declare_parameter('width_match_sigma', 0.03)
        self.declare_parameter('opening_perp_weight', 0.90)
        self.declare_parameter('approach_perp_weight', 1.20)
        self.declare_parameter('body_align_weight', 0.60)
        self.declare_parameter('mid_score_weight', 0.70)
        self.declare_parameter('radial_score_weight', 0.60)
        self.declare_parameter('mask_score_bonus', 0.60)
        self.declare_parameter('bg_clearance_weight', 0.20)
        self.declare_parameter('min_bg_clearance', 0.000)
        self.declare_parameter('enable_local_width_gate', True)
        self.declare_parameter('enable_object_width_gate', False)
        self.declare_parameter('width_safety_margin', 0.010)
        self.declare_parameter('object_width_ratio_limit', 1.00)
        self.declare_parameter('rank_threshold', -10.0)
        self.declare_parameter('prefer_mid_body', True)
        self.declare_parameter('prefer_side_grasp', True)

        # OK-Robot style grasp heuristic.
        # OK-Robot filters grasps by the language mask and then ranks with
        # a graspness-vs-horizontal-grasp heuristic. In practice, for this
        # camera-frame node, we make the floor normal configurable.
        self.declare_parameter('enable_okrobot_heuristic', True)
        self.declare_parameter('okrobot_rank_weight', 1.0)
        self.declare_parameter('okrobot_theta_penalty_divisor', 10.0)
        self.declare_parameter('okrobot_theta_power', 4.0)
        self.declare_parameter('okrobot_floor_normal_camera', [0.0, 1.0, 0.0])
        self.declare_parameter('okrobot_use_horizontal_theta', True)
        self.declare_parameter('okrobot_keep_existing_shape_terms', False)
        self.declare_parameter('okrobot_shape_terms_weight', 0.15)

        # Arm-side feasibility filter in base_link.
        # IMPORTANT:
        # This is a hard reject filter, not a ranking bonus.
        # Right arm: reject grasps whose gripper face axis points to robot-right (-Y).
        # Left arm : reject grasps whose gripper face axis points to robot-left  (+Y).
        # It does NOT force the gripper to look strongly toward the opposite side.
        self.declare_parameter('enable_arm_side_filter', True)
        self.declare_parameter('arm_side', 'right')  # 'right' or 'left'
        # Deprecated old name kept for launch compatibility. Not used for scoring.
        self.declare_parameter('arm_side_min_dot', 0.0)
        # Reject only when dot with the desired inward side is smaller than -this value.
        # Example for right arm: desired is +Y. If face_axis_y < -0.05, it is looking right and rejected.
        self.declare_parameter('arm_side_reject_dot', 0.05)
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('base_frame_candidates', ['base_link', 'lift_link', 'arm_base_link'])
        self.declare_parameter('gripper_frame', 'gripper_r_rh_p12_rn_base')
        self.declare_parameter('camera_frame', 'camera_r_color_optical_frame')
        self.declare_parameter('tf_timeout_sec', 0.05)
        self.declare_parameter('apply_anygrasp_pose_frame_alignment_for_filter', True)
        self.declare_parameter('auto_flip_pose_z_180_if_x_points_down_for_filter', True)
        self.declare_parameter('x_axis_downward_flip_threshold_for_filter', 0.0)
        self.declare_parameter('gripper_face_axis_index', 2)  # final pose local z-axis by default
        self.declare_parameter('gripper_face_axis_sign', 1.0)
        self.declare_parameter('arm_side_filter_penalty', 6.0)
        self.declare_parameter('hard_reject_wrong_arm_side', True)

        # Top-down preference for shelf/table objects.
        # This is a ranking bonus only, not a hard filter.
        # It prefers gripper-facing direction from above to below, because
        # bottom-up approaches can collide with the shelf/table surface.
        self.declare_parameter('enable_top_down_bonus', True)
        self.declare_parameter('top_down_bonus_weight', 0.35)
        self.declare_parameter('top_down_axis_index', 2)
        self.declare_parameter('top_down_axis_sign', 1.0)
        self.declare_parameter('top_down_desired_z_sign', -1.0)  # -1: above -> below, +1: below -> above
        self.declare_parameter('top_down_penalize_bottom_up', False)
        self.declare_parameter('bottom_up_penalty_weight', 0.20)

        # Hand-eye copied from the right-arm calibration node.
        # Used only to predict final grasp orientation for ranking/filtering.
        self.T_cam_to_link7 = np.array([
            [ 0.9954,  0.0000, -0.0958,  0.0982],
            [ 0.0000, -1.0000,  0.0000,  0.0000],
            [-0.0958,  0.0000, -0.9954, -0.0725],
            [ 0.0000,  0.0000,  0.0000,  1.0000],
        ], dtype=np.float64)
        self.T_link7_to_gripper_base = np.eye(4, dtype=np.float64)
        self.T_link7_to_gripper_base[:3, :3] = np.array([
            [ 1.0,  0.0,  0.0],
            [ 0.0, -1.0,  0.0],
            [ 0.0,  0.0, -1.0],
        ], dtype=np.float64)
        self.T_link7_to_gripper_base[:3, 3] = np.array([0.0, 0.0, -0.0780], dtype=np.float64)
        self.T_cam_to_gripper = self.T_link7_to_gripper_base @ self.T_cam_to_link7
        self.T_pose_align_y90 = np.eye(4, dtype=np.float64)
        self.T_pose_align_y90[:3, :3] = R.from_euler('y', 90.0, degrees=True).as_matrix()
        self.T_pose_align_z180 = np.eye(4, dtype=np.float64)
        self.T_pose_align_z180[:3, :3] = R.from_euler('z', 180.0, degrees=True).as_matrix()

        # Target/background hard filtering.
        # AnyGrasp may generate grasps from the full scene, but execution must be
        # restricted to the SAM3-selected target. Background is used as obstacles.
        self.declare_parameter('hard_filter_to_target', True)
        self.declare_parameter('require_target_data', True)
        self.declare_parameter('target_filter_use_mask', True)
        self.declare_parameter('target_filter_use_object_pc', True)
        self.declare_parameter('target_filter_radius', 0.060)
        self.declare_parameter('hard_reject_background', True)
        self.declare_parameter('use_gripper_volume_collision', True)
        self.declare_parameter('max_background_collision_points', 30)
        self.declare_parameter('background_collision_margin', 0.002)
        self.declare_parameter('background_collision_sample_limit', 50000)
        self.declare_parameter('collision_use_fixed_width', True)
        self.declare_parameter('collision_gripper_width', 0.10)
        self.declare_parameter('collision_finger_length', 0.055)
        self.declare_parameter('collision_palm_depth', 0.030)
        self.declare_parameter('collision_tail_length', 0.020)
        self.declare_parameter('collision_finger_thickness', 0.010)
        self.declare_parameter('verbose_filter_log', True)

        # outputs
        self.declare_parameter('best_grasp_topic', '/anygrasp/best_grasp')
        self.declare_parameter('best_pose_raw_topic', '/anygrasp/best_pose_raw')
        self.declare_parameter('best_width_topic', '/anygrasp/best_width')
        # Raw AnyGrasp confidence for the selected best grasp.
        # This remains an internal scalar topic. The final custom ObjectGrasp
        # is published only by the calib node after frame conversion.
        self.declare_parameter('best_score_topic', '/anygrasp/best_score')
        self.declare_parameter('grasps_topic', '/anygrasp/grasps')
        self.declare_parameter('markers_topic', '/anygrasp/grasp_markers')
        self.declare_parameter('all_markers_topic', '/anygrasp/all_grasp_markers')
        self.declare_parameter('best_marker_topic', '/anygrasp/best_pose_marker')
        self.declare_parameter('best_contact_marker_topic', '/anygrasp/best_contact_marker')
        self.declare_parameter('best_contact_point_topic', '/anygrasp/best_contact_point')
        self.declare_parameter('best_axes_topic', '/anygrasp/best_axes_markers')

        # visualization
        self.declare_parameter('marker_lifetime_sec', 0.0)
        self.declare_parameter('marker_alpha', 0.85)
        self.declare_parameter('marker_topk', 50)
        self.declare_parameter('best_contact_scale', 0.012)
        self.declare_parameter('candidate_contact_scale', 0.008)
        self.declare_parameter('best_gripper_line_width', 0.0030)
        self.declare_parameter('candidate_gripper_line_width', 0.0022)
        self.declare_parameter('gripper_finger_length', 0.032)
        self.declare_parameter('gripper_palm_depth', 0.010)
        self.declare_parameter('gripper_tail_length', 0.010)
        self.declare_parameter('gripper_knuckle_forward', 0.006)
        self.declare_parameter('gripper_finger_thickness', 0.004)
        self.declare_parameter('use_visualization_rotation_fix', False)
        self.declare_parameter('use_fixed_visual_gripper_width', True)
        self.declare_parameter('visual_gripper_width', 0.10)
        self.declare_parameter('axis_marker_length', 0.06)


        self.sdk_root = self.get_parameter('sdk_root').value
        self.checkpoint_path = self.get_parameter('checkpoint_path').value
        self.scene_cloud_topic = self.get_parameter('scene_cloud_topic').value
        self.target_cloud_topic = self.get_parameter('target_cloud_topic').value
        self.object_cloud_topic = self.get_parameter('object_cloud_topic').value
        self.background_cloud_topic = self.get_parameter('background_cloud_topic').value
        self.target_mask_topic = self.get_parameter('target_mask_topic').value
        self.camera_info_topic = self.get_parameter('camera_info_topic').value
        self.use_scene_cloud = bool(self.get_parameter('use_scene_cloud').value)
        self.run_once = bool(self.get_parameter('run_once').value)
        self.min_points = int(self.get_parameter('min_points').value)
        self.voxel_size = float(self.get_parameter('voxel_size').value)
        self.crop_margin_x = float(self.get_parameter('crop_margin_x').value)
        self.crop_margin_y = float(self.get_parameter('crop_margin_y').value)
        self.crop_margin_z = float(self.get_parameter('crop_margin_z').value)
        self.score_threshold = float(self.get_parameter('score_threshold').value)
        self.max_publish_grasps = int(self.get_parameter('max_publish_grasps').value)
        self.mask_filter_margin_px = int(self.get_parameter('mask_filter_margin_px').value)
        self.local_radius = float(self.get_parameter('local_radius').value)
        self.width_match_sigma = float(self.get_parameter('width_match_sigma').value)
        self.opening_perp_weight = float(self.get_parameter('opening_perp_weight').value)
        self.approach_perp_weight = float(self.get_parameter('approach_perp_weight').value)
        self.body_align_weight = float(self.get_parameter('body_align_weight').value)
        self.mid_score_weight = float(self.get_parameter('mid_score_weight').value)
        self.radial_score_weight = float(self.get_parameter('radial_score_weight').value)
        self.mask_score_bonus = float(self.get_parameter('mask_score_bonus').value)
        self.bg_clearance_weight = float(self.get_parameter('bg_clearance_weight').value)
        self.min_bg_clearance = float(self.get_parameter('min_bg_clearance').value)
        self.enable_local_width_gate = bool(self.get_parameter('enable_local_width_gate').value)
        self.enable_object_width_gate = bool(self.get_parameter('enable_object_width_gate').value)
        self.width_safety_margin = float(self.get_parameter('width_safety_margin').value)
        self.object_width_ratio_limit = float(self.get_parameter('object_width_ratio_limit').value)
        self.rank_threshold = float(self.get_parameter('rank_threshold').value)
        self.prefer_mid_body = bool(self.get_parameter('prefer_mid_body').value)
        self.prefer_side_grasp = bool(self.get_parameter('prefer_side_grasp').value)

        self.enable_okrobot_heuristic = bool(self.get_parameter('enable_okrobot_heuristic').value)
        self.okrobot_rank_weight = float(self.get_parameter('okrobot_rank_weight').value)
        self.okrobot_theta_penalty_divisor = max(1e-6, float(self.get_parameter('okrobot_theta_penalty_divisor').value))
        self.okrobot_theta_power = max(1.0, float(self.get_parameter('okrobot_theta_power').value))
        self.okrobot_floor_normal_camera = self.normalize_vec(
            np.asarray(self.get_parameter('okrobot_floor_normal_camera').value, dtype=np.float32),
            fallback=np.array([0.0, 1.0, 0.0], dtype=np.float32),
        )
        self.okrobot_use_horizontal_theta = bool(self.get_parameter('okrobot_use_horizontal_theta').value)
        self.okrobot_keep_existing_shape_terms = bool(self.get_parameter('okrobot_keep_existing_shape_terms').value)
        self.okrobot_shape_terms_weight = float(self.get_parameter('okrobot_shape_terms_weight').value)

        self.enable_arm_side_filter = bool(self.get_parameter('enable_arm_side_filter').value)
        self.arm_side = str(self.get_parameter('arm_side').value).lower().strip()
        self.arm_side_min_dot = float(self.get_parameter('arm_side_min_dot').value)
        self.arm_side_reject_dot = abs(float(self.get_parameter('arm_side_reject_dot').value))
        self.base_frame = self.get_parameter('base_frame').value
        self.base_frame_candidates = list(self.get_parameter('base_frame_candidates').value)
        self.gripper_frame = self.get_parameter('gripper_frame').value
        self.camera_frame = self.get_parameter('camera_frame').value
        self.tf_timeout_sec = float(self.get_parameter('tf_timeout_sec').value)
        self.apply_anygrasp_pose_frame_alignment_for_filter = bool(self.get_parameter('apply_anygrasp_pose_frame_alignment_for_filter').value)
        self.auto_flip_pose_z_180_if_x_points_down_for_filter = bool(self.get_parameter('auto_flip_pose_z_180_if_x_points_down_for_filter').value)
        self.x_axis_downward_flip_threshold_for_filter = float(self.get_parameter('x_axis_downward_flip_threshold_for_filter').value)
        self.gripper_face_axis_index = int(self.get_parameter('gripper_face_axis_index').value)
        self.gripper_face_axis_sign = float(self.get_parameter('gripper_face_axis_sign').value)
        self.arm_side_filter_penalty = float(self.get_parameter('arm_side_filter_penalty').value)
        self.hard_reject_wrong_arm_side = bool(self.get_parameter('hard_reject_wrong_arm_side').value)

        self.enable_top_down_bonus = bool(self.get_parameter('enable_top_down_bonus').value)
        self.top_down_bonus_weight = float(self.get_parameter('top_down_bonus_weight').value)
        self.top_down_axis_index = max(0, min(2, int(self.get_parameter('top_down_axis_index').value)))
        self.top_down_axis_sign = float(self.get_parameter('top_down_axis_sign').value)
        self.top_down_desired_z_sign = -1.0 if float(self.get_parameter('top_down_desired_z_sign').value) < 0.0 else 1.0
        self.top_down_penalize_bottom_up = bool(self.get_parameter('top_down_penalize_bottom_up').value)
        self.bottom_up_penalty_weight = float(self.get_parameter('bottom_up_penalty_weight').value)

        if self.arm_side not in ('right', 'left'):
            self.get_logger().warn(f'Unknown arm_side={self.arm_side}. Falling back to right.')
            self.arm_side = 'right'
        self.desired_lateral_sign = 1.0 if self.arm_side == 'right' else -1.0
        self.gripper_face_axis_index = max(0, min(2, self.gripper_face_axis_index))

        self.hard_filter_to_target = bool(self.get_parameter('hard_filter_to_target').value)
        self.require_target_data = bool(self.get_parameter('require_target_data').value)
        self.target_filter_use_mask = bool(self.get_parameter('target_filter_use_mask').value)
        self.target_filter_use_object_pc = bool(self.get_parameter('target_filter_use_object_pc').value)
        self.target_filter_radius = float(self.get_parameter('target_filter_radius').value)
        self.hard_reject_background = bool(self.get_parameter('hard_reject_background').value)
        self.use_gripper_volume_collision = bool(self.get_parameter('use_gripper_volume_collision').value)
        self.max_background_collision_points = int(self.get_parameter('max_background_collision_points').value)
        self.background_collision_margin = float(self.get_parameter('background_collision_margin').value)
        self.background_collision_sample_limit = int(self.get_parameter('background_collision_sample_limit').value)
        self.collision_use_fixed_width = bool(self.get_parameter('collision_use_fixed_width').value)
        self.collision_gripper_width = float(self.get_parameter('collision_gripper_width').value)
        self.collision_finger_length = float(self.get_parameter('collision_finger_length').value)
        self.collision_palm_depth = float(self.get_parameter('collision_palm_depth').value)
        self.collision_tail_length = float(self.get_parameter('collision_tail_length').value)
        self.collision_finger_thickness = float(self.get_parameter('collision_finger_thickness').value)
        self.verbose_filter_log = bool(self.get_parameter('verbose_filter_log').value)

        self.marker_lifetime_sec = float(self.get_parameter('marker_lifetime_sec').value)
        self.marker_alpha = float(self.get_parameter('marker_alpha').value)
        self.marker_topk = int(self.get_parameter('marker_topk').value)
        self.best_contact_scale = float(self.get_parameter('best_contact_scale').value)
        self.candidate_contact_scale = float(self.get_parameter('candidate_contact_scale').value)
        self.best_gripper_line_width = float(self.get_parameter('best_gripper_line_width').value)
        self.candidate_gripper_line_width = float(self.get_parameter('candidate_gripper_line_width').value)
        self.gripper_finger_length = float(self.get_parameter('gripper_finger_length').value)
        self.gripper_palm_depth = float(self.get_parameter('gripper_palm_depth').value)
        self.gripper_tail_length = float(self.get_parameter('gripper_tail_length').value)
        self.gripper_knuckle_forward = float(self.get_parameter('gripper_knuckle_forward').value)
        self.gripper_finger_thickness = float(self.get_parameter('gripper_finger_thickness').value)
        self.use_visualization_rotation_fix = bool(self.get_parameter('use_visualization_rotation_fix').value)
        self.use_fixed_visual_gripper_width = bool(self.get_parameter('use_fixed_visual_gripper_width').value)
        self.visual_gripper_width = float(self.get_parameter('visual_gripper_width').value)
        self.axis_marker_length = float(self.get_parameter('axis_marker_length').value)

        self.r_fix = np.array([
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
        ], dtype=np.float32)

        self._append_sdk_path(self.sdk_root)
        self.anygrasp = self._build_anygrasp()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self._already_processed = False

        self.scene_points: Optional[np.ndarray] = None
        self.scene_colors: Optional[np.ndarray] = None
        self.object_points: Optional[np.ndarray] = None
        self.background_points: Optional[np.ndarray] = None
        self.target_mask: Optional[np.ndarray] = None
        self.camera_info: Optional[CameraInfo] = None
        self.body_axis: Optional[np.ndarray] = None
        self.body_center: Optional[np.ndarray] = None
        self.body_min: float = 0.0
        self.body_max: float = 0.0
        self.body_radius: float = 0.02
        self.object_width_est: float = 0.0
        self.header = None
        self.last_filter_stats = {}

        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )

        input_topic = self.scene_cloud_topic if self.use_scene_cloud else self.target_cloud_topic
        self.scene_sub = self.create_subscription(PointCloud2, input_topic, self.scene_cloud_callback, qos)
        self.object_sub = self.create_subscription(PointCloud2, self.object_cloud_topic, self.object_cloud_callback, qos)
        self.background_sub = self.create_subscription(PointCloud2, self.background_cloud_topic, self.background_cloud_callback, qos)
        self.mask_sub = self.create_subscription(PointCloud2 if False else __import__('sensor_msgs.msg').msg.Image, self.target_mask_topic, self.mask_callback, qos)
        self.cam_sub = self.create_subscription(CameraInfo, self.camera_info_topic, self.camera_info_callback, qos)

        self.best_pub = self.create_publisher(PoseStamped, self.get_parameter('best_grasp_topic').value, 10)
        self.best_pose_raw_pub = self.create_publisher(PoseStamped, self.get_parameter('best_pose_raw_topic').value, 10)
        self.best_width_pub = self.create_publisher(Float32, self.get_parameter('best_width_topic').value, 10)
        self.best_score_pub = self.create_publisher(Float32, self.get_parameter('best_score_topic').value, 10)
        self.grasps_pub = self.create_publisher(PoseArray, self.get_parameter('grasps_topic').value, 10)
        self.markers_pub = self.create_publisher(MarkerArray, self.get_parameter('markers_topic').value, 10)
        self.all_markers_pub = self.create_publisher(MarkerArray, self.get_parameter('all_markers_topic').value, 10)
        self.best_marker_pub = self.create_publisher(Marker, self.get_parameter('best_marker_topic').value, 10)
        self.best_contact_marker_pub = self.create_publisher(Marker, self.get_parameter('best_contact_marker_topic').value, 10)
        self.best_contact_point_pub = self.create_publisher(PointStamped, self.get_parameter('best_contact_point_topic').value, 10)
        self.best_axes_pub = self.create_publisher(MarkerArray, self.get_parameter('best_axes_topic').value, 10)

        self.get_logger().info('AnyGrasp topic node ready.')
        self.get_logger().info(f'input_topic={input_topic}')
        self.get_logger().info(f'object_cloud_topic={self.object_cloud_topic}')
        self.get_logger().info(f'background_cloud_topic={self.background_cloud_topic}')
        self.get_logger().info(f'target_mask_topic={self.target_mask_topic}')
        self.get_logger().info(f'use_scene_cloud={self.use_scene_cloud}')
        self.get_logger().info(f'best_pose_raw_topic={self.get_parameter("best_pose_raw_topic").value}')
        self.get_logger().info(f'best_width_topic={self.get_parameter("best_width_topic").value}')
        self.get_logger().info(f'best_score_topic={self.get_parameter("best_score_topic").value}')
        self.get_logger().info(f'enable_local_width_gate={self.enable_local_width_gate}')
        self.get_logger().info(f'enable_object_width_gate={self.enable_object_width_gate}')
        self.get_logger().info(f'width_safety_margin={self.width_safety_margin:.4f}')
        self.get_logger().info(f'hard_filter_to_target={self.hard_filter_to_target}')
        self.get_logger().info(f'require_target_data={self.require_target_data}')
        self.get_logger().info(f'target_filter_radius={self.target_filter_radius:.4f}')
        self.get_logger().info(f'hard_reject_background={self.hard_reject_background}')
        self.get_logger().info(f'use_gripper_volume_collision={self.use_gripper_volume_collision}')
        self.get_logger().info(f'min_bg_clearance={self.min_bg_clearance:.4f}')
        self.get_logger().info(f'background_collision_margin={self.background_collision_margin:.4f}')
        self.get_logger().info(f'enable_okrobot_heuristic={self.enable_okrobot_heuristic}')
        self.get_logger().info(f'okrobot_floor_normal_camera={self.okrobot_floor_normal_camera.tolist()}')
        self.get_logger().info(f'okrobot_use_horizontal_theta={self.okrobot_use_horizontal_theta}')
        self.get_logger().info(f'enable_arm_side_filter={self.enable_arm_side_filter}')
        self.get_logger().info(f'arm_side={self.arm_side} desired_lateral_sign={self.desired_lateral_sign:+.1f}')
        self.get_logger().info(f'arm_side_min_dot={self.arm_side_min_dot:.3f}  # deprecated/ignored')
        self.get_logger().info(f'arm_side_reject_dot={self.arm_side_reject_dot:.3f}  # reject if side_dot < -this')
        self.get_logger().info(f'gripper_face_axis_index={self.gripper_face_axis_index} sign={self.gripper_face_axis_sign:+.1f}')
        self.get_logger().info(f'enable_top_down_bonus={self.enable_top_down_bonus}')
        self.get_logger().info(f'top_down_axis_index={self.top_down_axis_index} sign={self.top_down_axis_sign:+.1f}')
        self.get_logger().info(f'top_down_desired_z_sign={self.top_down_desired_z_sign:+.1f}  # -1 means above-to-below')
        self.get_logger().info(f'top_down_bonus_weight={self.top_down_bonus_weight:.3f}')

    def _append_sdk_path(self, sdk_root: str) -> None:
        if not os.path.isdir(sdk_root):
            raise RuntimeError(f'sdk_root does not exist: {sdk_root}')
        if sdk_root not in sys.path:
            sys.path.insert(0, sdk_root)

    def _build_anygrasp(self):
        from gsnet import AnyGrasp  # type: ignore
        import argparse
        cfgs = argparse.Namespace(
            checkpoint_path=self.checkpoint_path,
            max_gripper_width=max(0.0, min(0.1, float(self.get_parameter('max_gripper_width').value))),
            gripper_height=float(self.get_parameter('gripper_height').value),
            top_down_grasp=bool(self.get_parameter('top_down_grasp').value),
            debug=bool(self.get_parameter('debug').value),
        )
        ag = AnyGrasp(cfgs)
        ag.load_net()
        return ag

    def camera_info_callback(self, msg: CameraInfo) -> None:
        self.camera_info = msg

    def mask_callback(self, msg) -> None:
        try:
            import cv2
            from cv_bridge import CvBridge
            bridge = CvBridge()
            mask = bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
            self.target_mask = (mask > 0)
        except Exception as exc:
            self.get_logger().warn(f'Failed to parse target mask: {repr(exc)}')

    def object_cloud_callback(self, msg: PointCloud2) -> None:
        try:
            pts, _ = self.pointcloud2_to_numpy(msg)
            if pts.shape[0] >= 20:
                self.object_points = pts
                self.update_body_model()
                if self.verbose_filter_log:
                    self.get_logger().info(
                        f'target/object cloud updated: points={pts.shape[0]} '
                        f'width_est={self.object_width_est:.4f}m'
                    )
        except Exception as exc:
            self.get_logger().warn(f'object cloud parse failed: {repr(exc)}')

    def background_cloud_callback(self, msg: PointCloud2) -> None:
        try:
            pts, _ = self.pointcloud2_to_numpy(msg)
            self.background_points = pts if pts.shape[0] > 0 else None
            if self.verbose_filter_log:
                self.get_logger().info(f'background cloud updated: points={pts.shape[0]}')
        except Exception as exc:
            self.get_logger().warn(f'background cloud parse failed: {repr(exc)}')

    def update_body_model(self) -> None:
        if self.object_points is None or self.object_points.shape[0] < 20:
            return
        pts = self.object_points
        center = pts.mean(axis=0)
        X = pts - center
        _, _, vh = np.linalg.svd(X, full_matrices=False)
        axis = vh[0].astype(np.float32)
        axis /= max(np.linalg.norm(axis), 1e-8)
        proj = X @ axis
        ortho = X - np.outer(proj, axis)
        radial = np.linalg.norm(ortho, axis=1)
        self.body_axis = axis
        self.body_center = center.astype(np.float32)
        self.body_min = float(np.percentile(proj, 2.0))
        self.body_max = float(np.percentile(proj, 98.0))
        self.body_radius = max(0.01, float(np.percentile(radial, 75.0)))
        self.object_width_est = max(0.0, float(2.0 * np.percentile(radial, 90.0)))

    def scene_cloud_callback(self, msg: PointCloud2) -> None:
        if self.run_once and self._already_processed:
            return
        try:
            points, colors = self.pointcloud2_to_numpy(msg)
            self.header = msg.header
            self.get_logger().info(
                f'cloud parsed: points_shape={points.shape}, points_dtype={points.dtype}, '
                f'colors_shape={colors.shape}, colors_dtype={colors.dtype}'
            )
            if points.shape[0] < self.min_points:
                self.get_logger().warn(f'Input cloud too small: {points.shape[0]} < {self.min_points}')
                self.publish_empty_markers(msg.header)
                return
            points, colors = self.voxel_downsample(points, colors, self.voxel_size)
            self.scene_points = points
            self.scene_colors = colors

            if self.enable_object_width_gate and self.object_width_est > 0.0:
                gate_limit = max(0.0, float(self.get_parameter('max_gripper_width').value) * self.object_width_ratio_limit - self.width_safety_margin)
                if self.object_width_est > gate_limit:
                    self.get_logger().warn(
                        f'Object width gate rejected object: est_width={self.object_width_est:.4f} m > limit={gate_limit:.4f} m'
                    )
                    self.publish_empty_markers(msg.header)
                    return

            lims = self.compute_lims(points)
            gg, _ = self.anygrasp.get_grasp(
                points,
                colors,
                lims=lims,
                apply_object_mask=bool(self.get_parameter('apply_object_mask').value),
                dense_grasp=bool(self.get_parameter('dense_grasp').value),
                collision_detection=bool(self.get_parameter('collision_detection').value),
            )
            grasps = self.convert_grasp_group(gg)
            grasps = [g for g in grasps if np.isfinite(g.score) and g.score >= self.score_threshold]
            if len(grasps) == 0:
                self.get_logger().warn('AnyGrasp produced no grasps above score threshold.')
                self.publish_empty_markers(msg.header)
                return

            ranked_all = self.rank_grasps(grasps)
            if len(ranked_all) == 0:
                stats = getattr(self, 'last_filter_stats', {})
                self.get_logger().warn(
                    'No grasps survived target/background hard filters. '
                    f'stats={stats}'
                )
                self.publish_empty_markers(msg.header)
                return

            width_ok_count = sum(1 for g in ranked_all if g.width_gate_ok)
            if self.enable_local_width_gate and width_ok_count == 0:
                self.get_logger().warn('All surviving target grasps are rejected by local width gate. Object may be too wide locally for the gripper.')
            ranked = [g for g in ranked_all if g.rank >= self.rank_threshold]
            if len(ranked) == 0:
                self.get_logger().warn('All grasps removed after ranking threshold. Falling back to raw scores.')
                ranked = self.rank_grasps(grasps, force_keep=True)
                if len(ranked) == 0:
                    self.publish_empty_markers(msg.header)
                    return

            ranked.sort(key=lambda g: g.rank, reverse=True)
            ranked = ranked[: self.max_publish_grasps]
            self.log_top_candidates(ranked, 10)
            self.publish_outputs(msg.header, ranked)
            self._already_processed = True
        except Exception as exc:
            self.get_logger().error(f'AnyGrasp inference failed: {repr(exc)}')
            self.publish_empty_markers(msg.header if hasattr(msg, 'header') else None)

    def rank_grasps(self, grasps: Sequence[RankedGrasp], force_keep: bool = False) -> List[RankedGrasp]:
        """
        Rank grasps while enforcing the intended pipeline semantics:
          - target/object_pc or SAM3 mask defines what can be grasped
          - background_pc defines obstacles that must not be grasped/collided with

        force_keep only bypasses soft ranking penalties. It does NOT bypass
        target/background hard filters, because those protect against grasping
        the wrong object.
        """
        stats = {
            'raw': len(grasps),
            'target_reject': 0,
            'background_reject': 0,
            'width_soft_reject': 0,
            'arm_side_reject': 0,
            'kept': 0,
            'target_data_available': int(self.target_data_available()),
            'background_available': int(self.background_points is not None and self.background_points.shape[0] > 0),
        }

        ranked: List[RankedGrasp] = []
        for grasp in grasps:
            g = RankedGrasp(
                score=grasp.score,
                width=grasp.width,
                translation=grasp.translation.copy(),
                rotation_matrix=grasp.rotation_matrix.copy(),
            )

            # 1) Target membership hard filter
            g.target_ok, g.mask_ok, g.target_dist = self.target_membership(g.translation)
            if self.hard_filter_to_target and not g.target_ok:
                stats['target_reject'] += 1
                continue

            # 2) Compute axes using target object geometry only
            local_pts = self.local_object_points(g.translation)
            opening_axis, approach_axis, body_axis = self.infer_axes(g, local_pts)
            g.opening_axis = opening_axis
            g.approach_axis = approach_axis
            g.body_axis = body_axis

            if self.body_axis is not None:
                g.body_align = abs(float(np.dot(body_axis, self.body_axis)))
                g.opening_perp = 1.0 - abs(float(np.dot(opening_axis, self.body_axis)))
                g.approach_perp = 1.0 - abs(float(np.dot(approach_axis, self.body_axis)))
                g.radial_score, g.mid_score = self.position_scores(g.translation)
            else:
                g.body_align = 0.0
                g.opening_perp = 0.0
                g.approach_perp = 0.0
                g.radial_score = 0.0
                g.mid_score = 0.0

            # 3) Width/local geometry gate on target object only
            g.width_match, g.local_span = self.width_match(local_pts, g.translation, opening_axis, g.width)
            g.local_width_limit = max(0.0, float(self.get_parameter('max_gripper_width').value) - self.width_safety_margin)
            g.width_gate_ok = (not self.enable_local_width_gate) or (g.local_span <= g.local_width_limit)

            # 4) Background obstacle hard filter
            g.bg_clearance = self.background_clearance(g.translation)
            g.bg_collision_count = self.background_collision_count(g)

            bg_too_close = g.bg_clearance < self.min_bg_clearance
            bg_in_gripper = g.bg_collision_count > self.max_background_collision_points
            if self.hard_reject_background and (bg_too_close or bg_in_gripper):
                stats['background_reject'] += 1
                continue

            # 5) Arm-side feasibility filter in final base_link orientation.
            # Right arm: reject grasps whose gripper face axis points to robot-right (-Y).
            # Left arm : reject grasps whose gripper face axis points to robot-left (+Y).
            g.arm_side_ok, g.arm_side_dot = self.arm_side_feasibility(g)
            if self.enable_arm_side_filter and self.hard_reject_wrong_arm_side and (not g.arm_side_ok):
                stats['arm_side_reject'] += 1
                continue

            # 6) Top-down preference score in final base_link orientation.
            # This is only a soft ranking term. It does not remove candidates.
            g.top_down_score, g.top_down_dot = self.top_down_preference_score(g)

            # 7) Final ranking after hard filters
            # OK-Robot heuristic:
            #   1) filter candidates by target mask/segment membership
            #   2) select by raw graspness while penalizing non-horizontal grasps
            # The paper writes the heuristic as S - theta^4/10. Here theta is
            # implemented as the deviation from a horizontal side grasp by default,
            # because that is the behavior the paper says it prefers for robustness
            # to hand-eye calibration error.
            g.okrobot_theta, g.okrobot_score, g.horizontal_score = self.okrobot_heuristic_score(g)

            if self.enable_okrobot_heuristic:
                rank = self.okrobot_rank_weight * g.okrobot_score
                if self.enable_top_down_bonus:
                    rank += self.top_down_bonus_weight * g.top_down_score

                shape_rank = 0.0
                if g.mask_ok:
                    shape_rank += self.mask_score_bonus
                if self.prefer_side_grasp:
                    shape_rank += self.opening_perp_weight * g.opening_perp
                    shape_rank += self.approach_perp_weight * g.approach_perp
                shape_rank += self.body_align_weight * g.body_align
                shape_rank += 0.5 * g.width_match
                shape_rank += self.radial_score_weight * g.radial_score
                if self.prefer_mid_body:
                    shape_rank += self.mid_score_weight * g.mid_score
                shape_rank += self.bg_clearance_weight * min(g.bg_clearance, 0.05)

                if self.okrobot_keep_existing_shape_terms:
                    rank += self.okrobot_shape_terms_weight * shape_rank
                else:
                    # Keep only very weak safety/quality preferences so that the
                    # OK-Robot horizontal heuristic, not the older body-axis ranking,
                    # decides the final orientation.
                    rank += 0.05 * float(g.mask_ok)
                    rank += 0.05 * g.width_match
                    rank += 0.03 * min(g.bg_clearance, 0.05)
                    if self.enable_arm_side_filter and not g.arm_side_ok:
                        rank -= self.arm_side_filter_penalty
            else:
                rank = float(g.score)
                if self.enable_top_down_bonus:
                    rank += self.top_down_bonus_weight * g.top_down_score
                if g.mask_ok:
                    rank += self.mask_score_bonus
                if self.prefer_side_grasp:
                    rank += self.opening_perp_weight * g.opening_perp
                    rank += self.approach_perp_weight * g.approach_perp
                rank += self.body_align_weight * g.body_align
                rank += 0.5 * g.width_match
                rank += self.radial_score_weight * g.radial_score
                if self.prefer_mid_body:
                    rank += self.mid_score_weight * g.mid_score
                rank += self.bg_clearance_weight * min(g.bg_clearance, 0.05)
                if self.enable_arm_side_filter and not g.arm_side_ok:
                    rank -= self.arm_side_filter_penalty

            if (not force_keep) and self.body_axis is not None:
                if not g.width_gate_ok:
                    rank -= 4.0
                    stats['width_soft_reject'] += 1
                if not self.enable_okrobot_heuristic:
                    if g.approach_perp < 0.45:
                        rank -= 1.25
                    if g.opening_perp < 0.55:
                        rank -= 1.00
                    if g.mid_score < 0.20:
                        rank -= 0.45
                    if g.radial_score < 0.10:
                        rank -= 0.45

            g.rank = rank
            ranked.append(g)

        stats['kept'] = len(ranked)
        self.last_filter_stats = stats
        if self.verbose_filter_log:
            self.get_logger().info(
                '[filter] '
                f"raw={stats['raw']} kept={stats['kept']} "
                f"target_reject={stats['target_reject']} "
                f"background_reject={stats['background_reject']} "
                f"width_soft={stats['width_soft_reject']} "
                f"arm_side_reject={stats['arm_side_reject']} "
                f"target_data={stats['target_data_available']} "
                f"background_data={stats['background_available']}"
            )
        return ranked


    @staticmethod
    def normalize_vec(vec: np.ndarray, fallback: Optional[np.ndarray] = None) -> np.ndarray:
        vec = np.asarray(vec, dtype=np.float32).reshape(-1)
        if vec.shape[0] != 3 or not np.all(np.isfinite(vec)):
            if fallback is None:
                fallback = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            return fallback.astype(np.float32)
        n = float(np.linalg.norm(vec))
        if n < 1e-8:
            if fallback is None:
                fallback = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            return fallback.astype(np.float32)
        return (vec / n).astype(np.float32)

    def okrobot_heuristic_score(self, grasp: RankedGrasp) -> Tuple[float, float, float]:
        """
        OK-Robot-style heuristic score.

        Paper reference behavior:
          - Project grasp points into the target mask and keep grasps inside it.
          - Rank with S - theta^4 / 10.
          - Prefer flat/horizontal grasps because they are more robust to
            hand-eye calibration error than vertical grasps.

        This node has already performed the mask/object hard filter before this
        function. The remaining job here is the angle penalty.

        theta returned here is in radians. With okrobot_use_horizontal_theta=True,
        theta means deviation from a horizontal side grasp, i.e. angle between
        approach_axis and the plane perpendicular to floor normal. Thus theta=0
        is best. If False, theta is the literal angle to the floor normal.
        """
        if grasp.approach_axis is None:
            # Fallback to the third rotation column, which is commonly the
            # approach/normal-like axis in AnyGrasp-style frames.
            approach = grasp.rotation_matrix[:, 2].astype(np.float32)
        else:
            approach = grasp.approach_axis.astype(np.float32)
        approach = self.normalize_vec(approach, fallback=np.array([0.0, 0.0, 1.0], dtype=np.float32))
        floor_n = self.okrobot_floor_normal_camera

        cos_abs = max(-1.0, min(1.0, abs(float(np.dot(approach, floor_n)))))
        literal_theta = math.acos(cos_abs)  # 0: parallel to floor normal, pi/2: horizontal side approach

        if self.okrobot_use_horizontal_theta:
            theta = abs((math.pi * 0.5) - literal_theta)  # 0: horizontal side approach
        else:
            theta = literal_theta

        penalty = (theta ** self.okrobot_theta_power) / self.okrobot_theta_penalty_divisor
        ok_score = float(grasp.score) - float(penalty)
        horizontal_score = max(0.0, 1.0 - theta / (math.pi * 0.5))
        return float(theta), float(ok_score), float(horizontal_score)

    def top_down_preference_score(self, grasp: RankedGrasp) -> Tuple[float, float]:
        """
        Soft preference for grasps whose final gripper-facing axis points downward.

        This is intended for objects on a shelf/table: an approach/facing direction
        from above to below is usually safer than a bottom-up direction, because
        bottom-up motion can collide with the supporting surface.

        Returns:
          score : [0, 1] by default, where 1 means strongly above-to-below.
                  If top_down_penalize_bottom_up=True, the score can be negative
                  for bottom-up directions.
          dot   : signed alignment with desired vertical direction.
                  +1 means desired direction, -1 means opposite direction.
        """
        if not self.enable_top_down_bonus:
            return 0.0, 0.0

        T_final_base = self.predict_final_pose_base_for_filter(grasp)
        if T_final_base is None:
            return 0.0, 0.0

        axis_idx = max(0, min(2, int(self.top_down_axis_index)))
        axis = T_final_base[:3, axis_idx].astype(np.float64) * float(self.top_down_axis_sign)
        n = float(np.linalg.norm(axis))
        if n < 1e-9:
            return 0.0, 0.0
        axis = axis / n

        # base_link convention: +Z is up. desired_z_sign=-1 means above-to-below.
        dot = float(axis[2] * self.top_down_desired_z_sign)
        if self.top_down_penalize_bottom_up:
            score = dot
        else:
            score = max(0.0, dot)
        return float(score), float(dot)

    def arm_side_feasibility(self, grasp: RankedGrasp) -> Tuple[bool, float]:
        """
        Check final, calibration-aligned gripper facing direction in base_link.

        base_link convention assumed here:
          +X: robot forward
          +Y: robot left
          +Z: up

        This filter only removes the clearly wrong side.

        For right arm, desired/inward side is +Y. A grasp is rejected only if
        the face axis points clearly to -Y, i.e. dot < -arm_side_reject_dot.
        For left arm, desired/inward side is -Y. A grasp is rejected only if
        the face axis points clearly to +Y.

        Returns:
          ok  : True if the grasp is not clearly facing the wrong side
          dot : signed lateral alignment. Positive is inward side, negative is wrong side.
        """
        if not self.enable_arm_side_filter:
            return True, 0.0
        T_final_base = self.predict_final_pose_base_for_filter(grasp)
        if T_final_base is None:
            # Do not kill all grasps if TF is not available at inference time.
            return True, 0.0
        axis_idx = max(0, min(2, int(self.gripper_face_axis_index)))
        face_axis = T_final_base[:3, axis_idx].astype(np.float64) * float(self.gripper_face_axis_sign)
        n = float(np.linalg.norm(face_axis))
        if n < 1e-9:
            return True, 0.0
        face_axis = face_axis / n
        # Dot with desired lateral direction. right arm desired/inward is +Y, left arm desired/inward is -Y.
        # Do NOT require a positive dot. Only reject when it is clearly negative.
        dot = float(face_axis[1] * self.desired_lateral_sign)
        return bool(dot >= -self.arm_side_reject_dot), dot

    def predict_final_pose_base_for_filter(self, grasp: RankedGrasp) -> Optional[np.ndarray]:
        """Replicate the calibration node's pose-orientation transform for filtering only."""
        try:
            T_pose_in = np.eye(4, dtype=np.float64)
            T_pose_in[:3, :3] = grasp.rotation_matrix.astype(np.float64)
            T_pose_in[:3, 3] = grasp.translation.astype(np.float64)

            t = self.resolve_base_from_gripper_for_filter()
            if t is None:
                return None
            T_gripper_to_base = self.make_transform_matrix_from_tf(t)
            T_raw_base = T_gripper_to_base @ self.T_cam_to_gripper @ T_pose_in
            T_final_base = T_raw_base.copy()
            if self.apply_anygrasp_pose_frame_alignment_for_filter:
                T_final_base = T_final_base @ self.T_pose_align_y90
                x_axis_after_y90 = T_final_base[:3, 0].copy()
                if self.auto_flip_pose_z_180_if_x_points_down_for_filter:
                    if x_axis_after_y90[2] < self.x_axis_downward_flip_threshold_for_filter:
                        T_final_base = T_final_base @ self.T_pose_align_z180
            return T_final_base
        except Exception as exc:
            if self.verbose_filter_log:
                self.get_logger().warn(f'arm_side_filter transform failed: {repr(exc)}')
            return None

    def resolve_base_from_gripper_for_filter(self):
        frames = []
        if self.base_frame:
            frames.append(self.base_frame)
        for f in self.base_frame_candidates:
            if f not in frames:
                frames.append(f)
        for target in frames:
            try:
                return self.tf_buffer.lookup_transform(
                    target,
                    self.gripper_frame,
                    rclpy.time.Time(),
                    timeout=Duration(seconds=self.tf_timeout_sec),
                )
            except Exception:
                continue
        return None

    @staticmethod
    def make_transform_matrix_from_tf(tf_msg) -> np.ndarray:
        t = tf_msg.transform.translation
        q = tf_msg.transform.rotation
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
        T[:3, 3] = np.array([t.x, t.y, t.z], dtype=np.float64)
        return T

    def target_data_available(self) -> bool:
        has_mask = self.target_filter_use_mask and self.target_mask is not None and self.camera_info is not None
        has_object = (
            self.target_filter_use_object_pc
            and self.object_points is not None
            and self.object_points.shape[0] > 0
        )
        return bool(has_mask or has_object)

    def target_membership(self, xyz: np.ndarray) -> Tuple[bool, bool, float]:
        """
        Returns: (target_ok, mask_ok, target_dist).
        target_ok is true if the grasp contact/center belongs to the selected
        SAM3 target by either projected mask membership or 3D object_pc proximity.
        """
        mask_ok = False
        target_dist = float('inf')
        data_available = False

        if self.target_filter_use_mask and self.target_mask is not None and self.camera_info is not None:
            data_available = True
            mask_ok = self.is_inside_mask(xyz)

        if self.target_filter_use_object_pc and self.object_points is not None and self.object_points.shape[0] > 0:
            data_available = True
            target_dist = self.distance_to_object(xyz)

        if not data_available:
            return (not self.require_target_data), mask_ok, target_dist

        object_ok = target_dist <= self.target_filter_radius
        target_ok = bool(mask_ok or object_ok)
        return target_ok, mask_ok, target_dist

    def distance_to_object(self, xyz: np.ndarray) -> float:
        if self.object_points is None or self.object_points.shape[0] == 0:
            return float('inf')
        d = np.linalg.norm(self.object_points - xyz[None, :], axis=1)
        return float(np.min(d))

    def is_inside_mask(self, xyz: np.ndarray) -> bool:
        if self.target_mask is None or self.camera_info is None:
            return False
        u, v = self.project_point(xyz)
        if u is None:
            return False
        h, w = self.target_mask.shape
        if not (0 <= u < w and 0 <= v < h):
            return False
        if self.target_mask[v, u]:
            return True
        m = self.mask_filter_margin_px
        x1 = max(0, u - m)
        x2 = min(w, u + m + 1)
        y1 = max(0, v - m)
        y2 = min(h, v + m + 1)
        return bool(np.any(self.target_mask[y1:y2, x1:x2]))

    def project_point(self, xyz: np.ndarray) -> Tuple[Optional[int], Optional[int]]:
        if self.camera_info is None:
            return None, None
        x, y, z = [float(v) for v in xyz]
        if z <= 1e-6:
            return None, None
        k = self.camera_info.k
        fx, fy, cx, cy = float(k[0]), float(k[4]), float(k[2]), float(k[5])
        u = int(round((x * fx / z) + cx))
        v = int(round((y * fy / z) + cy))
        return u, v

    def local_object_points(self, center: np.ndarray) -> np.ndarray:
        if self.object_points is None or self.object_points.shape[0] == 0:
            return np.empty((0, 3), dtype=np.float32)
        d = np.linalg.norm(self.object_points - center[None, :], axis=1)
        idx = d <= self.local_radius
        pts = self.object_points[idx]
        if pts.shape[0] < 12:
            nearest = np.argsort(d)[: min(40, self.object_points.shape[0])]
            pts = self.object_points[nearest]
        return pts.astype(np.float32)

    def infer_axes(self, grasp: RankedGrasp, local_pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        Rm = grasp.rotation_matrix.astype(np.float32)
        cols = [Rm[:, i].astype(np.float32) for i in range(3)]
        cols = [c / max(np.linalg.norm(c), 1e-8) for c in cols]

        if self.body_axis is None:
            return cols[0], cols[2], cols[1]

        alignments = [abs(float(np.dot(c, self.body_axis))) for c in cols]
        body_idx = int(np.argmax(alignments))
        body_axis = cols[body_idx]
        rem = [i for i in range(3) if i != body_idx]

        if local_pts.shape[0] >= 6:
            spans = []
            centered = local_pts - grasp.translation[None, :]
            for idx in rem:
                proj = centered @ cols[idx]
                span = float(np.percentile(proj, 95.0) - np.percentile(proj, 5.0))
                spans.append(span)
            errs = [abs(s - max(0.005, float(grasp.width))) for s in spans]
            opening_idx = rem[int(np.argmin(errs))]
        else:
            opening_idx = rem[0]
        approach_idx = rem[0] if rem[1] == opening_idx else rem[1]
        return cols[opening_idx], cols[approach_idx], body_axis

    def width_match(self, local_pts: np.ndarray, center: np.ndarray, opening_axis: np.ndarray, width: float) -> Tuple[float, float]:
        if local_pts.shape[0] < 6:
            return 0.0, 0.0
        proj = (local_pts - center[None, :]) @ opening_axis
        span = float(np.percentile(proj, 95.0) - np.percentile(proj, 5.0))
        sigma = max(1e-4, self.width_match_sigma)
        score = float(math.exp(-abs(span - max(0.005, width)) / sigma))
        return score, span

    def position_scores(self, center: np.ndarray) -> Tuple[float, float]:
        if self.body_axis is None or self.body_center is None:
            return 0.0, 0.0
        rel = center - self.body_center
        t = float(np.dot(rel, self.body_axis))
        ortho = rel - t * self.body_axis
        radial = float(np.linalg.norm(ortho))
        radial_score = max(0.0, 1.0 - radial / max(self.body_radius * 1.5, 1e-4))

        denom = max(1e-4, self.body_max - self.body_min)
        norm_t = (t - self.body_min) / denom
        mid_score = max(0.0, 1.0 - abs(norm_t - 0.55) / 0.45)
        return radial_score, mid_score

    def background_clearance(self, center: np.ndarray) -> float:
        if self.background_points is None or self.background_points.shape[0] == 0:
            return float('inf')
        d = np.linalg.norm(self.background_points - center[None, :], axis=1)
        return float(np.min(d))

    def background_collision_count(self, grasp: RankedGrasp) -> int:
        """
        Conservative background collision check using a simplified parallel-jaw
        gripper volume. This treats background_pc as obstacles. Target/object_pc
        is intentionally not used here.
        """
        if not self.use_gripper_volume_collision:
            return 0
        if self.background_points is None or self.background_points.shape[0] == 0:
            return 0
        if grasp.opening_axis is None or grasp.approach_axis is None or grasp.body_axis is None:
            return 0

        pts = self.background_points
        limit = int(self.background_collision_sample_limit)
        if limit > 0 and pts.shape[0] > limit:
            idx = np.linspace(0, pts.shape[0] - 1, limit).astype(np.int64)
            pts = pts[idx]

        opening = grasp.opening_axis / max(np.linalg.norm(grasp.opening_axis), 1e-8)
        approach = grasp.approach_axis / max(np.linalg.norm(grasp.approach_axis), 1e-8)
        body = grasp.body_axis / max(np.linalg.norm(grasp.body_axis), 1e-8)

        rel = pts - grasp.translation[None, :]
        x = rel @ opening
        y = rel @ approach
        z = rel @ body

        margin = max(0.0, float(self.background_collision_margin))
        jaw_width = float(self.collision_gripper_width) if self.collision_use_fixed_width else max(0.0, float(grasp.width))
        jaw_width = max(0.0, jaw_width)
        half_w = 0.5 * jaw_width
        finger_len = max(0.001, float(self.collision_finger_length))
        palm_depth = max(0.0, float(self.collision_palm_depth))
        tail_len = max(0.0, float(self.collision_tail_length))
        thick = max(0.001, float(self.collision_finger_thickness))

        left_finger = (
            (np.abs(x + half_w) <= thick + margin)
            & (y >= -palm_depth - margin)
            & (y <= finger_len - palm_depth + margin)
            & (np.abs(z) <= thick + margin)
        )
        right_finger = (
            (np.abs(x - half_w) <= thick + margin)
            & (y >= -palm_depth - margin)
            & (y <= finger_len - palm_depth + margin)
            & (np.abs(z) <= thick + margin)
        )
        palm = (
            (np.abs(x) <= half_w + thick + margin)
            & (y >= -palm_depth - tail_len - margin)
            & (y <= -palm_depth + thick + margin)
            & (np.abs(z) <= thick + margin)
        )
        close_core = (
            (np.abs(x) <= half_w + margin)
            & (np.abs(y) <= max(finger_len, palm_depth) + margin)
            & (np.abs(z) <= thick + margin)
        )

        collision = left_finger | right_finger | palm | close_core
        return int(np.count_nonzero(collision))

    def log_top_candidates(self, grasps: Sequence[RankedGrasp], topk: int = 10) -> None:
        self.get_logger().info(f'[final] top {min(topk, len(grasps))} candidates:')
        for i, g in enumerate(grasps[:topk], start=1):
            t = g.translation
            self.get_logger().info(
                f'  #{i:02d} score={g.score:.4f} rank={g.rank:.4f} width={100.0*g.width:.2f}cm '
                f'local_span={100.0*g.local_span:.2f}cm gate={int(g.width_gate_ok)} limit={100.0*g.local_width_limit:.2f}cm '
                f'target={int(g.target_ok)} mask={int(g.mask_ok)} target_d={100.0*g.target_dist:.1f}cm '
                f'ok_theta={math.degrees(g.okrobot_theta):.1f}deg ok_score={g.okrobot_score:.4f} horiz={g.horizontal_score:.3f} '
                f'side_dot={g.arm_side_dot:.3f} top_down={g.top_down_score:.3f} top_dot={g.top_down_dot:.3f} '
                f'body_align={g.body_align:.3f} open_perp={g.opening_perp:.3f} '
                f'appr_perp={g.approach_perp:.3f} width_match={g.width_match:.3f} '
                f'radial={g.radial_score:.3f} mid={g.mid_score:.3f} '
                f'bg={g.bg_clearance:.3f} bg_col={g.bg_collision_count} '
                f'pos=({t[0]:.3f},{t[1]:.3f},{t[2]:.3f})'
            )

    def pointcloud2_to_numpy(self, msg: PointCloud2) -> Tuple[np.ndarray, np.ndarray]:
        field_names = [f.name for f in msg.fields]
        use_rgb_pack = 'rgb' in field_names
        use_separate_rgb = all(c in field_names for c in ('r', 'g', 'b'))

        if use_rgb_pack:
            raw = list(point_cloud2.read_points(msg, field_names=['x', 'y', 'z', 'rgb'], skip_nans=True))
            if len(raw) == 0:
                return np.empty((0, 3), np.float32), np.empty((0, 3), np.float32)
            xyz = np.array([[float(p[0]), float(p[1]), float(p[2])] for p in raw], dtype=np.float32)
            rgb = np.array([self.unpack_rgb_float(p[3]) for p in raw], dtype=np.float32)
            return xyz, rgb

        if use_separate_rgb:
            raw = list(point_cloud2.read_points(msg, field_names=['x', 'y', 'z', 'r', 'g', 'b'], skip_nans=True))
            if len(raw) == 0:
                return np.empty((0, 3), np.float32), np.empty((0, 3), np.float32)
            xyz = np.array([[float(p[0]), float(p[1]), float(p[2])] for p in raw], dtype=np.float32)
            rgb = np.array([[float(p[3]) / 255.0, float(p[4]) / 255.0, float(p[5]) / 255.0] for p in raw], dtype=np.float32)
            return xyz, rgb

        raw = list(point_cloud2.read_points(msg, field_names=['x', 'y', 'z'], skip_nans=True))
        if len(raw) == 0:
            return np.empty((0, 3), np.float32), np.empty((0, 3), np.float32)
        xyz = np.array([[float(p[0]), float(p[1]), float(p[2])] for p in raw], dtype=np.float32)
        rgb = np.full((xyz.shape[0], 3), 0.5, dtype=np.float32)
        return xyz, rgb

    @staticmethod
    def unpack_rgb_float(rgb_float) -> Tuple[float, float, float]:
        arr = np.asarray([rgb_float], dtype=np.float32)
        packed = arr.view(np.uint32)[0]
        r = (packed >> 16) & 0xFF
        g = (packed >> 8) & 0xFF
        b = packed & 0xFF
        return (r / 255.0, g / 255.0, b / 255.0)

    @staticmethod
    def voxel_downsample(points: np.ndarray, colors: np.ndarray, voxel_size: float) -> Tuple[np.ndarray, np.ndarray]:
        if points.shape[0] == 0 or voxel_size <= 0.0:
            return points, colors
        keys = np.floor(points / voxel_size).astype(np.int64)
        _, unique_idx = np.unique(keys, axis=0, return_index=True)
        unique_idx = np.sort(unique_idx)
        return points[unique_idx], colors[unique_idx]

    def compute_lims(self, points: np.ndarray) -> List[float]:
        mins = points.min(axis=0)
        maxs = points.max(axis=0)
        return [
            float(mins[0] - self.crop_margin_x), float(maxs[0] + self.crop_margin_x),
            float(mins[1] - self.crop_margin_y), float(maxs[1] + self.crop_margin_y),
            float(mins[2] - self.crop_margin_z), float(maxs[2] + self.crop_margin_z),
        ]

    def convert_grasp_group(self, gg) -> List[RankedGrasp]:
        out: List[RankedGrasp] = []
        try:
            n = len(gg)
        except Exception:
            n = 0
        for i in range(n):
            g = gg[i]
            score = float(getattr(g, 'score', 0.0))
            width = float(getattr(g, 'width', 0.0))
            translation = np.asarray(getattr(g, 'translation', np.zeros(3)), dtype=np.float32).reshape(3)
            rotation = np.asarray(getattr(g, 'rotation_matrix', np.eye(3)), dtype=np.float32).reshape(3, 3)
            out.append(RankedGrasp(score=score, width=width, translation=translation, rotation_matrix=rotation))
        return out

    def publish_outputs(self, header, grasps: Sequence[RankedGrasp]) -> None:
        pose_array = PoseArray()
        pose_array.header = header
        pose_array.poses = [self.to_pose(g.translation, g.rotation_matrix) for g in grasps]
        self.grasps_pub.publish(pose_array)

        best = grasps[0]
        best_pose = PoseStamped()
        best_pose.header = header
        best_pose.pose = self.to_pose(best.translation, best.rotation_matrix)
        self.best_pub.publish(best_pose)
        self.best_pose_raw_pub.publish(best_pose)

        width_msg = Float32()
        width_msg.data = float(best.width)
        self.best_width_pub.publish(width_msg)

        score_msg = Float32()
        score_msg.data = float(best.score)
        self.best_score_pub.publish(score_msg)

        self.publish_best_contact_point(best.translation, header)
        self.publish_best_contact_marker(best, header)
        self.publish_best_marker(best, header)
        self.publish_best_axes(best, header)

        all_markers = MarkerArray()
        all_markers.markers.append(self.make_delete_all_marker(header))
        for idx, grasp in enumerate(grasps):
            all_markers.markers.extend(self.make_candidate_markers(header, idx, grasp, best=False))
        self.all_markers_pub.publish(all_markers)

        topk = self.marker_topk if self.marker_topk > 0 else len(grasps)
        final_markers = MarkerArray()
        final_markers.markers.append(self.make_delete_all_marker(header))
        for idx, grasp in enumerate(grasps[:topk]):
            final_markers.markers.extend(self.make_candidate_markers(header, idx, grasp, best=(idx == 0)))
        self.markers_pub.publish(final_markers)

        t = best.translation
        vis_width = float(getattr(self, "visual_gripper_width", 0.10)) if bool(getattr(self, "use_fixed_visual_gripper_width", True)) else float(best.width)
        self.get_logger().info(
            f'grasps={len(grasps)} best_score={best.score:.4f} best_xyz=({t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f}) '
            f'pred_width={best.width:.4f} vis_width={vis_width:.4f} rank={best.rank:.4f} '
            f'ok_theta={math.degrees(best.okrobot_theta):.1f}deg ok_score={best.okrobot_score:.4f} side_dot={best.arm_side_dot:.3f}'
        )

    def publish_empty_markers(self, header) -> None:
        if header is None:
            return
        markers = MarkerArray()
        markers.markers.append(self.make_delete_all_marker(header))
        self.markers_pub.publish(markers)
        self.all_markers_pub.publish(markers)
        self.best_axes_pub.publish(markers)

        delete_best = self.make_delete_all_marker(header)
        self.best_marker_pub.publish(delete_best)
        self.best_contact_marker_pub.publish(delete_best)

    def make_delete_all_marker(self, header) -> Marker:
        marker = Marker()
        marker.header = header
        marker.action = Marker.DELETEALL
        return marker

    def to_pose(self, translation: np.ndarray, rotation_matrix: np.ndarray) -> Pose:
        quat_xyzw = R.from_matrix(rotation_matrix).as_quat()
        pose = Pose()
        pose.position.x = float(translation[0])
        pose.position.y = float(translation[1])
        pose.position.z = float(translation[2])
        pose.orientation.x = float(quat_xyzw[0])
        pose.orientation.y = float(quat_xyzw[1])
        pose.orientation.z = float(quat_xyzw[2])
        pose.orientation.w = float(quat_xyzw[3])
        return pose

    def grasp_contact_point(self, grasp: RankedGrasp) -> np.ndarray:
        center = grasp.translation
        return center.astype(np.float32)

    def transform_rotation_for_vis(self, rotation_matrix: np.ndarray) -> np.ndarray:
        if not self.use_visualization_rotation_fix:
            return rotation_matrix
        return (rotation_matrix @ self.r_fix).astype(np.float32)

    def build_gripper_wire_segments(self, grasp: RankedGrasp) -> List[Tuple[np.ndarray, np.ndarray]]:
        center = grasp.translation.astype(np.float32)
        if grasp.opening_axis is None or grasp.approach_axis is None or grasp.body_axis is None:
            opening, approach, body = self.infer_axes(grasp, self.local_object_points(center))
        else:
            opening, approach, body = grasp.opening_axis, grasp.approach_axis, grasp.body_axis

        opening = opening / max(np.linalg.norm(opening), 1e-8)
        approach = approach / max(np.linalg.norm(approach), 1e-8)
        body = body / max(np.linalg.norm(body), 1e-8)

        use_fixed = getattr(self, 'use_fixed_visual_gripper_width', True)
        vis_width = getattr(self, 'visual_gripper_width', 0.10)
        if use_fixed:
            jaw_width = max(0.0, float(vis_width))
        else:
            jaw_width = max(0.0, float(grasp.width))
        finger_len = float(self.gripper_finger_length)
        palm_depth = float(self.gripper_palm_depth)
        tail_len = float(self.gripper_tail_length)
        knuckle_fwd = float(self.gripper_knuckle_forward)
        tip_bar_half = 0.5 * float(self.gripper_finger_thickness)

        palm_center = center - palm_depth * approach
        tail = palm_center - tail_len * approach

        left_root = palm_center - 0.5 * jaw_width * opening
        right_root = palm_center + 0.5 * jaw_width * opening
        left_knuckle = left_root + knuckle_fwd * approach
        right_knuckle = right_root + knuckle_fwd * approach
        left_tip = left_root + finger_len * approach
        right_tip = right_root + finger_len * approach

        left_tip_a = left_tip - tip_bar_half * body
        left_tip_b = left_tip + tip_bar_half * body
        right_tip_a = right_tip - tip_bar_half * body
        right_tip_b = right_tip + tip_bar_half * body

        return [
            (tail, palm_center),
            (left_root, right_root),
            (left_root, left_knuckle),
            (right_root, right_knuckle),
            (left_knuckle, left_tip),
            (right_knuckle, right_tip),
            (left_tip_a, left_tip_b),
            (right_tip_a, right_tip_b),
        ]

    def segments_to_points(self, segments: Sequence[Tuple[np.ndarray, np.ndarray]]) -> List[Point]:
        pts: List[Point] = []
        for a, b in segments:
            pa = Point(x=float(a[0]), y=float(a[1]), z=float(a[2]))
            pb = Point(x=float(b[0]), y=float(b[1]), z=float(b[2]))
            pts.extend([pa, pb])
        return pts

    def marker_lifetime_msg(self):
        return Duration(seconds=self.marker_lifetime_sec).to_msg()

    def rank_color(self, score: float, best: bool = False) -> ColorRGBA:
        s = max(0.0, min(1.0, float(score)))
        c = ColorRGBA()
        if best:
            c.r = 1.0; c.g = 0.25; c.b = 0.0; c.a = 1.0
        else:
            c.r = 1.0 - s; c.g = s; c.b = 1.0 - 0.45 * s; c.a = self.marker_alpha
        return c

    def make_line_marker(self, header, marker_id: int, ns: str, segments, line_width: float, color: ColorRGBA) -> Marker:
        marker = Marker()
        marker.header = header
        marker.ns = ns
        marker.id = int(marker_id)
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD
        marker.scale.x = float(line_width)
        marker.color = color
        marker.points = self.segments_to_points(segments)
        marker.lifetime = self.marker_lifetime_msg()
        return marker

    def make_contact_marker(self, header, marker_id: int, ns: str, point_xyz: np.ndarray, scale: float, color: ColorRGBA) -> Marker:
        marker = Marker()
        marker.header = header
        marker.ns = ns
        marker.id = int(marker_id)
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = float(point_xyz[0])
        marker.pose.position.y = float(point_xyz[1])
        marker.pose.position.z = float(point_xyz[2])
        marker.pose.orientation.w = 1.0
        marker.scale.x = float(scale)
        marker.scale.y = float(scale)
        marker.scale.z = float(scale)
        marker.color = color
        marker.lifetime = self.marker_lifetime_msg()
        return marker

    def make_axis_arrow(self, header, marker_id: int, ns: str, origin: np.ndarray, axis: np.ndarray, length: float, color: Tuple[float,float,float]) -> Marker:
        axis = axis / max(np.linalg.norm(axis), 1e-8)
        marker = Marker()
        marker.header = header
        marker.ns = ns
        marker.id = marker_id
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        marker.scale.x = 0.004
        marker.scale.y = 0.008
        marker.scale.z = 0.012
        marker.color = ColorRGBA(r=float(color[0]), g=float(color[1]), b=float(color[2]), a=0.95)
        p0 = Point(x=float(origin[0]), y=float(origin[1]), z=float(origin[2]))
        p1v = origin + length * axis
        p1 = Point(x=float(p1v[0]), y=float(p1v[1]), z=float(p1v[2]))
        marker.points = [p0, p1]
        marker.lifetime = self.marker_lifetime_msg()
        return marker

    def make_candidate_markers(self, header, idx: int, grasp: RankedGrasp, best: bool = False) -> List[Marker]:
        segments = self.build_gripper_wire_segments(grasp)
        contact = self.grasp_contact_point(grasp)

        if best:
            line_width = self.best_gripper_line_width
            contact_scale = self.best_contact_scale
            line_color = self.rank_color(grasp.rank, best=True)
            contact_color = ColorRGBA(r=0.0, g=1.0, b=1.0, a=1.0)
            ns_line = 'best_grasp'; ns_contact = 'best_contact'
        else:
            line_width = self.candidate_gripper_line_width
            contact_scale = self.candidate_contact_scale
            line_color = self.rank_color(grasp.rank, best=False)
            contact_color = ColorRGBA(r=0.0, g=0.7, b=1.0, a=0.85)
            ns_line = 'gripper_candidates'; ns_contact = 'contact_candidates'

        line_marker = self.make_line_marker(header, idx * 2, ns_line, segments, line_width, line_color)
        contact_marker = self.make_contact_marker(header, idx * 2 + 1, ns_contact, contact, contact_scale, contact_color)
        return [line_marker, contact_marker]

    def publish_best_marker(self, best: RankedGrasp, header) -> None:
        self.best_marker_pub.publish(self.make_candidate_markers(header, 0, best, best=True)[0])

    def publish_best_contact_marker(self, best: RankedGrasp, header) -> None:
        self.best_contact_marker_pub.publish(self.make_candidate_markers(header, 0, best, best=True)[1])

    def publish_best_contact_point(self, point_xyz: np.ndarray, header) -> None:
        msg = PointStamped()
        msg.header = header
        msg.point.x = float(point_xyz[0])
        msg.point.y = float(point_xyz[1])
        msg.point.z = float(point_xyz[2])
        self.best_contact_point_pub.publish(msg)

    def publish_best_axes(self, best: RankedGrasp, header) -> None:
        ma = MarkerArray()
        ma.markers.append(self.make_delete_all_marker(header))
        origin = best.translation.astype(np.float32)
        if best.opening_axis is None or best.approach_axis is None or best.body_axis is None:
            opening, approach, body = self.infer_axes(best, self.local_object_points(origin))
        else:
            opening, approach, body = best.opening_axis, best.approach_axis, best.body_axis
        ma.markers.append(self.make_axis_arrow(header, 10, 'axes', origin, opening, self.axis_marker_length, (1.0, 0.0, 0.0)))
        ma.markers.append(self.make_axis_arrow(header, 11, 'axes', origin, approach, self.axis_marker_length, (0.0, 1.0, 0.0)))
        ma.markers.append(self.make_axis_arrow(header, 12, 'axes', origin, body, self.axis_marker_length, (0.0, 0.4, 1.0)))
        if self.body_axis is not None and self.body_center is not None:
            ma.markers.append(self.make_axis_arrow(header, 13, 'bottle_axis', self.body_center, self.body_axis, self.axis_marker_length, (1.0, 1.0, 0.0)))
        self.best_axes_pub.publish(ma)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = AnyGraspFromTopicNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()