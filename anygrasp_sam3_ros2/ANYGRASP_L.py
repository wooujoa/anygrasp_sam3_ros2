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
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from std_msgs.msg import Bool, ColorRGBA, Float32
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
    front_side_dot: float = 0.0
    front_side_ok: bool = True
    bottom_up_z_dot: float = 0.0
    bottom_up_ok: bool = True
    ee_x_up_dot: float = 0.0
    ee_y_obj_dot: float = 0.0
    ee_axis_score: float = 0.0
    ee_axis_ok: bool = True
    target_dist: float = float('inf')
    target_ok: bool = False
    local_span: float = 0.0
    local_width_limit: float = 0.0
    width_gate_ok: bool = True


class AnyGraspMaster2Node(Node):
    def __init__(self) -> None:
        super().__init__('anygrasp_l_master2_node')

        # master_2 control. Keep the communication order unchanged:
        # start -> wait fresh SAM3 topics -> run AnyGrasp once -> publish finish.
        self.declare_parameter('start_topic', '/anygrasp_l_start')
        self.declare_parameter('finish_topic', '/anygrasp_l_finish')
        self.declare_parameter('require_background_data', True)

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
        self.declare_parameter('scene_cloud_topic', '/sam3_l/full_scene_pc')
        self.declare_parameter('target_cloud_topic', '/sam3_l/target_pc')
        self.declare_parameter('object_cloud_topic', '/sam3_l/object_pc')
        self.declare_parameter('background_cloud_topic', '/sam3_l/background_pc')
        self.declare_parameter('target_mask_topic', '/sam3_l/target_mask')
        self.declare_parameter('camera_info_topic', '/camera_l/camera_l/aligned_depth_to_color/camera_info')
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
        self.declare_parameter('mask_filter_margin_px', 0)
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
        self.declare_parameter('enable_local_width_gate', False)
        self.declare_parameter('enable_object_width_gate', False)
        self.declare_parameter('width_safety_margin', 0.010)
        self.declare_parameter('object_width_ratio_limit', 1.00)
        self.declare_parameter('rank_threshold', -10.0)
        self.declare_parameter('prefer_mid_body', True)
        self.declare_parameter('prefer_side_grasp', False)

        # OK-Robot style grasp heuristic.
        # OK-Robot filters grasps by the language mask and then ranks with
        # a graspness-vs-horizontal-grasp heuristic. In practice, for this
        # camera-frame node, we make the floor normal configurable.
        self.declare_parameter('enable_okrobot_heuristic', False)
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
        # Left arm: reject grasps whose gripper face axis points to robot-right (-Y).
        # Left arm : reject grasps whose gripper face axis points to robot-left  (+Y).
        # It does NOT force the gripper to look strongly toward the opposite side.
        self.declare_parameter('enable_arm_side_filter', False)
        self.declare_parameter('arm_side', 'left')  # 'right' or 'left'
        # Deprecated old name kept for launch compatibility. Not used for scoring.
        self.declare_parameter('arm_side_min_dot', 0.0)
        # Reject only when dot with the desired inward side is smaller than -this value.
        # Example for left arm: desired is +Y. If face_axis_y < -0.05, it is looking right and rejected.
        self.declare_parameter('arm_side_reject_dot', 0.05)
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('base_frame_candidates', ['base_link', 'lift_link', 'arm_base_link'])
        self.declare_parameter('gripper_frame', 'gripper_l_rh_p12_rn_base')
        self.declare_parameter('camera_frame', 'camera_l_color_optical_frame')
        self.declare_parameter('tf_timeout_sec', 0.05)
        self.declare_parameter('apply_anygrasp_pose_frame_alignment_for_filter', True)
        self.declare_parameter('auto_flip_pose_z_180_if_x_points_down_for_filter', True)
        self.declare_parameter('x_axis_downward_flip_threshold_for_filter', 0.0)
        self.declare_parameter('gripper_face_axis_index', 1)  # final gripper +Y is the object-facing axis
        self.declare_parameter('gripper_face_axis_sign', 1.0)
        self.declare_parameter('arm_side_filter_penalty', 6.0)
        self.declare_parameter('hard_reject_wrong_arm_side', False)

        # Top-down preference for shelf/table objects.
        # This is a ranking bonus only, not a hard filter.
        # It prefers gripper-facing direction from above to below, because
        # bottom-up approaches can collide with the shelf/table surface.
        self.declare_parameter('enable_top_down_bonus', False)
        self.declare_parameter('top_down_bonus_weight', 0.0)
        self.declare_parameter('top_down_axis_index', 2)
        self.declare_parameter('top_down_axis_sign', 1.0)
        self.declare_parameter('top_down_desired_z_sign', -1.0)  # -1: above -> below, +1: below -> above
        self.declare_parameter('top_down_penalize_bottom_up', False)
        self.declare_parameter('bottom_up_penalty_weight', 0.20)

        # Front/back feasibility filter in base_link.
        # This is a hard reject filter, not a ranking bonus.
        # Purpose: reject grasps that approach from the back/shelf-inside side.
        # Assumption: base_link +X is robot-forward / shelf-inner direction.
        # If the final gripper face axis points too strongly toward +X, reject it.
        self.declare_parameter('enable_front_side_filter', False)
        self.declare_parameter('front_side_axis_index', 2)
        self.declare_parameter('front_side_axis_sign', 1.0)
        self.declare_parameter('front_side_reject_x_dot', 0.20)
        self.declare_parameter('hard_reject_back_side', False)

        # Bottom-up feasibility filter in base_link.
        # This is a hard reject filter, not a ranking bonus.
        # Purpose: reject grasps that look from below to above, because shelf/table
        # objects can collide with the supporting surface when approached bottom-up.
        # Assumption: base_link +Z is up. If the final gripper face axis points
        # too strongly toward +Z, reject it.
        self.declare_parameter('enable_bottom_up_filter', False)
        self.declare_parameter('bottom_up_axis_index', 2)
        self.declare_parameter('bottom_up_axis_sign', 1.0)
        self.declare_parameter('bottom_up_reject_z_dot', 0.05)
        self.declare_parameter('hard_reject_bottom_up', False)

        # End-effector axis preference in final base_link orientation.
        # Based on RViz TF check for gripper_l_rh_p12_rn_base:
        #   +X axis: camera direction. Desired to point upward (+Z in base_link).
        #   +Y axis: gripper/object-facing direction. Desired to point toward object center.
        # This is the main orientation ranking term. It is soft by default,
        # so raw candidates are not all killed when TF/segmentation is noisy.
        self.declare_parameter('enable_ee_axis_preference', True)
        self.declare_parameter('ee_axis_bonus_weight', 1.50)
        self.declare_parameter('ee_x_up_min_dot', 0.00)
        self.declare_parameter('ee_y_to_object_min_dot', 0.00)
        self.declare_parameter('hard_reject_bad_ee_axis', False)

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
        self.declare_parameter('target_filter_radius', 0.035)
        self.declare_parameter('hard_reject_background', True)
        self.declare_parameter('use_gripper_volume_collision', True)
        self.declare_parameter('max_background_collision_points', 45)
        self.declare_parameter('background_collision_margin', 0.001)
        self.declare_parameter('background_collision_sample_limit', 50000)
        self.declare_parameter('collision_use_fixed_width', True)
        self.declare_parameter('collision_gripper_width', 0.10)
        self.declare_parameter('collision_finger_length', 0.055)
        self.declare_parameter('collision_palm_depth', 0.030)
        self.declare_parameter('collision_tail_length', 0.020)
        self.declare_parameter('collision_finger_thickness', 0.010)
        self.declare_parameter('verbose_filter_log', True)

        # outputs
        self.declare_parameter('best_grasp_topic', '/anygrasp_l/best_grasp')
        self.declare_parameter('best_pose_raw_topic', '/anygrasp_l/best_pose_raw')
        self.declare_parameter('best_width_topic', '/anygrasp_l/best_width')
        # Raw AnyGrasp confidence for the selected best grasp.
        # This remains an internal scalar topic. The final custom ObjectGrasp
        # is published only by the calib node after frame conversion.
        self.declare_parameter('best_score_topic', '/anygrasp_l/best_score')
        self.declare_parameter('grasps_topic', '/anygrasp_l/grasps')
        self.declare_parameter('markers_topic', '/anygrasp_l/grasp_markers')
        self.declare_parameter('all_markers_topic', '/anygrasp_l/all_grasp_markers')
        self.declare_parameter('best_marker_topic', '/anygrasp_l/best_pose_marker')
        self.declare_parameter('best_contact_marker_topic', '/anygrasp_l/best_contact_marker')
        self.declare_parameter('best_contact_point_topic', '/anygrasp_l/best_contact_point')
        self.declare_parameter('best_axes_topic', '/anygrasp_l/best_axes_markers')

        # minimal internal visualization geometry parameters.
        # Actual RViz markers are generated only in CALI_D405 after base_link transform.
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

        # Raw inferred grasp visualization in base_frame.
        # This shows grasps generated by AnyGrasp after score threshold, BEFORE
        # target/background/arm-side/front-side/bottom-up hard filters.
        # It is debug visualization only and does not affect ranking or ObjectGrasp.
        self.declare_parameter('raw_inferred_grasp_markers_base_topic', '/anygrasp_l/raw_inferred_grasp_markers_base')
        self.declare_parameter('raw_inferred_marker_topk', 0)
        self.declare_parameter('raw_inferred_marker_alpha', 0.30)
        self.declare_parameter('raw_inferred_line_width', 0.0014)


        self.start_topic = self.get_parameter('start_topic').value
        self.finish_topic = self.get_parameter('finish_topic').value
        self.require_background_data = bool(self.get_parameter('require_background_data').value)

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

        self.enable_front_side_filter = bool(self.get_parameter('enable_front_side_filter').value)
        self.front_side_axis_index = max(0, min(2, int(self.get_parameter('front_side_axis_index').value)))
        self.front_side_axis_sign = float(self.get_parameter('front_side_axis_sign').value)
        self.front_side_reject_x_dot = float(self.get_parameter('front_side_reject_x_dot').value)
        self.hard_reject_back_side = bool(self.get_parameter('hard_reject_back_side').value)

        self.enable_bottom_up_filter = bool(self.get_parameter('enable_bottom_up_filter').value)
        self.bottom_up_axis_index = max(0, min(2, int(self.get_parameter('bottom_up_axis_index').value)))
        self.bottom_up_axis_sign = float(self.get_parameter('bottom_up_axis_sign').value)
        self.bottom_up_reject_z_dot = float(self.get_parameter('bottom_up_reject_z_dot').value)
        self.hard_reject_bottom_up = bool(self.get_parameter('hard_reject_bottom_up').value)

        self.enable_ee_axis_preference = bool(self.get_parameter('enable_ee_axis_preference').value)
        self.ee_axis_bonus_weight = float(self.get_parameter('ee_axis_bonus_weight').value)
        self.ee_x_up_min_dot = float(self.get_parameter('ee_x_up_min_dot').value)
        self.ee_y_to_object_min_dot = float(self.get_parameter('ee_y_to_object_min_dot').value)
        self.hard_reject_bad_ee_axis = bool(self.get_parameter('hard_reject_bad_ee_axis').value)

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
        self.raw_inferred_grasp_markers_base_topic = str(self.get_parameter('raw_inferred_grasp_markers_base_topic').value)
        self.raw_inferred_marker_topk = int(self.get_parameter('raw_inferred_marker_topk').value)
        self.raw_inferred_marker_alpha = float(self.get_parameter('raw_inferred_marker_alpha').value)
        self.raw_inferred_line_width = float(self.get_parameter('raw_inferred_line_width').value)

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
        self.active = False
        self.already_processed = False

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

        self.qos_cmd = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )

        self.start_sub = self.create_subscription(Bool, self.start_topic, self.start_callback, self.qos_cmd)
        self.finish_pub = self.create_publisher(Bool, self.finish_topic, self.qos_cmd)

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
        self.best_contact_point_pub = self.create_publisher(PointStamped, self.get_parameter('best_contact_point_topic').value, 10)
        self.raw_inferred_markers_base_pub = self.create_publisher(
            MarkerArray,
            self.raw_inferred_grasp_markers_base_topic,
            10,
        )

        self.get_logger().info('ANYGRASP MASTER2 Node Ready (LEFT ARM, TARGET-ONLY CANDIDATE OUTPUT)')
        self.get_logger().info(f'start_topic={self.start_topic}')
        self.get_logger().info(f'finish_topic={self.finish_topic}')
        self.get_logger().info(f'require_background_data={self.require_background_data}')
        self.get_logger().info(f'input_topic={input_topic}')
        self.get_logger().info(f'object_cloud_topic={self.object_cloud_topic}')
        self.get_logger().info(f'background_cloud_topic={self.background_cloud_topic}')
        self.get_logger().info(f'target_mask_topic={self.target_mask_topic}')
        self.get_logger().info(f'use_scene_cloud={self.use_scene_cloud}')
        self.get_logger().info(f'best_pose_raw_topic={self.get_parameter("best_pose_raw_topic").value}')
        self.get_logger().info(f'best_width_topic={self.get_parameter("best_width_topic").value}')
        self.get_logger().info(f'best_score_topic={self.get_parameter("best_score_topic").value}')
        self.get_logger().info(f'raw_inferred_grasp_markers_base_topic={self.raw_inferred_grasp_markers_base_topic}')
        self.get_logger().info(f'raw_inferred_marker_topk={self.raw_inferred_marker_topk}')
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
        self.get_logger().info(f'enable_front_side_filter={self.enable_front_side_filter}')
        self.get_logger().info(f'front_side_axis_index={self.front_side_axis_index} sign={self.front_side_axis_sign:+.1f}')
        self.get_logger().info(f'front_side_reject_x_dot={self.front_side_reject_x_dot:.3f}  # reject if face_x > this')
        self.get_logger().info(f'enable_bottom_up_filter={self.enable_bottom_up_filter}')
        self.get_logger().info(f'bottom_up_axis_index={self.bottom_up_axis_index} sign={self.bottom_up_axis_sign:+.1f}')
        self.get_logger().info(f'bottom_up_reject_z_dot={self.bottom_up_reject_z_dot:.3f}  # reject if face_z > this')
        self.get_logger().info(f'hard_reject_bottom_up={self.hard_reject_bottom_up}')
        self.get_logger().info(f'enable_ee_axis_preference={self.enable_ee_axis_preference}')
        self.get_logger().info(f'ee_axis_bonus_weight={self.ee_axis_bonus_weight:.3f}')
        self.get_logger().info(f'ee_x_up_min_dot={self.ee_x_up_min_dot:.3f}  # gripper +X/camera axis should point up')
        self.get_logger().info(f'ee_y_to_object_min_dot={self.ee_y_to_object_min_dot:.3f}  # gripper +Y should point toward object')
        self.get_logger().info(f'hard_reject_bad_ee_axis={self.hard_reject_bad_ee_axis}')
        self.get_logger().info(f'hard_reject_back_side={self.hard_reject_back_side}')

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

            # Debug visualization: publish raw AnyGrasp candidates in base_frame
            # before any hard filters. This is intentionally independent from
            # /anygrasp_l/grasps, which still contains only surviving candidates.
            self.publish_raw_inferred_grasp_markers_base(msg.header, grasps)

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
        1st-filter-only version.

        This node now only performs the perception-side filter:
          1) AnyGrasp score threshold was already applied before this function.
          2) Keep candidates whose grasp center belongs to / is near the SAM3 target object.
          3) Do NOT remove candidates by robot-side feasibility:
             - no background/gripper-volume collision rejection here
             - no arm-side/front-side/bottom-up hard reject here
             - no IK/reachability decision here

        The output list is sorted by perception confidence so candidates[0] remains
        the best AnyGrasp/CALI-side candidate. The robot arm node is expected to
        run collision checking, IK, and final selection.
        """
        stats = {
            'raw': len(grasps),
            'target_reject': 0,
            'background_reject': 0,
            'width_soft_reject': 0,
            'arm_side_reject': 0,
            'front_side_reject': 0,
            'bottom_up_reject': 0,
            'ee_axis_reject': 0,
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

            # The only hard filter kept in AnyGrasp node:
            # candidate must be on / close to the SAM3-selected target.
            g.target_ok, g.mask_ok, g.target_dist = self.target_membership(g.translation)
            # Strict target filtering must never be bypassed by force_keep.
            # force_keep is only for ranking-threshold fallback, not for reviving
            # candidates outside the SAM3 target mask.
            if self.hard_filter_to_target and (not g.target_ok):
                stats['target_reject'] += 1
                continue

            # Keep these values only for logging/debug. They are NOT used to reject.
            local_pts = self.local_object_points(g.translation)
            opening_axis, approach_axis, body_axis = self.infer_axes(g, local_pts)
            g.opening_axis = opening_axis
            g.approach_axis = approach_axis
            g.body_axis = body_axis
            g.width_match, g.local_span = self.width_match(local_pts, g.translation, opening_axis, g.width)
            g.local_width_limit = max(0.0, float(self.get_parameter('max_gripper_width').value) - self.width_safety_margin)
            g.width_gate_ok = True

            # Ranking is only for ordering candidates. It is not the final execution decision.
            # Keep AnyGrasp score dominant, and use tiny tie breakers so candidates[0]
            # remains the strongest perception candidate.
            rank = float(g.score)
            rank += 0.001 * float(g.mask_ok)
            if np.isfinite(g.target_dist):
                rank -= 0.0005 * float(g.target_dist)

            g.rank = float(rank)
            ranked.append(g)

        ranked.sort(key=lambda x: x.rank, reverse=True)
        stats['kept'] = len(ranked)
        self.last_filter_stats = stats

        if self.verbose_filter_log:
            self.get_logger().info(
                '[filter_target_only] '
                f"raw={stats['raw']} kept={stats['kept']} "
                f"target_reject={stats['target_reject']} "
                f"target_data={stats['target_data_available']} "
                f"background_data={stats['background_available']} "
                'robot_collision_ik_selection=delegated_to_arm_node'
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

    def transform_camera_point_to_base_for_filter(self, p_cam: np.ndarray) -> Optional[np.ndarray]:
        """Transform a point expressed in camera optical frame to base_link for filtering/ranking."""
        try:
            p = np.asarray(p_cam, dtype=np.float64).reshape(3)
            t = self.resolve_base_from_gripper_for_filter()
            if t is None:
                return None
            T_gripper_to_base = self.make_transform_matrix_from_tf(t)
            T = T_gripper_to_base @ self.T_cam_to_gripper
            hp = np.ones(4, dtype=np.float64)
            hp[:3] = p
            return (T @ hp)[:3]
        except Exception as exc:
            if self.verbose_filter_log:
                self.get_logger().warn(f'camera point to base transform failed: {repr(exc)}')
            return None

    def ee_axis_preference_score(self, grasp: RankedGrasp) -> Tuple[bool, float, float, float]:
        """
        Preferred final gripper orientation based on actual gripper_l_rh_p12_rn_base axes.

        From RViz TF inspection:
          - gripper +X is the camera direction, so it should point upward (+Z in base_link).
          - gripper +Y is the object-facing/grasp direction, so it should point toward the object center.

        Returns:
          ok           : True if optional hard thresholds are satisfied.
          score        : [0, 1] soft score used for ranking.
          x_up_dot     : dot(gripper +X, base +Z). Higher is better.
          y_obj_dot    : dot(gripper +Y, direction from grasp to object center). Higher is better.
        """
        if not self.enable_ee_axis_preference:
            return True, 0.0, 0.0, 0.0

        T_final_base = self.predict_final_pose_base_for_filter(grasp)
        if T_final_base is None:
            return True, 0.0, 0.0, 0.0

        x_axis = T_final_base[:3, 0].astype(np.float64)
        y_axis = T_final_base[:3, 1].astype(np.float64)
        x_axis /= max(float(np.linalg.norm(x_axis)), 1e-9)
        y_axis /= max(float(np.linalg.norm(y_axis)), 1e-9)

        x_up_dot = float(np.dot(x_axis, np.array([0.0, 0.0, 1.0], dtype=np.float64)))

        obj_center_cam = None
        if self.body_center is not None:
            obj_center_cam = np.asarray(self.body_center, dtype=np.float64)
        elif self.object_points is not None and self.object_points.shape[0] > 0:
            obj_center_cam = np.mean(self.object_points.astype(np.float64), axis=0)

        y_obj_dot = 0.0
        if obj_center_cam is not None:
            obj_center_base = self.transform_camera_point_to_base_for_filter(obj_center_cam)
            if obj_center_base is not None:
                grasp_pos_base = T_final_base[:3, 3].astype(np.float64)
                to_obj = obj_center_base - grasp_pos_base
                n = float(np.linalg.norm(to_obj))
                if n > 1e-4:
                    to_obj /= n
                    y_obj_dot = float(np.dot(y_axis, to_obj))

        score = 0.5 * max(0.0, x_up_dot) + 0.5 * max(0.0, y_obj_dot)
        ok = (x_up_dot >= self.ee_x_up_min_dot) and (y_obj_dot >= self.ee_y_to_object_min_dot)
        return bool(ok), float(score), float(x_up_dot), float(y_obj_dot)

    def front_side_feasibility(self, grasp: RankedGrasp) -> Tuple[bool, float]:
        """
        Reject back-side / shelf-inner grasps in final base_link orientation.

        This is different from arm-side filtering.
        - arm_side_feasibility removes right-arm grasps facing robot-right (-Y).
        - front_side_feasibility removes grasps whose face axis points too much
          toward base_link +X, which corresponds to approaching from behind the
          object / inside the shelf for the current setup assumption.

        Return:
          ok    : True if not clearly a back-side grasp
          x_dot : final face-axis x component in base_link
        """
        if not self.enable_front_side_filter:
            return True, 0.0

        T_final_base = self.predict_final_pose_base_for_filter(grasp)
        if T_final_base is None:
            # Do not kill all grasps when TF is temporarily unavailable.
            return True, 0.0

        axis_idx = max(0, min(2, int(self.front_side_axis_index)))
        face_axis = T_final_base[:3, axis_idx].astype(np.float64) * float(self.front_side_axis_sign)
        n = float(np.linalg.norm(face_axis))
        if n < 1e-9:
            return True, 0.0
        face_axis = face_axis / n

        # base_link +X is treated as back/shelf-inner direction.
        # If x_dot is too positive, the gripper is approaching from the object back side.
        x_dot = float(face_axis[0])
        return bool(x_dot <= self.front_side_reject_x_dot), x_dot

    def bottom_up_feasibility(self, grasp: RankedGrasp) -> Tuple[bool, float]:
        """
        Reject bottom-up grasps in final base_link orientation.

        base_link convention assumed here:
          +Z: up

        If the final gripper face axis points too much toward +Z, the gripper
        is effectively looking from below to above. For objects sitting on a
        shelf/table, this can drive the hand into the supporting surface, so it
        is treated as a hard reject.

        Return:
          ok    : True if not clearly bottom-up
          z_dot : final face-axis z component in base_link
        """
        if not self.enable_bottom_up_filter:
            return True, 0.0

        T_final_base = self.predict_final_pose_base_for_filter(grasp)
        if T_final_base is None:
            # Do not kill all grasps when TF is temporarily unavailable.
            return True, 0.0

        axis_idx = max(0, min(2, int(self.bottom_up_axis_index)))
        face_axis = T_final_base[:3, axis_idx].astype(np.float64) * float(self.bottom_up_axis_sign)
        n = float(np.linalg.norm(face_axis))
        if n < 1e-9:
            return True, 0.0
        face_axis = face_axis / n

        # base_link +Z is up. If z_dot is positive enough, the grasp is
        # below-to-above / bottom-up and should be rejected.
        z_dot = float(face_axis[2])
        return bool(z_dot <= self.bottom_up_reject_z_dot), z_dot

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

        For left arm, desired/inward side is +Y. A grasp is rejected only if
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
        # Dot with desired lateral direction. left arm desired/inward is +Y, left arm desired/inward is -Y.
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
        # Strict target-mask mode:
        # If target_filter_use_mask is enabled, the SAM3 target mask and camera_info
        # are mandatory. object_pc alone must NOT make a candidate valid, because
        # it can contain depth noise or background points.
        has_mask = self.target_filter_use_mask and self.target_mask is not None and self.camera_info is not None
        has_object = (
            self.target_filter_use_object_pc
            and self.object_points is not None
            and self.object_points.shape[0] > 0
        )
        if self.target_filter_use_mask:
            return bool(has_mask)
        return bool(has_object)

    def target_membership(self, xyz: np.ndarray) -> Tuple[bool, bool, float]:
        """
        Returns: (target_ok, mask_ok, target_dist).

        Strict SAM3 target-mask version.
        A grasp candidate is valid only when its grasp center projects inside
        the current SAM3 target_mask. The 3D object_pc distance is kept only
        for logging/ranking/debug, and it must never rescue a candidate whose
        projection is outside the target mask.
        """
        mask_ok = False
        target_dist = float('inf')

        if self.target_filter_use_object_pc and self.object_points is not None and self.object_points.shape[0] > 0:
            target_dist = self.distance_to_object(xyz)

        if self.target_filter_use_mask:
            if self.target_mask is None or self.camera_info is None:
                return False, False, target_dist
            mask_ok = self.is_inside_mask(xyz)
            if not mask_ok:
                return False, False, target_dist
            return True, True, target_dist

        # Mask filtering disabled: fall back to object_pc distance only.
        if self.target_filter_use_object_pc and self.object_points is not None and self.object_points.shape[0] > 0:
            object_ok = target_dist <= self.target_filter_radius
            return bool(object_ok), False, target_dist

        return (not self.require_target_data), False, target_dist

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

        # Strict mask check: do not accept candidates near the mask boundary.
        # A candidate must project onto a pixel that is actually inside target_mask.
        # This prevents object_pc/background noise from keeping grasps outside
        # the SAM3-selected object.
        return bool(self.target_mask[v, u])

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
                f'side_dot={g.arm_side_dot:.3f} front_dot={g.front_side_dot:.3f} '
                f'bottom_z={g.bottom_up_z_dot:.3f} '
                f'ee_score={g.ee_axis_score:.3f} x_up={g.ee_x_up_dot:.3f} y_obj={g.ee_y_obj_dot:.3f} '
                f'top_down={g.top_down_score:.3f} top_dot={g.top_down_dot:.3f} '
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

    def publish_raw_inferred_grasp_markers_base(self, header, grasps: Sequence[RankedGrasp]) -> None:
        """Publish AnyGrasp raw inferred candidates in base_frame before hard filters.

        This topic is for debugging only. It must not be used as the execution
        result because it includes candidates that may later be rejected by
        target/background/arm-side/front-side/bottom-up filters.
        """
        if header is None:
            return

        out_header = type(header)()
        out_header.stamp = header.stamp
        out_header.frame_id = self.base_frame

        ma = MarkerArray()
        delete_marker = Marker()
        delete_marker.header = out_header
        delete_marker.action = Marker.DELETEALL
        ma.markers.append(delete_marker)

        if len(grasps) == 0:
            self.raw_inferred_markers_base_pub.publish(ma)
            return

        topk = len(grasps)
        if self.raw_inferred_marker_topk > 0:
            topk = min(topk, int(self.raw_inferred_marker_topk))

        # Sort raw candidates by AnyGrasp score only, because filtering/ranking
        # has not been applied yet.
        raw_sorted = sorted(grasps, key=lambda g: float(g.score), reverse=True)[:topk]
        marker_id = 1
        for i, g in enumerate(raw_sorted):
            T_final_base = self.predict_final_pose_base_for_filter(g)
            if T_final_base is None:
                continue
            line_marker = Marker()
            line_marker.header = out_header
            line_marker.ns = 'raw_inferred_grasps_base'
            line_marker.id = marker_id
            marker_id += 1
            line_marker.type = Marker.LINE_LIST
            line_marker.action = Marker.ADD
            line_marker.scale.x = float(self.raw_inferred_line_width)
            line_marker.color = ColorRGBA(r=0.8, g=0.8, b=0.8, a=float(self.raw_inferred_marker_alpha))
            line_marker.points = self.gripper_wire_points_from_matrix(T_final_base)
            line_marker.lifetime = self.marker_lifetime_msg()
            ma.markers.append(line_marker)

            # Small center point so dense raw grasps are still readable.
            if i < 80:
                contact_marker = Marker()
                contact_marker.header = out_header
                contact_marker.ns = 'raw_inferred_grasp_centers_base'
                contact_marker.id = marker_id
                marker_id += 1
                contact_marker.type = Marker.SPHERE
                contact_marker.action = Marker.ADD
                contact_marker.pose.position.x = float(T_final_base[0, 3])
                contact_marker.pose.position.y = float(T_final_base[1, 3])
                contact_marker.pose.position.z = float(T_final_base[2, 3])
                contact_marker.pose.orientation.w = 1.0
                s = max(0.002, float(self.candidate_contact_scale) * 0.65)
                contact_marker.scale.x = s
                contact_marker.scale.y = s
                contact_marker.scale.z = s
                contact_marker.color = ColorRGBA(r=0.8, g=0.8, b=0.8, a=min(0.8, float(self.raw_inferred_marker_alpha) + 0.2))
                contact_marker.lifetime = self.marker_lifetime_msg()
                ma.markers.append(contact_marker)

        self.raw_inferred_markers_base_pub.publish(ma)
        if self.verbose_filter_log:
            self.get_logger().info(
                f'[raw_vis] published raw inferred grasps in {self.base_frame}: '
                f'{max(0, len(ma.markers)-1)} markers from {len(grasps)} score-filtered grasps'
            )

    def gripper_wire_points_from_matrix(self, T: np.ndarray) -> List[Point]:
        width = max(0.01, float(self.visual_gripper_width))
        finger = float(self.gripper_finger_length)
        palm = float(self.gripper_palm_depth)
        tail = float(self.gripper_tail_length)

        segs_local = [
            ([0.0, -width / 2.0, 0.0], [finger, -width / 2.0, 0.0]),
            ([0.0,  width / 2.0, 0.0], [finger,  width / 2.0, 0.0]),
            ([0.0, -width / 2.0, 0.0], [0.0,   width / 2.0, 0.0]),
            ([-palm, 0.0, 0.0], [0.0, 0.0, 0.0]),
            ([-palm - tail, 0.0, 0.0], [-palm, 0.0, 0.0]),
        ]

        pts: List[Point] = []
        for a, b in segs_local:
            av = T @ np.array([a[0], a[1], a[2], 1.0], dtype=np.float64)
            bv = T @ np.array([b[0], b[1], b[2], 1.0], dtype=np.float64)
            pts.append(Point(x=float(av[0]), y=float(av[1]), z=float(av[2])))
            pts.append(Point(x=float(bv[0]), y=float(bv[1]), z=float(bv[2])))
        return pts

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

        # RViz marker topics are intentionally not published from this node.
        # Surviving candidate grasps are sent as PoseArray to CALI_D405, which
        # transforms them to base_link and publishes the only RViz grasp markers.
        self.publish_best_contact_point(best.translation, header)

        t = best.translation
        vis_width = float(getattr(self, "visual_gripper_width", 0.10)) if bool(getattr(self, "use_fixed_visual_gripper_width", True)) else float(best.width)
        self.get_logger().info(
            f'grasps={len(grasps)} best_score={best.score:.4f} best_xyz=({t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f}) '
            f'pred_width={best.width:.4f} vis_width={vis_width:.4f} rank={best.rank:.4f} '
            f'ok_theta={math.degrees(best.okrobot_theta):.1f}deg ok_score={best.okrobot_score:.4f} '
            f'side_dot={best.arm_side_dot:.3f} front_dot={best.front_side_dot:.3f}'
        )

    def publish_empty_markers(self, header) -> None:
        # Only clear candidate visualization downstream by publishing an empty
        # PoseArray. CALI_D405 converts this into DELETEALL on its base_link
        # marker topics. No camera-frame RViz markers are published here.
        if header is None:
            return
        empty = PoseArray()
        empty.header = header
        self.grasps_pub.publish(empty)
        self.publish_raw_inferred_grasp_markers_base(header, [])

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



    # ============================================================
    # Master_2 control wrappers. These methods intentionally keep the
    # master_2 communication structure unchanged while reusing the v5
    # filtering/reranking logic above.
    # ============================================================
    def start_callback(self, msg: Bool) -> None:
        if msg.data:
            self.reset_runtime_inputs()
            self.active = True
            self.already_processed = False
            self._already_processed = False
            self.get_logger().info('[START] /anygrasp_l_start true. Cleared old inputs; waiting fresh SAM3 clouds.')
            self.maybe_run_after_input_update()
        else:
            self.active = False
            self.already_processed = False
            self._already_processed = False
            self.reset_runtime_inputs()
            self.publish_empty_markers(self.header)
            self.get_logger().info('[STOP] /anygrasp_l_start false. paused and input cache cleared.')

    def reset_runtime_inputs(self) -> None:
        self.scene_points = None
        self.scene_colors = None
        self.object_points = None
        self.background_points = None
        self.target_mask = None
        # camera_info is not cleared because it is camera calibration, not a perception result.
        self.body_axis = None
        self.body_center = None
        self.body_min = 0.0
        self.body_max = 0.0
        self.body_radius = 0.02
        self.object_width_est = 0.0
        self.header = None
        self.last_filter_stats = {}

    def maybe_run_after_input_update(self) -> None:
        if not self.active or self.already_processed:
            return
        if self.header is None or self.scene_points is None or self.scene_colors is None:
            return
        if self.require_target_data and (self.object_points is None or self.object_points.shape[0] == 0):
            return
        if self.target_filter_use_mask and (self.target_mask is None or self.camera_info is None):
            # Do not run AnyGrasp before the SAM3 target mask arrives.
            # Otherwise all candidates may be rejected or stale masks may be used.
            return
        if self.require_background_data and (self.background_points is None or self.background_points.shape[0] == 0):
            return
        self.run_inference(self.header)

    def publish_finish(self, value: bool = True) -> None:
        msg = Bool()
        msg.data = bool(value)
        self.finish_pub.publish(msg)
        self.get_logger().info(f'[PUB] {self.finish_topic} data={str(value).lower()}')

    def camera_info_callback(self, msg: CameraInfo) -> None:
        self.camera_info = msg
        self.maybe_run_after_input_update()

    def mask_callback(self, msg) -> None:
        try:
            from cv_bridge import CvBridge
            bridge = CvBridge()
            mask = bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
            self.target_mask = (mask > 0)
            self.maybe_run_after_input_update()
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
            self.maybe_run_after_input_update()
        except Exception as exc:
            self.get_logger().warn(f'object cloud parse failed: {repr(exc)}')

    def background_cloud_callback(self, msg: PointCloud2) -> None:
        try:
            pts, _ = self.pointcloud2_to_numpy(msg)
            self.background_points = pts if pts.shape[0] > 0 else None
            if self.verbose_filter_log:
                self.get_logger().info(f'background cloud updated: points={pts.shape[0]}')
            self.maybe_run_after_input_update()
        except Exception as exc:
            self.get_logger().warn(f'background cloud parse failed: {repr(exc)}')

    def scene_cloud_callback(self, msg: PointCloud2) -> None:
        try:
            points, colors = self.pointcloud2_to_numpy(msg)
            self.header = msg.header
            self.scene_points = points
            self.scene_colors = colors
            self.get_logger().info(
                f'cloud parsed: points_shape={points.shape}, points_dtype={points.dtype}, '
                f'colors_shape={colors.shape}, colors_dtype={colors.dtype}, active={self.active}'
            )
            self.maybe_run_after_input_update()
        except Exception as exc:
            self.get_logger().error(f'scene cloud parse failed: {repr(exc)}')

    def run_inference(self, header) -> None:
        if not self.active or self.already_processed:
            return
        if self.scene_points is None or self.scene_colors is None:
            return
        if self.scene_points.shape[0] < self.min_points:
            self.get_logger().warn(f'Input cloud too small: {self.scene_points.shape[0]} < {self.min_points}')
            return
        if self.require_target_data and (self.object_points is None or self.object_points.shape[0] == 0):
            self.get_logger().warn('Waiting object_pc before AnyGrasp inference.')
            return
        if self.target_filter_use_mask and (self.target_mask is None or self.camera_info is None):
            self.get_logger().warn('Waiting target_mask/camera_info before AnyGrasp inference.')
            return
        if self.require_background_data and (self.background_points is None or self.background_points.shape[0] == 0):
            self.get_logger().warn('Waiting background_pc before AnyGrasp inference.')
            return

        try:
            points, colors = self.voxel_downsample(self.scene_points, self.scene_colors, self.voxel_size)
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

            # Debug visualization: publish raw AnyGrasp candidates in base_frame
            # before any hard filters. This is intentionally independent from
            # /anygrasp_l/grasps, which still contains only surviving candidates.
            self.publish_raw_inferred_grasp_markers_base(header, grasps)

            if len(grasps) == 0:
                self.get_logger().warn('AnyGrasp produced no grasps above score threshold.')
                self.publish_empty_markers(header)
                self.already_processed = True
                self.active = False
                self.publish_finish(False)
                return

            ranked_all = self.rank_grasps(grasps)
            if len(ranked_all) == 0:
                stats = getattr(self, 'last_filter_stats', {})
                self.get_logger().warn(f'No grasps survived target/background hard filters. stats={stats}')
                self.publish_empty_markers(header)
                self.already_processed = True
                self.active = False
                self.publish_finish(False)
                return

            width_ok_count = sum(1 for g in ranked_all if g.width_gate_ok)
            if self.enable_local_width_gate and width_ok_count == 0:
                self.get_logger().warn('All surviving target grasps are rejected by local width gate. Object may be too wide locally for the gripper.')

            ranked = [g for g in ranked_all if g.rank >= self.rank_threshold]
            if len(ranked) == 0:
                self.get_logger().warn('All grasps removed after ranking threshold. Falling back to force_keep ranking.')
                ranked = self.rank_grasps(grasps, force_keep=True)
                if len(ranked) == 0:
                    self.publish_empty_markers(header)
                    self.already_processed = True
                    self.active = False
                    self.publish_finish(False)
                    return

            ranked.sort(key=lambda g: g.rank, reverse=True)
            ranked = ranked[: self.max_publish_grasps]
            self.log_top_candidates(ranked, 10)
            self.publish_outputs(header, ranked)
            self.already_processed = True
            self._already_processed = True
            self.active = False
            self.publish_finish(True)
        except Exception as exc:
            self.get_logger().error(f'AnyGrasp inference failed: {repr(exc)}')
            self.publish_empty_markers(header)
            self.already_processed = True
            self.active = False
            self.publish_finish(False)

def main(args=None) -> None:
    rclpy.init(args=args)
    node = AnyGraspMaster2Node()
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