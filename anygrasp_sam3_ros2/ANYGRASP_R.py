#!/usr/bin/env python3
# AnyGrasp node for master_2 (RIGHT ARM).
# - waits for /anygrasp_r_start true
# - consumes SAM3 output topics
# - runs AnyGrasp once per start signal
# - publishes the same intermediate topics used by CALI_D405:
#     /anygrasp_r/best_pose_raw
#     /anygrasp_r/best_contact_point
#     /anygrasp_r/best_width
#     /anygrasp_r/best_score
#     /anygrasp_r/grasps
#     visualization markers
# - publishes optional /anygrasp_r_finish

import os
import sys
import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

# NumPy compatibility patch for legacy AnyGrasp / SDK code
for _name, _value in {'float': float, 'int': int, 'complex': complex}.items():
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

from std_msgs.msg import Bool, Float32, ColorRGBA
from geometry_msgs.msg import Pose, PoseArray, PoseStamped, Point, PointStamped
from sensor_msgs.msg import PointCloud2, CameraInfo, Image
from sensor_msgs_py import point_cloud2
from visualization_msgs.msg import Marker, MarkerArray


@dataclass
class RankedGrasp:
    score: float
    width: float
    translation: np.ndarray
    rotation_matrix: np.ndarray
    rank: float = 0.0
    target_ok: bool = False
    target_dist: float = float('inf')
    bg_clearance: float = float('inf')


class AnyGraspMaster2Node(Node):
    def __init__(self) -> None:
        super().__init__('anygrasp_r_master2_node')

        # master control
        self.declare_parameter('start_topic', '/anygrasp_r_start')
        self.declare_parameter('finish_topic', '/anygrasp_r_finish')

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

        # inputs from SAM3
        self.declare_parameter('scene_cloud_topic', '/sam3_r/full_scene_pc')
        self.declare_parameter('target_cloud_topic', '/sam3_r/target_pc')
        self.declare_parameter('object_cloud_topic', '/sam3_r/object_pc')
        self.declare_parameter('background_cloud_topic', '/sam3_r/background_pc')
        self.declare_parameter('target_mask_topic', '/sam3_r/target_mask')
        self.declare_parameter('camera_info_topic', '/camera_r/camera_r/aligned_depth_to_color/camera_info')
        self.declare_parameter('use_scene_cloud', True)
        self.declare_parameter('min_points', 150)
        self.declare_parameter('voxel_size', 0.004)
        self.declare_parameter('crop_margin_x', 0.02)
        self.declare_parameter('crop_margin_y', 0.02)
        self.declare_parameter('crop_margin_z', 0.02)

        # filtering / ranking
        self.declare_parameter('score_threshold', 0.05)
        self.declare_parameter('max_publish_grasps', 30)
        self.declare_parameter('hard_filter_to_target', True)
        self.declare_parameter('require_target_data', True)
        self.declare_parameter('target_filter_radius', 0.030)
        self.declare_parameter('hard_reject_background', True)
        self.declare_parameter('min_bg_clearance', 0.012)
        self.declare_parameter('verbose_filter_log', True)

        # outputs
        self.declare_parameter('best_grasp_topic', '/anygrasp_r/best_grasp')
        self.declare_parameter('best_pose_raw_topic', '/anygrasp_r/best_pose_raw')
        self.declare_parameter('best_width_topic', '/anygrasp_r/best_width')
        self.declare_parameter('best_score_topic', '/anygrasp_r/best_score')
        self.declare_parameter('grasps_topic', '/anygrasp_r/grasps')
        self.declare_parameter('markers_topic', '/anygrasp_r/grasp_markers')
        self.declare_parameter('all_markers_topic', '/anygrasp_r/all_grasp_markers')
        self.declare_parameter('best_marker_topic', '/anygrasp_r/best_pose_marker')
        self.declare_parameter('best_contact_marker_topic', '/anygrasp_r/best_contact_marker')
        self.declare_parameter('best_contact_point_topic', '/anygrasp_r/best_contact_point')

        # marker style
        self.declare_parameter('marker_lifetime_sec', 0.0)
        self.declare_parameter('marker_alpha', 0.85)
        self.declare_parameter('marker_topk', 30)
        self.declare_parameter('best_contact_scale', 0.012)
        self.declare_parameter('candidate_contact_scale', 0.008)
        self.declare_parameter('best_gripper_line_width', 0.0030)
        self.declare_parameter('candidate_gripper_line_width', 0.0022)
        self.declare_parameter('visual_gripper_width', 0.10)
        self.declare_parameter('gripper_finger_length', 0.032)
        self.declare_parameter('gripper_palm_depth', 0.010)
        self.declare_parameter('gripper_tail_length', 0.010)

        # fetch params
        self.start_topic = self.get_parameter('start_topic').value
        self.finish_topic = self.get_parameter('finish_topic').value
        self.sdk_root = self.get_parameter('sdk_root').value
        self.checkpoint_path = self.get_parameter('checkpoint_path').value
        self.scene_cloud_topic = self.get_parameter('scene_cloud_topic').value
        self.target_cloud_topic = self.get_parameter('target_cloud_topic').value
        self.object_cloud_topic = self.get_parameter('object_cloud_topic').value
        self.background_cloud_topic = self.get_parameter('background_cloud_topic').value
        self.target_mask_topic = self.get_parameter('target_mask_topic').value
        self.camera_info_topic = self.get_parameter('camera_info_topic').value
        self.use_scene_cloud = bool(self.get_parameter('use_scene_cloud').value)
        self.min_points = int(self.get_parameter('min_points').value)
        self.voxel_size = float(self.get_parameter('voxel_size').value)
        self.crop_margin_x = float(self.get_parameter('crop_margin_x').value)
        self.crop_margin_y = float(self.get_parameter('crop_margin_y').value)
        self.crop_margin_z = float(self.get_parameter('crop_margin_z').value)
        self.score_threshold = float(self.get_parameter('score_threshold').value)
        self.max_publish_grasps = int(self.get_parameter('max_publish_grasps').value)
        self.hard_filter_to_target = bool(self.get_parameter('hard_filter_to_target').value)
        self.require_target_data = bool(self.get_parameter('require_target_data').value)
        self.target_filter_radius = float(self.get_parameter('target_filter_radius').value)
        self.hard_reject_background = bool(self.get_parameter('hard_reject_background').value)
        self.min_bg_clearance = float(self.get_parameter('min_bg_clearance').value)
        self.verbose_filter_log = bool(self.get_parameter('verbose_filter_log').value)
        self.marker_lifetime_sec = float(self.get_parameter('marker_lifetime_sec').value)
        self.marker_alpha = float(self.get_parameter('marker_alpha').value)
        self.marker_topk = int(self.get_parameter('marker_topk').value)
        self.best_contact_scale = float(self.get_parameter('best_contact_scale').value)
        self.candidate_contact_scale = float(self.get_parameter('candidate_contact_scale').value)
        self.best_gripper_line_width = float(self.get_parameter('best_gripper_line_width').value)
        self.candidate_gripper_line_width = float(self.get_parameter('candidate_gripper_line_width').value)
        self.visual_gripper_width = float(self.get_parameter('visual_gripper_width').value)
        self.gripper_finger_length = float(self.get_parameter('gripper_finger_length').value)
        self.gripper_palm_depth = float(self.get_parameter('gripper_palm_depth').value)
        self.gripper_tail_length = float(self.get_parameter('gripper_tail_length').value)

        self.active = False
        self.already_processed = False
        self.scene_points: Optional[np.ndarray] = None
        self.scene_colors: Optional[np.ndarray] = None
        self.object_points: Optional[np.ndarray] = None
        self.background_points: Optional[np.ndarray] = None
        self.target_mask: Optional[np.ndarray] = None
        self.camera_info: Optional[CameraInfo] = None
        self.header = None

        self._append_sdk_path(self.sdk_root)
        self.anygrasp = self._build_anygrasp()

        self.qos_cmd = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        qos_data = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=5)

        input_topic = self.scene_cloud_topic if self.use_scene_cloud else self.target_cloud_topic

        # subscriptions
        self.create_subscription(Bool, self.start_topic, self.start_callback, self.qos_cmd)
        self.scene_sub = self.create_subscription(PointCloud2, input_topic, self.scene_cloud_callback, qos_data)
        self.object_sub = self.create_subscription(PointCloud2, self.object_cloud_topic, self.object_cloud_callback, qos_data)
        self.background_sub = self.create_subscription(PointCloud2, self.background_cloud_topic, self.background_cloud_callback, qos_data)
        self.mask_sub = self.create_subscription(Image, self.target_mask_topic, self.mask_callback, qos_data)
        self.cam_sub = self.create_subscription(CameraInfo, self.camera_info_topic, self.camera_info_callback, qos_data)

        # publishers
        self.finish_pub = self.create_publisher(Bool, self.finish_topic, self.qos_cmd)
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

        self.get_logger().info('========================================')
        self.get_logger().info('ANYGRASP MASTER2 Node Ready (RIGHT ARM)')
        self.get_logger().info(f'start_topic={self.start_topic}')
        self.get_logger().info(f'finish_topic={self.finish_topic}')
        self.get_logger().info(f'input_topic={input_topic}')
        self.get_logger().info(f'object_cloud_topic={self.object_cloud_topic}')
        self.get_logger().info(f'background_cloud_topic={self.background_cloud_topic}')
        self.get_logger().info('========================================')

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

    # ============================================================
    # Master control
    # ============================================================
    def start_callback(self, msg: Bool) -> None:
        if msg.data:
            # IMPORTANT FOR REPEATED INIT2 CYCLES:
            # Do not reuse SAM3 / object clouds from the previous object.
            # Wait until fresh callbacks arrive after this start signal.
            self.reset_runtime_inputs()
            self.active = True
            self.already_processed = False
            self.get_logger().info('[START] /anygrasp_r_start true. Cleared old inputs; waiting fresh SAM3 clouds.')
        else:
            self.active = False
            self.already_processed = False
            self.reset_runtime_inputs()
            self.get_logger().info('[STOP] /anygrasp_r_start false. paused and input cache cleared.')

    def reset_runtime_inputs(self) -> None:
        self.scene_points = None
        self.scene_colors = None
        self.object_points = None
        self.background_points = None
        self.target_mask = None
        self.header = None

    def maybe_run_after_input_update(self) -> None:
        if self.active and not self.already_processed and self.header is not None:
            self.run_inference(self.header)

    def publish_finish(self, value: bool = True):
        msg = Bool()
        msg.data = bool(value)
        self.finish_pub.publish(msg)
        self.get_logger().info(f'[PUB] {self.finish_topic} data={str(value).lower()}')

    # ============================================================
    # Input callbacks
    # ============================================================
    def camera_info_callback(self, msg: CameraInfo) -> None:
        self.camera_info = msg

    def mask_callback(self, msg: Image) -> None:
        try:
            from cv_bridge import CvBridge
            bridge = CvBridge()
            mask = bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
            self.target_mask = (mask > 0)
        except Exception as exc:
            self.get_logger().warn(f'Failed to parse target mask: {repr(exc)}')

    def object_cloud_callback(self, msg: PointCloud2) -> None:
        try:
            pts, _ = self.pointcloud2_to_numpy(msg)
            if pts.shape[0] >= 5:
                self.object_points = pts
                if self.verbose_filter_log:
                    self.get_logger().info(f'object cloud updated: points={pts.shape[0]}')
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
            self.get_logger().info(f'scene cloud updated: points={points.shape[0]}, active={self.active}')
            self.maybe_run_after_input_update()
        except Exception as exc:
            self.get_logger().error(f'scene cloud parse failed: {repr(exc)}')

    # ============================================================
    # AnyGrasp inference
    # ============================================================
    def run_inference(self, header) -> None:
        if not self.active:
            return
        if self.already_processed:
            return
        if self.scene_points is None or self.scene_colors is None:
            return
        if self.scene_points.shape[0] < self.min_points:
            self.get_logger().warn(f'Input cloud too small: {self.scene_points.shape[0]} < {self.min_points}')
            return
        if self.require_target_data and (self.object_points is None or self.object_points.shape[0] == 0):
            self.get_logger().warn('Waiting object_pc before AnyGrasp inference.')
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
            if len(grasps) == 0:
                self.get_logger().warn('AnyGrasp produced no grasps above score threshold.')
                self.publish_finish(False)
                self.already_processed = True
                self.active = False
                return

            ranked = self.rank_grasps(grasps)
            if len(ranked) == 0:
                self.get_logger().warn('No grasps survived filters.')
                self.publish_finish(False)
                self.already_processed = True
                self.active = False
                return

            ranked.sort(key=lambda g: g.rank, reverse=True)
            ranked = ranked[: self.max_publish_grasps]
            self.publish_outputs(header, ranked)

            self.already_processed = True
            self.active = False
            self.publish_finish(True)

        except Exception as exc:
            self.get_logger().error(f'AnyGrasp inference failed: {repr(exc)}')
            self.already_processed = True
            self.active = False
            self.publish_finish(False)

    def rank_grasps(self, grasps: Sequence[RankedGrasp]) -> List[RankedGrasp]:
        ranked: List[RankedGrasp] = []
        stats = {'raw': len(grasps), 'target_reject': 0, 'background_reject': 0, 'kept': 0}

        for g in grasps:
            g.target_dist = self.distance_to_object(g.translation)
            g.target_ok = (g.target_dist <= self.target_filter_radius)
            if self.hard_filter_to_target and not g.target_ok:
                stats['target_reject'] += 1
                continue

            g.bg_clearance = self.background_clearance(g.translation)
            if self.hard_reject_background and g.bg_clearance < self.min_bg_clearance:
                stats['background_reject'] += 1
                continue

            target_bonus = max(0.0, 1.0 - g.target_dist / max(self.target_filter_radius, 1e-6))
            bg_bonus = min(g.bg_clearance, 0.05)
            g.rank = float(g.score) + 0.5 * target_bonus + 0.2 * bg_bonus
            ranked.append(g)

        stats['kept'] = len(ranked)
        if self.verbose_filter_log:
            self.get_logger().info(f'[filter] {stats}')
        return ranked

    def distance_to_object(self, xyz: np.ndarray) -> float:
        if self.object_points is None or self.object_points.shape[0] == 0:
            return 0.0 if not self.require_target_data else float('inf')
        d = np.linalg.norm(self.object_points - xyz[None, :], axis=1)
        return float(np.min(d))

    def background_clearance(self, xyz: np.ndarray) -> float:
        if self.background_points is None or self.background_points.shape[0] == 0:
            return float('inf')
        d = np.linalg.norm(self.background_points - xyz[None, :], axis=1)
        return float(np.min(d))

    # ============================================================
    # Conversion / publishing
    # ============================================================
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

        contact = PointStamped()
        contact.header = header
        contact.point.x = float(best.translation[0])
        contact.point.y = float(best.translation[1])
        contact.point.z = float(best.translation[2])
        self.best_contact_point_pub.publish(contact)

        self.publish_best_contact_marker(best, header)
        self.publish_best_marker(best, header)
        self.publish_markers(header, grasps)

        t = best.translation
        self.get_logger().info(
            f'[BEST_GRASP] n={len(grasps)} score={best.score:.4f} rank={best.rank:.4f} '
            f'xyz=({t[0]:.4f},{t[1]:.4f},{t[2]:.4f}) width={best.width:.4f}'
        )

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

    # ============================================================
    # PointCloud conversion
    # ============================================================
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

    # ============================================================
    # Markers
    # ============================================================
    def marker_lifetime_msg(self):
        return Duration(seconds=self.marker_lifetime_sec).to_msg()

    def make_delete_all_marker(self, header) -> Marker:
        marker = Marker()
        marker.header = header
        marker.action = Marker.DELETEALL
        return marker

    def publish_markers(self, header, grasps: Sequence[RankedGrasp]) -> None:
        all_markers = MarkerArray()
        all_markers.markers.append(self.make_delete_all_marker(header))
        topk = self.marker_topk if self.marker_topk > 0 else len(grasps)
        for idx, grasp in enumerate(grasps[:topk]):
            all_markers.markers.extend(self.make_candidate_markers(header, idx, grasp, best=(idx == 0)))
        self.markers_pub.publish(all_markers)
        self.all_markers_pub.publish(all_markers)

    def make_candidate_markers(self, header, idx: int, grasp: RankedGrasp, best: bool = False) -> List[Marker]:
        line_marker = Marker()
        line_marker.header = header
        line_marker.ns = 'best_grasp' if best else 'gripper_candidates'
        line_marker.id = idx * 2
        line_marker.type = Marker.LINE_LIST
        line_marker.action = Marker.ADD
        line_marker.scale.x = self.best_gripper_line_width if best else self.candidate_gripper_line_width
        line_marker.color = ColorRGBA(r=1.0 if best else 0.3, g=0.2 if best else 1.0, b=0.0 if best else 0.6, a=1.0 if best else self.marker_alpha)
        line_marker.lifetime = self.marker_lifetime_msg()
        line_marker.points = self.gripper_wire_points(grasp)

        contact_marker = Marker()
        contact_marker.header = header
        contact_marker.ns = 'best_contact' if best else 'contact_candidates'
        contact_marker.id = idx * 2 + 1
        contact_marker.type = Marker.SPHERE
        contact_marker.action = Marker.ADD
        contact_marker.pose.position.x = float(grasp.translation[0])
        contact_marker.pose.position.y = float(grasp.translation[1])
        contact_marker.pose.position.z = float(grasp.translation[2])
        contact_marker.pose.orientation.w = 1.0
        scale = self.best_contact_scale if best else self.candidate_contact_scale
        contact_marker.scale.x = scale
        contact_marker.scale.y = scale
        contact_marker.scale.z = scale
        contact_marker.color = ColorRGBA(r=0.0, g=1.0, b=1.0, a=1.0)
        contact_marker.lifetime = self.marker_lifetime_msg()
        return [line_marker, contact_marker]

    def publish_best_marker(self, best: RankedGrasp, header) -> None:
        self.best_marker_pub.publish(self.make_candidate_markers(header, 0, best, best=True)[0])

    def publish_best_contact_marker(self, best: RankedGrasp, header) -> None:
        self.best_contact_marker_pub.publish(self.make_candidate_markers(header, 0, best, best=True)[1])

    def gripper_wire_points(self, grasp: RankedGrasp) -> List[Point]:
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = grasp.rotation_matrix.astype(np.float64)
        T[:3, 3] = grasp.translation.astype(np.float64)
        width = max(0.01, self.visual_gripper_width)
        finger = self.gripper_finger_length
        palm = self.gripper_palm_depth
        tail = self.gripper_tail_length
        segs_local = [
            ([0, -width/2, 0], [finger, -width/2, 0]),
            ([0,  width/2, 0], [finger,  width/2, 0]),
            ([0, -width/2, 0], [0, width/2, 0]),
            ([-palm, 0, 0], [0, 0, 0]),
            ([-palm-tail, 0, 0], [-palm, 0, 0]),
        ]
        pts = []
        for a, b in segs_local:
            av = T @ np.array([a[0], a[1], a[2], 1.0], dtype=np.float64)
            bv = T @ np.array([b[0], b[1], b[2], 1.0], dtype=np.float64)
            pts.append(Point(x=float(av[0]), y=float(av[1]), z=float(av[2])))
            pts.append(Point(x=float(bv[0]), y=float(bv[1]), z=float(bv[2])))
        return pts


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