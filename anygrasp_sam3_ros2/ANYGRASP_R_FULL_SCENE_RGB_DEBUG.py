#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AnyGrasp RIGHT RealSense one-shot full-scene overlay saver.

Purpose:
  - No SAM3
  - No mask
  - No master start topic
  - No continuous inference
  - No PointCloud2 publish
  - Receive exactly one RGB + one depth + one CameraInfo
  - Build full-scene point cloud internally
  - Run AnyGrasp once
  - Overlay grasps on that RGB image
  - Save exactly one image file, then exit

Default RIGHT RealSense topics from the user's topic list:
  RGB compressed:
    /camera_r/camera_r/color/image_rect_raw/compressed
  Depth compressedDepth:
    /camera_r/camera_r/aligned_depth_to_color/image_raw/compressedDepth
  CameraInfo:
    /camera_r/camera_r/aligned_depth_to_color/camera_info

Run:
  source /opt/ros/humble/setup.bash
  source ~/colcon_ws/install/setup.bash
  conda activate anygrasp
  python3 ~/colcon_ws/src/anygrasp_sam3_ros2/anygrasp_sam3_ros2/ANYGRASP_R_ONESHOT_SAVE_OVERLAY.py
"""

from __future__ import annotations

import os
import sys
import time
import threading
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

# NumPy compatibility patch for legacy AnyGrasp / SDK code.
for _name, _value in {"float": float, "int": int, "complex": complex}.items():
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

import cv2
from cv_bridge import CvBridge

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import CameraInfo, Image, CompressedImage


@dataclass
class SimpleGrasp:
    score: float
    width: float
    translation: np.ndarray
    rotation_matrix: np.ndarray


class AnyGraspRightOneShotOverlay(Node):
    def __init__(self) -> None:
        super().__init__("anygrasp_r_oneshot_overlay")

        # ---------------- AnyGrasp SDK ----------------
        self.declare_parameter("sdk_root", "/home/jwg/anygrasp_sdk/grasp_detection")
        self.declare_parameter("checkpoint_path", "/home/jwg/anygrasp_sdk/ckpt/checkpoint_detection.tar")
        self.declare_parameter("max_gripper_width", 0.10)
        self.declare_parameter("gripper_height", 0.03)
        self.declare_parameter("top_down_grasp", False)
        self.declare_parameter("dense_grasp", False)
        self.declare_parameter("collision_detection", True)
        self.declare_parameter("debug", False)

        # ---------------- One-shot inputs ----------------
        self.declare_parameter("rgb_compressed_topic", "/camera_r/camera_r/color/image_rect_raw/compressed")
        self.declare_parameter("camera_info_topic", "/camera_r/camera_r/aligned_depth_to_color/camera_info")

        # Default is compressedDepth to reduce DDS traffic. Set false to use raw Image depth.
        self.declare_parameter("use_compressed_depth", True)
        self.declare_parameter("depth_compressed_topic", "/camera_r/camera_r/aligned_depth_to_color/image_raw/compressedDepth")
        self.declare_parameter("depth_raw_topic", "/camera_r/camera_r/aligned_depth_to_color/image_raw")

        # ---------------- Output file ----------------
        self.declare_parameter("output_path", "~/anygrasp_r_full_scene_overlay.jpg")
        self.declare_parameter("save_input_rgb", False)
        self.declare_parameter("input_rgb_path", "~/anygrasp_r_input_rgb.jpg")
        self.declare_parameter("jpeg_quality", 95)

        # ---------------- Point cloud / inference params ----------------
        self.declare_parameter("depth_min_m", 0.15)
        self.declare_parameter("depth_max_m", 1.20)
        # This is still full-scene because no object mask is used; stride only reduces point count.
        self.declare_parameter("pixel_stride", 4)
        self.declare_parameter("voxel_size", 0.005)
        self.declare_parameter("max_infer_points", 60000)
        self.declare_parameter("min_points", 300)
        self.declare_parameter("score_threshold", 0.05)
        self.declare_parameter("max_grasps", 80)
        self.declare_parameter("draw_topk", 20)
        self.declare_parameter("timeout_sec", 10.0)

        # Full-scene crop limits computed from valid points, then expanded by margin.
        self.declare_parameter("crop_margin_x", 0.02)
        self.declare_parameter("crop_margin_y", 0.02)
        self.declare_parameter("crop_margin_z", 0.02)

        # 2D visualization parameters.
        self.declare_parameter("use_fixed_visual_width", True)
        self.declare_parameter("visual_gripper_width", 0.10)
        self.declare_parameter("finger_length_m", 0.055)
        self.declare_parameter("palm_depth_m", 0.020)
        self.declare_parameter("tail_length_m", 0.015)
        self.declare_parameter("draw_axes", True)
        self.declare_parameter("axis_length_m", 0.050)

        # ---------------- Read params ----------------
        self.sdk_root = str(self.get_parameter("sdk_root").value)
        self.checkpoint_path = str(self.get_parameter("checkpoint_path").value)
        self.rgb_compressed_topic = str(self.get_parameter("rgb_compressed_topic").value)
        self.camera_info_topic = str(self.get_parameter("camera_info_topic").value)
        self.use_compressed_depth = bool(self.get_parameter("use_compressed_depth").value)
        self.depth_compressed_topic = str(self.get_parameter("depth_compressed_topic").value)
        self.depth_raw_topic = str(self.get_parameter("depth_raw_topic").value)
        self.output_path = os.path.expanduser(str(self.get_parameter("output_path").value))
        self.save_input_rgb = bool(self.get_parameter("save_input_rgb").value)
        self.input_rgb_path = os.path.expanduser(str(self.get_parameter("input_rgb_path").value))
        self.jpeg_quality = int(np.clip(int(self.get_parameter("jpeg_quality").value), 1, 100))

        self.depth_min_m = float(self.get_parameter("depth_min_m").value)
        self.depth_max_m = float(self.get_parameter("depth_max_m").value)
        self.pixel_stride = max(1, int(self.get_parameter("pixel_stride").value))
        self.voxel_size = max(0.0, float(self.get_parameter("voxel_size").value))
        self.max_infer_points = max(1000, int(self.get_parameter("max_infer_points").value))
        self.min_points = max(1, int(self.get_parameter("min_points").value))
        self.score_threshold = float(self.get_parameter("score_threshold").value)
        self.max_grasps = max(1, int(self.get_parameter("max_grasps").value))
        self.draw_topk = max(1, int(self.get_parameter("draw_topk").value))
        self.timeout_sec = max(1.0, float(self.get_parameter("timeout_sec").value))
        self.crop_margin_x = float(self.get_parameter("crop_margin_x").value)
        self.crop_margin_y = float(self.get_parameter("crop_margin_y").value)
        self.crop_margin_z = float(self.get_parameter("crop_margin_z").value)
        self.use_fixed_visual_width = bool(self.get_parameter("use_fixed_visual_width").value)
        self.visual_gripper_width = float(self.get_parameter("visual_gripper_width").value)
        self.finger_length_m = float(self.get_parameter("finger_length_m").value)
        self.palm_depth_m = float(self.get_parameter("palm_depth_m").value)
        self.tail_length_m = float(self.get_parameter("tail_length_m").value)
        self.draw_axes = bool(self.get_parameter("draw_axes").value)
        self.axis_length_m = float(self.get_parameter("axis_length_m").value)

        self.bridge = CvBridge()
        self.rng = np.random.default_rng(0)
        self.lock = threading.Lock()
        self.rgb_msg: Optional[CompressedImage] = None
        self.depth_raw_msg: Optional[Image] = None
        self.depth_compressed_msg: Optional[CompressedImage] = None
        self.camera_info: Optional[CameraInfo] = None
        self.processing_started = False
        self.start_time = time.monotonic()

        self._append_sdk_path(self.sdk_root)
        self.anygrasp = self._build_anygrasp()

        # Keep only latest. Callback stores first/latest, then subscriptions are destroyed before inference.
        self.sub_rgb = self.create_subscription(
            CompressedImage, self.rgb_compressed_topic, self.rgb_callback, qos_profile_sensor_data
        )
        if self.use_compressed_depth:
            self.sub_depth = self.create_subscription(
                CompressedImage, self.depth_compressed_topic, self.depth_compressed_callback, qos_profile_sensor_data
            )
        else:
            self.sub_depth = self.create_subscription(
                Image, self.depth_raw_topic, self.depth_raw_callback, qos_profile_sensor_data
            )
        self.sub_info = self.create_subscription(
            CameraInfo, self.camera_info_topic, self.camera_info_callback, qos_profile_sensor_data
        )
        self.timer = self.create_timer(0.5, self.watchdog_timer)

        self.get_logger().info("========================================")
        self.get_logger().info("AnyGrasp RIGHT ONE-SHOT overlay saver started")
        self.get_logger().info("NO SAM3 / NO mask / NO continuous inference / NO ROS output publish")
        self.get_logger().info(f"rgb_compressed_topic : {self.rgb_compressed_topic}")
        self.get_logger().info(f"depth input mode      : {'compressedDepth' if self.use_compressed_depth else 'raw Image'}")
        self.get_logger().info(f"depth_compressed_topic: {self.depth_compressed_topic}")
        self.get_logger().info(f"depth_raw_topic       : {self.depth_raw_topic}")
        self.get_logger().info(f"camera_info_topic     : {self.camera_info_topic}")
        self.get_logger().info(f"output_path           : {self.output_path}")
        self.get_logger().info(f"depth_min/max         : {self.depth_min_m:.3f} / {self.depth_max_m:.3f} m")
        self.get_logger().info(f"pixel_stride          : {self.pixel_stride}")
        self.get_logger().info(f"voxel_size            : {self.voxel_size:.4f}")
        self.get_logger().info(f"max_infer_points      : {self.max_infer_points}")
        self.get_logger().info("Waiting for exactly one RGB + depth + CameraInfo...")
        self.get_logger().info("========================================")

    # ============================================================
    # AnyGrasp setup
    # ============================================================
    def _append_sdk_path(self, sdk_root: str) -> None:
        if not os.path.isdir(sdk_root):
            raise RuntimeError(f"sdk_root does not exist: {sdk_root}")
        if sdk_root not in sys.path:
            sys.path.insert(0, sdk_root)

    def _build_anygrasp(self):
        from gsnet import AnyGrasp  # type: ignore
        import argparse

        cfgs = argparse.Namespace(
            checkpoint_path=self.checkpoint_path,
            max_gripper_width=max(0.0, min(0.1, float(self.get_parameter("max_gripper_width").value))),
            gripper_height=float(self.get_parameter("gripper_height").value),
            top_down_grasp=bool(self.get_parameter("top_down_grasp").value),
            debug=bool(self.get_parameter("debug").value),
        )
        ag = AnyGrasp(cfgs)
        ag.load_net()
        return ag

    # ============================================================
    # ROS callbacks: store one frame set, then run once
    # ============================================================
    def rgb_callback(self, msg: CompressedImage) -> None:
        with self.lock:
            if self.processing_started:
                return
            self.rgb_msg = msg
        self.try_start_processing()

    def depth_raw_callback(self, msg: Image) -> None:
        with self.lock:
            if self.processing_started:
                return
            self.depth_raw_msg = msg
        self.try_start_processing()

    def depth_compressed_callback(self, msg: CompressedImage) -> None:
        with self.lock:
            if self.processing_started:
                return
            self.depth_compressed_msg = msg
        self.try_start_processing()

    def camera_info_callback(self, msg: CameraInfo) -> None:
        with self.lock:
            if self.processing_started:
                return
            self.camera_info = msg
        self.try_start_processing()

    def try_start_processing(self) -> None:
        with self.lock:
            if self.processing_started:
                return
            rgb_msg = self.rgb_msg
            depth_msg = self.depth_compressed_msg if self.use_compressed_depth else self.depth_raw_msg
            camera_info = self.camera_info
            if rgb_msg is None or depth_msg is None or camera_info is None:
                return
            self.processing_started = True

        # Stop receiving images before heavy inference to avoid DDS/callback load.
        self.destroy_subscription(self.sub_rgb)
        self.destroy_subscription(self.sub_depth)
        self.destroy_subscription(self.sub_info)
        self.destroy_timer(self.timer)
        self.get_logger().info("Got one RGB + depth + CameraInfo. Subscriptions destroyed. Running AnyGrasp once...")
        threading.Thread(target=self.run_once_and_exit, args=(rgb_msg, depth_msg, camera_info), daemon=True).start()

    def watchdog_timer(self) -> None:
        if self.processing_started:
            return
        elapsed = time.monotonic() - self.start_time
        with self.lock:
            missing = []
            if self.rgb_msg is None:
                missing.append("RGB")
            if (self.depth_compressed_msg if self.use_compressed_depth else self.depth_raw_msg) is None:
                missing.append("depth")
            if self.camera_info is None:
                missing.append("CameraInfo")
        self.get_logger().info(f"Waiting... missing={missing if missing else '-'} elapsed={elapsed:.1f}s")
        if elapsed > self.timeout_sec:
            self.get_logger().error(f"Timeout {self.timeout_sec:.1f}s. Check topic names/types. missing={missing}")
            rclpy.shutdown()

    # ============================================================
    # Main one-shot inference
    # ============================================================
    def run_once_and_exit(self, rgb_msg: CompressedImage, depth_msg, camera_info: CameraInfo) -> None:
        t0 = time.time()
        ok = False
        try:
            rgb = self.compressed_rgb_to_rgb(rgb_msg)
            depth_m = self.compressed_depth_to_meters(depth_msg) if self.use_compressed_depth else self.depth_raw_to_meters(depth_msg)

            if rgb.shape[:2] != depth_m.shape[:2]:
                self.get_logger().warn(
                    f"RGB/depth size mismatch: rgb={rgb.shape[:2]}, depth={depth_m.shape[:2]}. "
                    "Resizing depth to RGB size. Aligned depth is recommended."
                )
                depth_m = cv2.resize(depth_m, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)

            if self.save_input_rgb:
                self.save_bgr(self.rgb_to_bgr(rgb), self.input_rgb_path)

            points, colors, intr = self.build_full_scene_cloud(rgb, depth_m, camera_info)
            raw_point_count = int(points.shape[0])
            vis = self.rgb_to_bgr(rgb)

            if raw_point_count < self.min_points:
                self.draw_text_box(vis, f"Too few valid points: {raw_point_count}", (12, 32), (0, 0, 0), (0, 0, 255))
                self.save_bgr(vis, self.output_path)
                self.get_logger().error(f"Too few valid points: {raw_point_count}. Saved image: {self.output_path}")
                ok = True
                return

            infer_points, infer_colors = self.voxel_downsample(points, colors, self.voxel_size)
            if infer_points.shape[0] > self.max_infer_points:
                idx = self.rng.choice(infer_points.shape[0], size=self.max_infer_points, replace=False)
                infer_points = infer_points[idx]
                infer_colors = infer_colors[idx]

            lims = self.compute_lims(infer_points)
            gg, _ = self.anygrasp.get_grasp(
                infer_points,
                infer_colors,
                lims=lims,
                apply_object_mask=False,
                dense_grasp=bool(self.get_parameter("dense_grasp").value),
                collision_detection=bool(self.get_parameter("collision_detection").value),
            )

            grasps = self.convert_grasp_group(gg)
            grasps = [g for g in grasps if np.isfinite(g.score) and g.score >= self.score_threshold]
            grasps.sort(key=lambda g: g.score, reverse=True)
            grasps = grasps[: self.max_grasps]

            if not grasps:
                self.draw_text_box(
                    vis,
                    f"AnyGrasp: no grasp above threshold | full_pc={raw_point_count} infer_pc={infer_points.shape[0]}",
                    (12, 32),
                    (0, 0, 0),
                    (0, 0, 255),
                )
                self.get_logger().warn(f"No grasps above threshold. full_pc={raw_point_count}, infer_pc={infer_points.shape[0]}")
            else:
                self.draw_grasps_on_image(vis, grasps, intr, raw_point_count, infer_points.shape[0])
                best = grasps[0]
                t = best.translation
                self.get_logger().info(
                    f"AnyGrasp one-shot OK: grasps={len(grasps)} best_score={best.score:.4f} "
                    f"xyz_cam=({t[0]:.4f},{t[1]:.4f},{t[2]:.4f}) width={best.width:.4f} "
                    f"full_pc={raw_point_count} infer_pc={infer_points.shape[0]} time={time.time() - t0:.3f}s"
                )

            self.save_bgr(vis, self.output_path)
            self.get_logger().info(f"Saved overlay image: {self.output_path}")
            ok = True

        except Exception as exc:
            self.get_logger().error(f"AnyGrasp one-shot failed: {repr(exc)}")
            try:
                rgb = self.compressed_rgb_to_rgb(rgb_msg)
                vis = self.rgb_to_bgr(rgb)
                self.draw_text_box(vis, f"AnyGrasp error: {repr(exc)[:90]}", (12, 32), (0, 0, 0), (0, 0, 255))
                self.save_bgr(vis, self.output_path)
                self.get_logger().info(f"Saved error overlay image: {self.output_path}")
                ok = True
            except Exception:
                pass
        finally:
            if not ok:
                self.get_logger().error("No output image was saved.")
            # Give logger a moment, then exit this ROS process.
            time.sleep(0.2)
            if rclpy.ok():
                rclpy.shutdown()

    # ============================================================
    # Decode / point cloud
    # ============================================================
    @staticmethod
    def compressed_rgb_to_rgb(msg: CompressedImage) -> np.ndarray:
        arr = np.frombuffer(msg.data, np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"cv2.imdecode failed for compressed RGB, format='{msg.format}'")
        return np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

    @staticmethod
    def rgb_to_bgr(rgb: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    def depth_raw_to_meters(self, msg: Image) -> np.ndarray:
        depth = np.asarray(self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough"))
        enc = msg.encoding.lower()
        if enc in ("16uc1", "mono16") or depth.dtype == np.uint16:
            return depth.astype(np.float32) * 0.001
        return depth.astype(np.float32)

    @staticmethod
    def compressed_depth_to_meters(msg: CompressedImage) -> np.ndarray:
        data = np.frombuffer(msg.data, np.uint8)
        depth = None
        # compressed_depth_image_transport usually prepends a small config header before PNG.
        # Try raw first, then common header offsets.
        for offset in (0, 12, 16):
            if data.size <= offset:
                continue
            depth_try = cv2.imdecode(data[offset:], cv2.IMREAD_UNCHANGED)
            if depth_try is not None:
                depth = depth_try
                break
        if depth is None:
            raise RuntimeError(f"cv2.imdecode failed for compressedDepth, format='{msg.format}', bytes={len(msg.data)}")
        if depth.dtype == np.uint16:
            return depth.astype(np.float32) * 0.001
        return depth.astype(np.float32)

    @staticmethod
    def scaled_intrinsics(camera_info: CameraInfo, width: int, height: int) -> Tuple[float, float, float, float]:
        fx = float(camera_info.k[0])
        fy = float(camera_info.k[4])
        cx = float(camera_info.k[2])
        cy = float(camera_info.k[5])
        ci_w = int(camera_info.width)
        ci_h = int(camera_info.height)
        if ci_w > 0 and ci_h > 0 and (ci_w != width or ci_h != height):
            sx = float(width) / float(ci_w)
            sy = float(height) / float(ci_h)
            fx *= sx
            cx *= sx
            fy *= sy
            cy *= sy
        return fx, fy, cx, cy

    def build_full_scene_cloud(
        self,
        rgb: np.ndarray,
        depth_m: np.ndarray,
        camera_info: CameraInfo,
    ) -> Tuple[np.ndarray, np.ndarray, Tuple[float, float, float, float]]:
        h, w = depth_m.shape[:2]
        fx, fy, cx, cy = self.scaled_intrinsics(camera_info, w, h)
        if fx <= 0.0 or fy <= 0.0:
            raise RuntimeError(f"Invalid intrinsics: fx={fx}, fy={fy}")

        ys = np.arange(0, h, self.pixel_stride, dtype=np.int32)
        xs = np.arange(0, w, self.pixel_stride, dtype=np.int32)
        uu, vv = np.meshgrid(xs, ys)
        z = depth_m[vv, uu]

        valid = np.isfinite(z) & (z >= self.depth_min_m) & (z <= self.depth_max_m)
        if not np.any(valid):
            return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.float32), (fx, fy, cx, cy)

        u = uu[valid].astype(np.float32)
        v = vv[valid].astype(np.float32)
        z = z[valid].astype(np.float32)
        x = (u - cx) / fx * z
        y = (v - cy) / fy * z

        points = np.stack([x, y, z], axis=1).astype(np.float32)
        colors = rgb[v.astype(np.int32), u.astype(np.int32)].astype(np.float32) / 255.0
        return points, colors, (fx, fy, cx, cy)

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
            float(mins[0] - self.crop_margin_x),
            float(maxs[0] + self.crop_margin_x),
            float(mins[1] - self.crop_margin_y),
            float(maxs[1] + self.crop_margin_y),
            float(mins[2] - self.crop_margin_z),
            float(maxs[2] + self.crop_margin_z),
        ]

    @staticmethod
    def convert_grasp_group(gg) -> List[SimpleGrasp]:
        out: List[SimpleGrasp] = []
        try:
            n = len(gg)
        except Exception:
            n = 0
        for i in range(n):
            g = gg[i]
            score = float(getattr(g, "score", 0.0))
            width = float(getattr(g, "width", 0.0))
            translation = np.asarray(getattr(g, "translation", np.zeros(3)), dtype=np.float32).reshape(3)
            rotation = np.asarray(getattr(g, "rotation_matrix", np.eye(3)), dtype=np.float32).reshape(3, 3)
            out.append(SimpleGrasp(score=score, width=width, translation=translation, rotation_matrix=rotation))
        return out

    # ============================================================
    # Visualization
    # ============================================================
    def draw_grasps_on_image(
        self,
        bgr: np.ndarray,
        grasps: List[SimpleGrasp],
        intr: Tuple[float, float, float, float],
        full_pc_count: int,
        infer_pc_count: int,
    ) -> None:
        h, _w = bgr.shape[:2]
        topk = min(self.draw_topk, len(grasps))
        for i in range(topk - 1, -1, -1):
            grasp = grasps[i]
            best = i == 0
            color = (0, 255, 0) if best else (0, 220, 255)
            thickness = 3 if best else 1
            self.draw_single_grasp(bgr, grasp, intr, color=color, thickness=thickness, best=best)

        best = grasps[0]
        text1 = f"AnyGrasp one-shot full scene | grasps={len(grasps)} | best score={best.score:.3f} width={best.width:.3f}m"
        text2 = f"best xyz cam=({best.translation[0]:.3f}, {best.translation[1]:.3f}, {best.translation[2]:.3f})m | pc={full_pc_count}->{infer_pc_count}"
        self.draw_text_box(bgr, text1, (12, 30), (0, 0, 0), (255, 255, 255))
        self.draw_text_box(bgr, text2, (12, 60), (0, 0, 0), (255, 255, 255))
        cv2.putText(bgr, "green=best / yellow=other grasps", (12, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    def draw_single_grasp(
        self,
        bgr: np.ndarray,
        grasp: SimpleGrasp,
        intr: Tuple[float, float, float, float],
        color: Tuple[int, int, int],
        thickness: int,
        best: bool,
    ) -> None:
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = grasp.rotation_matrix.astype(np.float64)
        T[:3, 3] = grasp.translation.astype(np.float64)

        width = self.visual_gripper_width if self.use_fixed_visual_width else max(0.01, float(grasp.width))
        for a_local, b_local in self.gripper_segments_local(width):
            a = (T @ np.array([a_local[0], a_local[1], a_local[2], 1.0], dtype=np.float64))[:3]
            b = (T @ np.array([b_local[0], b_local[1], b_local[2], 1.0], dtype=np.float64))[:3]
            pa = self.project_point(a, intr)
            pb = self.project_point(b, intr)
            if pa is not None and pb is not None:
                cv2.line(bgr, pa, pb, color, thickness, cv2.LINE_AA)

        c = self.project_point(grasp.translation, intr)
        if c is not None:
            cv2.circle(bgr, c, 4 if best else 2, color, -1)
            if best:
                cv2.putText(bgr, "BEST", (c[0] + 6, c[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        if self.draw_axes and best:
            origin = grasp.translation.astype(np.float64)
            Rm = grasp.rotation_matrix.astype(np.float64)
            axis_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
            p0 = self.project_point(origin, intr)
            for k in range(3):
                p1 = self.project_point(origin + Rm[:, k] * self.axis_length_m, intr)
                if p0 is not None and p1 is not None:
                    cv2.arrowedLine(bgr, p0, p1, axis_colors[k], 2, cv2.LINE_AA, tipLength=0.25)

    def gripper_segments_local(self, width: float) -> List[Tuple[np.ndarray, np.ndarray]]:
        finger = self.finger_length_m
        palm = self.palm_depth_m
        tail = self.tail_length_m
        return [
            (np.array([0.0, -width / 2.0, 0.0]), np.array([finger, -width / 2.0, 0.0])),
            (np.array([0.0,  width / 2.0, 0.0]), np.array([finger,  width / 2.0, 0.0])),
            (np.array([0.0, -width / 2.0, 0.0]), np.array([0.0,   width / 2.0, 0.0])),
            (np.array([-palm, 0.0, 0.0]),        np.array([0.0, 0.0, 0.0])),
            (np.array([-palm - tail, 0.0, 0.0]), np.array([-palm, 0.0, 0.0])),
        ]

    @staticmethod
    def project_point(point_xyz: np.ndarray, intr: Tuple[float, float, float, float]) -> Optional[Tuple[int, int]]:
        x, y, z = float(point_xyz[0]), float(point_xyz[1]), float(point_xyz[2])
        if not np.isfinite(z) or z <= 1e-6:
            return None
        fx, fy, cx, cy = intr
        u = int(round(fx * x / z + cx))
        v = int(round(fy * y / z + cy))
        if u < -10000 or v < -10000 or u > 10000 or v > 10000:
            return None
        return u, v

    @staticmethod
    def draw_text_box(img: np.ndarray, text: str, org: Tuple[int, int], bg_color, fg_color) -> None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.62
        thickness = 2
        (tw, th), base = cv2.getTextSize(text, font, scale, thickness)
        x, y = org
        cv2.rectangle(img, (x - 4, y - th - 7), (x + tw + 4, y + base + 5), bg_color, -1)
        cv2.putText(img, text, org, font, scale, fg_color, thickness, cv2.LINE_AA)

    def save_bgr(self, bgr: np.ndarray, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        ok = cv2.imwrite(path, bgr, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
        if not ok:
            raise RuntimeError(f"cv2.imwrite failed: {path}")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = AnyGraspRightOneShotOverlay()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()