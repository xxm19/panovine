#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Policy inference.

Inputs (live sensors):
- Camera frames from camServer.
- Encoders from RS485 serial stream. Converted to joint angles (rad) via lookup before feeding the policy.
- Base encoder from ROS topic /motor_encoder.
  Converted to length (m) via lookup before feeding the policy.

Outputs (action replay style):
- Base encoder target -> /motor_encoder_replay (std_msgs/Float32)
  Policy predicts base length (m); converted back to encoder ticks before publishing.
- Joint angles -> bending encoders target -> /encoders_replay_all (std_msgs/Int32MultiArray)
  data = [id1, enc1_1, enc2_1, id2, enc1_2, enc2_2, ...]
"""

import os
import re
import time
import threading
import argparse
import json
import pathlib
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
if not hasattr(torch, "xpu"):
    class _DummyXPU:
        @staticmethod
        def is_available():
            return False

        @staticmethod
        def empty_cache():
            return None

        @staticmethod
        def synchronize(device=None):
            return None

        @staticmethod
        def device_count():
            return 0

        @staticmethod
        def current_device():
            return 0

        @staticmethod
        def set_device(device):
            return None

        @staticmethod
        def get_device_name(index=0):
            return "dummy-xpu"

        @staticmethod
        def manual_seed(seed):
            return None

        @staticmethod
        def manual_seed_all(seed):
            return None

    torch.xpu = _DummyXPU()

import hydra
import dill
from multiprocessing.managers import SharedMemoryManager

try:
    from scipy.spatial.transform import Rotation as R
except Exception:
    R = None

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs

from base_length_mapping import encoder_to_length as base_encoder_to_length, length_to_encoder as base_length_to_encoder

try:
    import rospy
    from std_msgs.msg import Float32, Float32MultiArray, Int32MultiArray
except ImportError:
    rospy = None
    Float32 = None
    Float32MultiArray = None
    Int32MultiArray = None

try:
    from process_data_parallel import get_image_transform
except ImportError:
    get_image_transform = None

try:
    from encoder_angle_mapping import angles_rad_to_encoder_ticks, enc_list_to_joint_angles
except ImportError:
    angles_rad_to_encoder_ticks = None
    enc_list_to_joint_angles = None

try:
    from cameras.camera import USBCamera, camServer
except ImportError:
    USBCamera = None
    camServer = None

try:
    from cameras import visualization as viz
except ImportError:
    viz = None

try:
    import serial
except ImportError:
    serial = None

register_codecs()

FRAME_RE = re.compile(r"<([^>]*)>")
ID_RE = re.compile(r"^\s*ID\s*(\d+)\s*(?:DATA\s*:|:)\s*(.*)\s*$", re.IGNORECASE)
DRY_RUN = False


def _split_obs_for_viz(obs: Dict[str, np.ndarray]) -> Tuple[List[str], List[str]]:
    rgb_keys = []
    lowdim_keys = []
    for key in sorted(obs.keys()):
        value = obs[key]
        if value.ndim == 4 and value.shape[1] in (1, 3):
            rgb_keys.append(key)
        else:
            lowdim_keys.append(key)
    return rgb_keys, lowdim_keys


def _plot_rgb_slices_for_viz(
    obs: Dict[str, np.ndarray],
    rgb_keys: List[str],
    max_cameras: int,
    chunk_idx: int,
    out_dir: pathlib.Path,
    viz_gamma: float = 1.0,
    viz_contrast: float = 1.0,
    viz_autowb: bool = False,
) -> None:
    if not rgb_keys:
        return

    import matplotlib.pyplot as plt

    keys = rgb_keys[:max_cameras]
    horizon = int(obs[keys[0]].shape[0])
    fig, axes = plt.subplots(
        nrows=len(keys),
        ncols=horizon,
        figsize=(2.8 * horizon, 2.4 * len(keys)),
        squeeze=False,
    )

    def _enhance(img_in: np.ndarray) -> np.ndarray:
        img = np.clip(img_in, 0.0, 1.0)
        if viz_gamma > 0 and abs(viz_gamma - 1.0) > 1e-6:
            img = np.power(img, 1.0 / viz_gamma)
        if abs(viz_contrast - 1.0) > 1e-6:
            img = 0.5 + (img - 0.5) * viz_contrast
        if viz_autowb:
            ch_mean = np.mean(img, axis=(0, 1), keepdims=True)
            gray = float(np.mean(ch_mean))
            scale = gray / np.maximum(ch_mean, 1e-6)
            img = img * scale
        return np.clip(img, 0.0, 1.0)

    for row, key in enumerate(keys):
        seq = obs[key]  # (T, C, H, W)
        for col in range(horizon):
            img = np.transpose(seq[col], (1, 2, 0))
            img = _enhance(img)
            ax = axes[row, col]
            ax.imshow(img)
            if row == 0:
                ax.set_title(f"t={col}")
            if col == 0:
                ax.set_ylabel(key, rotation=0, ha="right", va="center")
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle(f"Chunk {chunk_idx}: RGB observation slices")
    fig.tight_layout()
    output = out_dir / f"chunk_{chunk_idx:06d}_images.png"
    fig.savefig(output, dpi=150)
    plt.close(fig)


def _plot_proprio_and_action_for_viz(
    obs: Dict[str, np.ndarray],
    lowdim_keys: List[str],
    action_seq: np.ndarray,
    chunk_idx: int,
    out_dir: pathlib.Path,
    action_label: str = "action",
    filename_suffix: str = "",
) -> None:
    import matplotlib.pyplot as plt

    lowdim_flat = {}
    for key in lowdim_keys:
        arr = obs[key]
        lowdim_flat[key] = arr.reshape(arr.shape[0], -1)

    action_flat = action_seq.reshape(action_seq.shape[0], -1)

    n_plots = int(sum(v.shape[1] for v in lowdim_flat.values()) + action_flat.shape[1])
    fig, axes = plt.subplots(
        nrows=n_plots,
        ncols=1,
        figsize=(11, max(4, 1.8 * n_plots)),
        squeeze=False,
    )
    axes = axes[:, 0]

    plot_i = 0
    for key in lowdim_keys:
        flat = lowdim_flat[key]
        x = np.arange(flat.shape[0])
        for dim in range(flat.shape[1]):
            ax = axes[plot_i]
            ax.plot(x, flat[:, dim], marker="o", linestyle="None", markersize=3)
            ax.set_title(f"{key}[{dim}]")
            ax.set_xlabel("obs horizon step")
            ax.set_ylabel("value")
            ax.grid(alpha=0.3)
            plot_i += 1

    x_act = np.arange(action_flat.shape[0])
    for dim in range(action_flat.shape[1]):
        ax = axes[plot_i]
        ax.plot(x_act, action_flat[:, dim], marker="o", linestyle="None", markersize=3)
        ax.set_title(f"action[{dim}]")
        ax.set_xlabel("action horizon step")
        ax.set_ylabel("value")
        ax.grid(alpha=0.3)
        plot_i += 1

    fig.suptitle(f"Chunk {chunk_idx}: proprio and {action_label}")
    fig.tight_layout()
    output = out_dir / f"chunk_{chunk_idx:06d}_signals{filename_suffix}.png"
    fig.savefig(output, dpi=150)
    plt.close(fig)


def _save_debug_policy_viz(
    obs_stack: Dict[str, np.ndarray],
    action_seq: np.ndarray,
    chunk_idx: int,
    out_dir: pathlib.Path,
    max_cameras: int,
    viz_gamma: float = 1.0,
    viz_contrast: float = 1.0,
    viz_autowb: bool = False,
    action_label: str = "action",
    filename_suffix: str = "",
) -> None:
    try:
        import matplotlib.pyplot as _  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for debug visualization. Install it with: pip install matplotlib"
        ) from exc

    rgb_keys, lowdim_keys = _split_obs_for_viz(obs_stack)
    _plot_rgb_slices_for_viz(
        obs_stack,
        rgb_keys,
        max_cameras,
        chunk_idx,
        out_dir,
        viz_gamma=viz_gamma,
        viz_contrast=viz_contrast,
        viz_autowb=viz_autowb,
    )
    _plot_proprio_and_action_for_viz(
        obs_stack,
        lowdim_keys,
        action_seq,
        chunk_idx,
        out_dir,
        action_label=action_label,
        filename_suffix=filename_suffix,
    )


def _predict_action_seq(policy, obs_tensor: Dict[str, torch.Tensor], num_samples: int) -> np.ndarray:
    with torch.no_grad():
        if num_samples <= 1:
            return policy.predict_action(obs_tensor)["action_pred"][0].detach().cpu().numpy()
        preds = []
        for _ in range(num_samples):
            preds.append(policy.predict_action(obs_tensor)["action_pred"][0].detach().cpu().numpy())
        return np.mean(np.stack(preds, axis=0), axis=0)


def _debug_plot_action_chunk(base_seq: np.ndarray, angle_seq: np.ndarray) -> None:
    import matplotlib.pyplot as plt

    fig = plt.figure("Policy Debug: Action Chunk")
    ax = fig.add_subplot(111, projection="3d")
    ax.cla()

    n_steps = angle_seq.shape[0]
    colors = plt.cm.viridis(np.linspace(0, 1, max(n_steps, 1)))
    for i in range(n_steps):
        length = float(base_seq[i])
        seg_lengths = viz.compute_segment_effective_lengths(length)
        angles = np.zeros(len(seg_lengths), dtype=float)
        fill_n = min(len(angles), angle_seq.shape[1])
        angles[:fill_n] = angle_seq[i, :fill_n]
        angles = viz.mask_inactive_joints(length, angles)
        pts = viz.forward_kinematics_3d(seg_lengths, angles)
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=colors[i], alpha=0.7)

    ax.set_xlabel("X (forward)")
    ax.set_ylabel("Y (horizontal)")
    ax.set_zlabel("Z (vertical)")
    ax.set_title("Action Chunk Preview")
    span = max(1.0, float(np.sum(viz.LINK_LENGTHS)))
    ax.set_xlim(0, span)
    ax.set_ylim(-span / 2, span / 2)
    ax.set_zlim(-span / 2, span / 2)
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.001)


class IMUSerialReader:
    def __init__(self, port: str, baud: int = 250000, timeout: float = 0.05):
        if serial is None:
            raise ImportError(
                "pyserial is required for live mode. Install it with: pip install pyserial"
            )
        self.ser = serial.Serial(
            port=port,
            baudrate=baud,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            bytesize=serial.EIGHTBITS,
            timeout=timeout,
        )
        self.enc_by_id: Dict[int, Tuple[int, int]] = {}
        self.rpy_by_id: Dict[int, Tuple[float, float, float]] = {}
        self.lock = threading.Lock()
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self._read_loop, daemon=True)

    def start(self):
        self.thread.start()

    def stop(self):
        self._stop.set()
        try:
            self.ser.close()
        except Exception:
            pass

    def _read_loop(self):
        while not self._stop.is_set():
            try:
                line = self.ser.readline()
                if not line:
                    time.sleep(0.001)
                    continue
                text = line.decode("utf-8", errors="replace").strip()
                m = FRAME_RE.search(text)
                if not m:
                    continue
                inside = m.group(1)
                chunks = [c.strip() for c in inside.split("/") if c.strip()]
                for content in chunks:
                    mm = ID_RE.match(content)
                    if not mm:
                        continue
                    imu_id = int(mm.group(1))
                    csv = mm.group(2)
                    vals = [v for v in csv.split(",") if v.strip() != ""]
                    if len(vals) < 6:
                        continue
                    qw, qx, qy, qz = map(float, vals[0:4])
                    e1 = int(float(vals[4]))
                    e2 = int(float(vals[5]))
                    with self.lock:
                        self.enc_by_id[imu_id] = (e1, e2)
                    if R is not None and not (qw == 0.0 and qx == 0.0 and qy == 0.0 and qz == 0.0):
                        rot = R.from_quat([qx, qy, qz, qw])
                        roll, pitch, yaw = rot.as_euler("xyz", degrees=True)
                        with self.lock:
                            self.rpy_by_id[imu_id] = (roll, pitch, yaw)
            except Exception:
                time.sleep(0.01)

    def get_ordered_ids(self) -> List[int]:
        with self.lock:
            ids = sorted(self.enc_by_id.keys())
        return ids

    def get_state(self, ordered_ids: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        enc = []
        rpy = []
        with self.lock:
            for imu_id in ordered_ids:
                e = self.enc_by_id.get(imu_id, (0, 0))
                enc.append([e[0], e[1]])
                r = self.rpy_by_id.get(imu_id, (0.0, 0.0, 0.0))
                rpy.append([r[0], r[1], r[2]])
        return np.array(enc, dtype=np.float32), np.array(rpy, dtype=np.float32)


class IMUTopicReader:
    def __init__(self, topic: str = "imu_state_all"):
        if rospy is None or Float32MultiArray is None:
            raise ImportError(
                "rospy/std_msgs Float32MultiArray are required for topic IMU mode."
            )
        self.enc_by_id: Dict[int, Tuple[int, int]] = {}
        self.rpy_by_id: Dict[int, Tuple[float, float, float]] = {}
        self.lock = threading.Lock()
        self.topic = topic
        rospy.Subscriber(self.topic, Float32MultiArray, self._cb)

    def start(self):
        return None

    def stop(self):
        return None

    def _cb(self, msg: Float32MultiArray):
        data = list(msg.data)
        n = (len(data) // 6) * 6
        if n <= 0:
            return
        with self.lock:
            for i in range(0, n, 6):
                try:
                    imu_id = int(round(float(data[i])))
                    if imu_id < 1:
                        continue
                    e1 = int(round(float(data[i + 1])))
                    e2 = int(round(float(data[i + 2])))
                    roll = float(data[i + 3])
                    pitch = float(data[i + 4])
                    yaw = float(data[i + 5])
                except Exception:
                    continue
                self.enc_by_id[imu_id] = (e1, e2)
                self.rpy_by_id[imu_id] = (roll, pitch, yaw)

    def get_ordered_ids(self) -> List[int]:
        with self.lock:
            ids = sorted(self.enc_by_id.keys())
        return ids

    def get_state(self, ordered_ids: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        enc = []
        rpy = []
        with self.lock:
            for imu_id in ordered_ids:
                e = self.enc_by_id.get(imu_id, (0, 0))
                enc.append([e[0], e[1]])
                r = self.rpy_by_id.get(imu_id, (0.0, 0.0, 0.0))
                rpy.append([r[0], r[1], r[2]])
        return np.array(enc, dtype=np.float32), np.array(rpy, dtype=np.float32)


class BaseEncoderSubscriber:
    def __init__(self, topic: str = "motor_encoder"):
        if rospy is None or Float32 is None:
            raise ImportError(
                "rospy/std_msgs are required for live mode."
            )
        self.value = 0.0
        self.lock = threading.Lock()
        rospy.Subscriber(topic, Float32, self._cb)

    def _cb(self, msg: Float32):
        with self.lock:
            self.value = float(msg.data)

    def get(self) -> float:
        with self.lock:
            return float(self.value)


def load_policy(ckpt_path: str):
    if not ckpt_path.endswith(".ckpt"):
        ckpt_path = os.path.join(ckpt_path, "checkpoints", "latest.ckpt")
    payload = torch.load(open(ckpt_path, "rb"), map_location="cpu", pickle_module=dill)
    cfg = payload["cfg"]
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model
    return policy, cfg


def _apply_brightness_gain(frame: np.ndarray, gain: float) -> np.ndarray:
    """Same as process_data_parallel: multiplicative gain, clip to uint8."""
    if abs(gain - 1.0) < 1e-6:
        return frame
    out = frame.astype(np.float32) * float(gain)
    return np.clip(out, 0, 255).astype(np.uint8)


def build_obs_dict(
    frames: np.ndarray,
    enc: np.ndarray,
    rpy: np.ndarray,
    base_length: float,
    resize_tf,
    num_cams: int,
    num_imus: int,
    include_keys: Optional[set] = None,
    image_brightness_gain: float = 1.0,
) -> Dict[str, np.ndarray]:
    """Camera tensors are CHW float in [0,1], RGB channel order if resize_tf uses bgr_to_rgb=True."""
    obs = {}
    for i in range(num_cams):
        key = f"camera{i}_rgb"
        if include_keys is not None and key not in include_keys:
            continue
        rgb = resize_tf(frames[i])
        rgb = _apply_brightness_gain(rgb, image_brightness_gain)
        rgb = np.moveaxis(rgb, -1, 0).astype(np.float32) / 255.0
        obs[key] = rgb

    if include_keys is None or "encoder" in include_keys:
        enc_padded = np.zeros((num_imus, 2), dtype=np.float32)
        n = min(num_imus, enc.shape[0])
        if n > 0:
            enc_padded[:n] = enc[:n]
        obs["encoder"] = enc_padded

    if include_keys is None or "imu_rpy" in include_keys:
        rpy_padded = np.zeros((num_imus, 3), dtype=np.float32)
        n = min(num_imus, rpy.shape[0])
        if n > 0:
            rpy_padded[:n] = rpy[:n]
        obs["imu_rpy"] = rpy_padded

    if include_keys is None or "base_encoder" in include_keys:
        obs["base_encoder"] = np.array([base_length], dtype=np.float32)
    if include_keys is None or "base_length_abs" in include_keys:
        obs["base_length_abs"] = np.array([base_length], dtype=np.float32)
    if include_keys is None or "pressure" in include_keys:
        obs["pressure"] = np.zeros((1,), dtype=np.float32)
    if include_keys is None or "motor_torque" in include_keys:
        obs["motor_torque"] = np.zeros((1,), dtype=np.float32)

    return obs


def _make_obs_stack_relative(obs_stack: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    rel = dict(obs_stack)
    for key in ("base_encoder", "joint_angle", "encoder"):
        if key in rel:
            ref = rel[key][-1:].copy()
            rel[key] = rel[key] - ref
    return rel


def _relative_action_to_absolute(
    action_seq: np.ndarray,
    action_repr: str,
    base_ref: float,
    joint_ref: np.ndarray,
    encoder_ref: np.ndarray,
) -> np.ndarray:
    out = action_seq.copy()
    if out.shape[1] >= 1:
        out[:, 0] = out[:, 0] + float(base_ref)
    if out.shape[1] <= 1:
        return out

    if action_repr == "angle":
        assert joint_ref.shape == (out.shape[1] - 1,), f"Expected joint_ref shape {(out.shape[1] - 1,)} but got {joint_ref.shape}"
        print(f"[DEBUG] Converting relative angle action {action_seq} to absolute using joint_ref={joint_ref}")
        out[:, 1:] = out[:, 1:] + joint_ref.reshape(1, -1)
        print(f"[DEBUG] Absolute angle action after conversion: {out}")
    else:
        assert encoder_ref.shape == (out.shape[1] - 1,), f"Expected encoder_ref shape {(out.shape[1] - 1,)} but got {encoder_ref.shape}"
        out[:, 1:] = out[:, 1:] + encoder_ref.reshape(1, -1)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--serial-port", default="/dev/serial/by-id/usb-Teensyduino_USB_Serial_17923780-if00")
    parser.add_argument("--baud", type=int, default=250000)
    parser.add_argument(
        "--imu-source",
        choices=("serial", "topic"),
        default="serial",
        help="IMU input source: direct serial or ROS topic published by controller replay.",
    )
    parser.add_argument(
        "--imu-topic",
        type=str,
        default="imu_state_all",
        help="ROS topic carrying IMU state when --imu-source topic.",
    )
    parser.add_argument("--control-rate", type=float, default=0.2)
    parser.add_argument("--camera-fps", type=int, default=15)
    parser.add_argument("--num-cams", type=int, default=None, help="Override camera count; default from config/device list.")
    parser.add_argument("--device-list", type=str, default=None, help="Path to text file with one device per line.")
    parser.add_argument("--node-name", type=str, default="policy_inference")
    parser.add_argument("--execute-steps", type=int, default=16, help="How many actions from the chunk to execute.")
    parser.add_argument("--debug-plot", action="store_true", help="Plot action chunk and require confirmation before executing.")
    parser.add_argument("--print-all-actions", action="store_true", help="Print every predicted action before execution.")
    parser.add_argument(
        "--debug-viz-out-dir",
        type=str,
        default="outputs/policy_debug_viz",
        help="Directory for debug RGB/proprio/action plots when --debug-plot is enabled.",
    )
    parser.add_argument(
        "--debug-viz-max-cameras",
        type=int,
        default=99,
        help="Maximum number of camera streams to draw in debug RGB visualization.",
    )
    parser.add_argument(
        "--predict-avg-samples",
        type=int,
        default=1,
        help="Average this many stochastic policy samples per observation before execution.",
    )
    parser.add_argument(
        "--viz-gamma",
        type=float,
        default=1.0,
        help="Visualization-only gamma for debug images.",
    )
    parser.add_argument(
        "--viz-contrast",
        type=float,
        default=1.0,
        help="Visualization-only contrast scale for debug images.",
    )
    parser.add_argument(
        "--viz-autowb",
        action="store_true",
        help="Visualization-only gray-world auto white balance for debug images.",
    )
    parser.add_argument(
        "--image-brightness-gain",
        type=float,
        default=1.5,
        help=(
            "Match process_data_parallel --image-brightness-gain (multiplier on 8-bit image after resize; "
            "applied in RGB order when BGR→RGB is enabled)."
        ),
    )
    parser.add_argument(
        "--no-bgr-to-rgb",
        action="store_true",
        help=(
            "Keep OpenCV BGR order in the tensor (default: convert BGR→RGB to match zarr training data "
            "from process_data_parallel)."
        ),
    )
    parser.add_argument(
        "--relative-action-ref-source",
        type=str,
        default=None,
        choices=("sensor", "commanded"),
        help=(
            "Reference source used to convert relative actions to absolute when "
            "use_relative_action_obs=True. "
            "Defaults to cfg.task.relative_action_ref_source if set, else 'sensor'."
        ),
    )
    args = parser.parse_args()

    debug_viz_out_dir = pathlib.Path(args.debug_viz_out_dir).expanduser().resolve()
    if args.debug_plot:
        debug_viz_out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Debug visualizations will be saved to: {debug_viz_out_dir}", flush=True)

    policy, cfg = load_policy(args.ckpt)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy.eval().to(device)

    missing_live_deps = []
    if rospy is None or Float32 is None or Float32MultiArray is None or Int32MultiArray is None:
        missing_live_deps.append("rospy/std_msgs")
    if get_image_transform is None:
        missing_live_deps.append("process_data_parallel dependencies")
    if angles_rad_to_encoder_ticks is None or enc_list_to_joint_angles is None:
        missing_live_deps.append("encoder_angle_mapping dependencies")
    if USBCamera is None or camServer is None:
        missing_live_deps.append("camera runtime dependencies")
    if args.imu_source == "serial" and serial is None:
        missing_live_deps.append("pyserial")
    if viz is None:
        missing_live_deps.append("cameras visualization dependencies")
    if missing_live_deps:
        raise ImportError(
            "Live mode missing dependencies: " + ", ".join(missing_live_deps)
        )

    rospy.init_node(args.node_name, anonymous=True)
    pub_base = rospy.Publisher("motor_encoder_replay", Float32, queue_size=10)
    pub_bend = rospy.Publisher("encoders_replay_all", Int32MultiArray, queue_size=10)
    print(f"[INFO] DRY_RUN={DRY_RUN} (policy outputs will not be published when True)", flush=True)

    img_obs_horizon = cfg.task.img_obs_horizon
    low_dim_obs_horizon = cfg.task.low_dim_obs_horizon
    num_imus = cfg.task.num_imus
    num_segments = int(getattr(cfg.task, "num_segments", num_imus))
    action_repr = getattr(cfg.task, "action_repr", "angle")
    use_relative_action_obs = bool(getattr(cfg.task, "use_relative_action_obs", False))
    relative_action_ref_source = args.relative_action_ref_source
    if relative_action_ref_source is None:
        relative_action_ref_source = str(getattr(cfg.task, "relative_action_ref_source", "sensor"))
    relative_action_ref_source = relative_action_ref_source.strip().lower()
    if relative_action_ref_source not in ("sensor", "commanded"):
        print(
            f"[WARN] Unsupported relative_action_ref_source='{relative_action_ref_source}'. "
            "Falling back to 'sensor'.",
            flush=True,
        )
        relative_action_ref_source = "sensor"
    if not use_relative_action_obs and relative_action_ref_source != "sensor":
        print(
            "[INFO] relative_action_ref_source is ignored because use_relative_action_obs=False.",
            flush=True,
        )
    execute_steps = args.execute_steps
    if execute_steps is None:
        execute_steps = int(getattr(cfg.task, "execute_steps", 1))
    execute_steps = max(1, execute_steps)

    obs_key_set = {
        k
        for k, attr in cfg.task.shape_meta.obs.items()
        if not attr.get("ignore_by_policy", False)
    }
    print(
        f"[INFO] configs: img_obs_horizon={img_obs_horizon}, "
        f"low_dim_obs_horizon={low_dim_obs_horizon}, num_imus={num_imus}, num_segments={num_segments}, "
        f"action_repr={action_repr}, use_relative_action_obs={use_relative_action_obs}, "
        f"joint_angle_abs_in_obs={'joint_angle_abs' in obs_key_set}, "
        f"relative_action_ref_source={relative_action_ref_source}, "
        f"execute_steps={execute_steps}",
        flush=True,
    )
    print(
        f"[INFO] action postprocess: predict_avg_samples={max(1, int(args.predict_avg_samples))}",
        flush=True,
    )
    print(
        f"[INFO] image_brightness_gain={float(args.image_brightness_gain)} "
        "(must match value used in process_data_parallel for training data)",
        flush=True,
    )
    print(
        f"[INFO] bgr_to_rgb={not bool(args.no_bgr_to_rgb)} "
        "(must match process_data_parallel unless dataset was built with --no-bgr-to-rgb)",
        flush=True,
    )

    if args.device_list is not None:
        with open(args.device_list, "r") as f:
            device_ids = [ln.strip() for ln in f.readlines() if ln.strip()]
    else:
        device_ids = [
            "/dev/v4l/by-path/pci-0000:38:00.0-usb-0:2:1.0-video-index0",    #0: S11
            "/dev/v4l/by-path/pci-0000:00:14.0-usb-0:11:1.0-video-index0",   #1: S21
            "/dev/v4l/by-path/pci-0000:00:14.0-usb-0:7:1.0-video-index0",    #2: S31
            "/dev/v4l/by-path/pci-0000:00:14.0-usb-0:8:1.0-video-index0",    #3: S32
            "/dev/v4l/by-path/pci-0000:1b:00.0-usb-0:2:1.0-video-index0",    #4: S33
            "/dev/v4l/by-path/pci-0000:00:14.0-usb-0:9:1.0-video-index0",    #5: S34
            "/dev/v4l/by-path/pci-0000:37:00.0-usb-0:2:1.0-video-index0",    #6: S35
            "/dev/v4l/by-path/pci-0000:00:14.0-usb-0:10:1.0-video-index0",   #7: S41
            "/dev/v4l/by-path/pci-0000:28:00.0-usb-0:2:1.0-video-index0",    #8: S51
            "/dev/v4l/by-path/pci-0000:26:00.0-usb-0:2:1.0-video-index0",    #9: S52
            "/dev/v4l/by-path/pci-0000:1f:00.0-usb-0:2:1.0-video-index0",    #10: S53
            "/dev/v4l/by-path/pci-0000:27:00.0-usb-0:2:1.0-video-index0",    #11: S54
            "/dev/v4l/by-path/pci-0000:00:14.0-usb-0:12:1.0-video-index0",   #12: S55
            "/dev/v4l/by-path/pci-0000:00:14.0-usb-0:3:1.0-video-index0",    #13: S61
            "/dev/v4l/by-path/pci-0000:2e:00.0-usb-0:2:1.0-video-index0",    #14: S71
            "/dev/v4l/by-path/pci-0000:2f:00.0-usb-0:2:1.0-video-index0",    #15: S72
            "/dev/v4l/by-path/pci-0000:30:00.0-usb-0:2:1.0-video-index0",    #16: S73
            "/dev/v4l/by-path/pci-0000:36:00.0-usb-0:2:1.0-video-index0",    #17: S74
            "/dev/v4l/by-path/pci-0000:2b:00.0-usb-0:2:1.0-video-index0",    #18: S75
        ]

    if args.num_cams is None:
        cam_keys = [k for k in obs_key_set if k.startswith("camera") and k.endswith("_rgb")]
        num_cams = len(cam_keys) if len(cam_keys) > 0 else len(device_ids)
    else:
        num_cams = args.num_cams
    device_ids = device_ids[:num_cams]

    usbcams = [USBCamera(device_id=idx, fps=args.camera_fps) for idx in device_ids]
    shm_manager = SharedMemoryManager()
    shm_manager.start()
    server = camServer(shm_manager, usbcams)
    time.sleep(2)

    if args.imu_source == "serial":
        imu_reader = IMUSerialReader(args.serial_port, baud=args.baud)
        imu_source_desc = f"serial:{args.serial_port}"
    else:
        imu_reader = IMUTopicReader(topic=args.imu_topic)
        imu_source_desc = f"topic:{args.imu_topic}"
    imu_reader.start()
    print(f"[INFO] IMU source: {imu_source_desc}", flush=True)
    base_sub = BaseEncoderSubscriber(topic="motor_encoder")

    # Image transform (OpenCV/USB cameras are BGR; training zarr uses RGB by default)
    in_res = (640, 480)
    out_res = (224, 224)
    resize_tf = get_image_transform(
        in_res,
        out_res,
        crop_ratio_h=1.0,
        crop_ratio_w=1.0,
        bgr_to_rgb=not bool(args.no_bgr_to_rgb),
    )

    hist: Dict[str, deque] = {}
    obs_keys = list(cfg.task.shape_meta.obs.keys())
    for key in obs_keys:
        hist[key] = deque(maxlen=max(img_obs_horizon, low_dim_obs_horizon))

    rate = rospy.Rate(args.control_rate)
    chunk_idx = 0
    prev_exec_action = None
    last_commanded_action: Optional[np.ndarray] = None
    pending_obs_ctx: Optional[Tuple[List[int], np.ndarray, float]] = None
    locked_ordered_ids: Optional[List[int]] = None
    last_imu_count_logged: Optional[int] = None
    imu_id_drift_warned = False

    def _capture_and_append_obs() -> Optional[Tuple[List[int], np.ndarray, float]]:
        """Single observation capture path used by both chunk-start and per-step updates."""
        nonlocal locked_ordered_ids, last_imu_count_logged, imu_id_drift_warned
        frames_now, _, _ = server.get_data()
        if frames_now is None or len(frames_now) < num_cams:
            return None

        observed_ids = imu_reader.get_ordered_ids()
        if len(observed_ids) != num_imus:
            if last_imu_count_logged != len(observed_ids):
                print(
                    f"[WARN] Waiting for exactly {num_imus} joint IDs; got {len(observed_ids)}: {observed_ids}",
                    flush=True,
                )
                last_imu_count_logged = len(observed_ids)
            return None
        if locked_ordered_ids is None:
            locked_ordered_ids = observed_ids
            print(f"[INFO] Locked joint ID order: {locked_ordered_ids}", flush=True)
        ordered_ids_now = locked_ordered_ids
        if observed_ids != locked_ordered_ids and not imu_id_drift_warned:
            print(
                f"[WARN] Observed joint IDs changed after lock. locked={locked_ordered_ids}, observed={observed_ids}",
                flush=True,
            )
            imu_id_drift_warned = True

        enc_now, rpy_now = imu_reader.get_state(ordered_ids_now)
        base_enc_now = base_sub.get()
        base_length_now = base_encoder_to_length(base_enc_now)

        obs_step_now = build_obs_dict(
            frames=frames_now,
            enc=enc_now,
            rpy=rpy_now,
            base_length=base_length_now,
            resize_tf=resize_tf,
            num_cams=num_cams,
            num_imus=num_imus,
            include_keys=obs_key_set,
            image_brightness_gain=float(args.image_brightness_gain),
        )
        if getattr(cfg.task, "use_joint_angle_obs", False) and "joint_angle" in obs_key_set:
            enc_flat_now = enc_now.reshape(-1)
            ja_full = enc_list_to_joint_angles(enc_flat_now).astype(np.float32)
            obs_step_now["joint_angle"] = ja_full[:num_segments]
            if "joint_angle_abs" in obs_key_set:
                obs_step_now["joint_angle_abs"] = obs_step_now["joint_angle"].copy()

        for key, value in obs_step_now.items():
            hist[key].append(value)
        return ordered_ids_now, enc_now, base_length_now

    while not rospy.is_shutdown():
        if pending_obs_ctx is None:
            obs_ctx = _capture_and_append_obs()
            if obs_ctx is None:
                rate.sleep()
                continue
        else:
            obs_ctx = pending_obs_ctx
            pending_obs_ctx = None

        ordered_ids, enc, base_length = obs_ctx

        obs_stack = {}
        for key in obs_keys:
            values = list(hist[key])
            if len(values) == 0:
                continue
            while len(values) < (img_obs_horizon if "camera" in key and key.endswith("_rgb") else low_dim_obs_horizon):
                values.insert(0, values[0])
            obs_stack[key] = np.stack(values[-(img_obs_horizon if "camera" in key and key.endswith("_rgb") else low_dim_obs_horizon):], axis=0)

        model_obs = dict(obs_stack)
        if use_relative_action_obs and "base_length_abs" in obs_key_set:
            model_obs["base_length_abs"] = obs_stack["base_encoder"].copy()
        if (
            use_relative_action_obs
            and "joint_angle_abs" in obs_key_set
            and "joint_angle" in obs_stack
        ):
            model_obs["joint_angle_abs"] = obs_stack["joint_angle"].copy()
        model_obs = (
            _make_obs_stack_relative(model_obs) if use_relative_action_obs else model_obs
        )
        obs_tensor = dict_apply(
            model_obs,
            lambda x: torch.from_numpy(x).unsqueeze(0).to(device),
        )
        action_seq = _predict_action_seq(
            policy=policy,
            obs_tensor=obs_tensor,
            num_samples=max(1, int(args.predict_avg_samples)),
        )
        if use_relative_action_obs:
            encoder_ref = enc.reshape(-1).astype(np.float32)
            base_ref = float(base_length)
            joint_ref = enc_list_to_joint_angles(encoder_ref).astype(np.float32)[:num_segments]
            ref_source_used = "sensor"

            if relative_action_ref_source == "commanded":
                if last_commanded_action is None:
                    ref_source_used = "sensor (no last commanded action yet)"
                elif last_commanded_action.shape[0] != action_seq.shape[1]:
                    print(
                        f"[WARN] last_commanded_action dim={last_commanded_action.shape[0]} "
                        f"does not match action dim={action_seq.shape[1]}; falling back to sensor ref.",
                        flush=True,
                    )
                    ref_source_used = "sensor (shape mismatch)"
                else:
                    base_ref = float(last_commanded_action[0])
                    if action_repr == "angle":
                        joint_ref = np.asarray(last_commanded_action[1:], dtype=np.float32)
                    else:
                        encoder_ref = np.asarray(last_commanded_action[1:], dtype=np.float32)
                    ref_source_used = "last_commanded_action"

            if action_repr == "angle":
                print(
                    f"[DEBUG] Using relative action obs. ref_source={ref_source_used}, "
                    f"base_ref={base_ref:.6f}, "
                    f"joint_ref={np.array2string(joint_ref, precision=5, suppress_small=True)}",
                    flush=True,
                )
            else:
                print(
                    f"[DEBUG] Using relative action obs. ref_source={ref_source_used}, "
                    f"base_ref={base_ref:.6f}, "
                    f"encoder_ref_shape={encoder_ref.shape}",
                    flush=True,
                )
            action_seq = _relative_action_to_absolute(
                action_seq=action_seq,
                action_repr=action_repr,
                base_ref=base_ref,
                joint_ref=joint_ref,
                encoder_ref=encoder_ref,
            )
        if args.print_all_actions:
            print("[INFO] Predicted action_seq:", flush=True)
            print(np.array2string(action_seq, precision=5, suppress_small=False), flush=True)

        chunk_idx += 1

        steps_to_execute = min(execute_steps, action_seq.shape[0])

        if args.debug_plot:
            _save_debug_policy_viz(
                obs_stack=obs_stack,
                action_seq=action_seq,
                chunk_idx=chunk_idx,
                out_dir=debug_viz_out_dir,
                max_cameras=args.debug_viz_max_cameras,
                viz_gamma=args.viz_gamma,
                viz_contrast=args.viz_contrast,
                viz_autowb=args.viz_autowb,
            )
            print(
                f"[DEBUG] Saved debug plots for chunk {chunk_idx} to {debug_viz_out_dir}",
                flush=True,
            )

            if action_repr == "angle":
                angle_seq = action_seq[:steps_to_execute, 1:]
                if angle_seq.shape[1] > num_segments:
                    angle_seq = angle_seq[:, :num_segments]
            else:
                enc_seq = action_seq[:steps_to_execute, 1:]
                angle_seq = np.stack(
                    [enc_list_to_joint_angles(row) for row in enc_seq], axis=0
                )
            base_seq = action_seq[:steps_to_execute, 0]
            _debug_plot_action_chunk(base_seq, angle_seq)
            resp = input(f"Execute {steps_to_execute} actions? [y/N]: ").strip().lower()
            if resp not in ("y", "yes"):
                print("Skipped action chunk.")
                rate.sleep()
                continue

        for step_idx in range(steps_to_execute):
            action = action_seq[step_idx]
            base_target_length = float(action[0])
            base_target_enc = base_length_to_encoder(base_target_length)
            if args.print_all_actions:
                print(
                    f"[STEP {step_idx + 1}/{steps_to_execute}] "
                    f"base_len={base_target_length:.6f}, base_enc={base_target_enc:.3f}, "
                    f"action={np.array2string(action, precision=5, suppress_small=False)}",
                    flush=True,
                )
            if not DRY_RUN:
                pub_base.publish(Float32(data=float(base_target_enc)))

            bend = []
            if action_repr == "angle":
                angles = action[1:]
                if angles.shape[0] > num_segments:
                    angles = angles[:num_segments]
                enc_ticks = angles_rad_to_encoder_ticks(angles)
                for i, imu_id in enumerate(ordered_ids[:num_segments]):
                    if i >= len(enc_ticks):
                        break
                    e1 = int(round(float(enc_ticks[i])))
                    e2 = -e1
                    bend.extend([int(imu_id), e1, e2])
            else:
                for i, imu_id in enumerate(ordered_ids[:num_imus]):
                    idx = 1 + 2 * i
                    if idx >= action.shape[0]:
                        break
                    e1 = int(round(float(action[idx])))
                    e2 = -e1
                    bend.extend([int(imu_id), e1, e2])
            msg = Int32MultiArray(data=bend)
            if args.print_all_actions:
                print(
                    f"[STEP {step_idx + 1}/{steps_to_execute}] "
                    f"publish encoders_replay_all msg.data={msg.data} "
                    f"(published={not DRY_RUN})",
                    flush=True,
                )
            if not DRY_RUN:
                pub_bend.publish(msg)
            last_commanded_action = np.asarray(action, dtype=np.float32).copy()

            rate.sleep()

            # Update history after every executed step through the single capture path.
            obs_ctx_after = _capture_and_append_obs()
            if obs_ctx_after is None:
                pending_obs_ctx = None
                if args.print_all_actions:
                    print(
                        f"[STEP {step_idx + 1}/{steps_to_execute}] "
                        "skipped history update (missing sensor data)",
                        flush=True,
                    )
                continue
            pending_obs_ctx = obs_ctx_after

    imu_reader.stop()
    server.end()


if __name__ == "__main__":
    main()
