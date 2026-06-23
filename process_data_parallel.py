import os
import glob
import json
from typing import Dict, List, Optional, Tuple

import cv2
import zarr
import numpy as np
from tqdm import tqdm
import logging
import concurrent.futures
import multiprocessing

from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs, JpegXl
from encoder_angle_mapping import enc_list_to_joint_angles
from base_length_mapping import encoders_to_length

register_codecs()
logger = logging.getLogger(__name__)


def flip(imgs):
    return np.flip(np.flip(imgs, axis=1), axis=2)


def get_image_transform(
    in_res,
    out_res,
    crop_ratio_h: float = 1.0,
    crop_ratio_w: float = 1.0,
    bgr_to_rgb: bool = False,
    w_slice_start=None,
    h_slice_start=None,
):
    iw, ih = in_res
    ow, oh = out_res
    ch = round(ih * crop_ratio_h)
    cw = round(ih * crop_ratio_w / oh * ow)
    interp_method = cv2.INTER_AREA

    if w_slice_start is None:
        w_slice_start = (iw - cw) // 2
    else:
        w_slice_start = round(iw * w_slice_start)
    w_slice = slice(w_slice_start, w_slice_start + cw)
    if h_slice_start is None:
        h_slice_start = (ih - ch) // 2
    else:
        h_slice_start = round(ih * h_slice_start)
    h_slice = slice(h_slice_start, h_slice_start + ch)
    c_slice = slice(None)
    if bgr_to_rgb:
        c_slice = slice(None, None, -1)

    def transform(img: np.ndarray):
        if img.shape != ((ih, iw, 3)):
            # Fallback to direct resize when camera resolution varies.
            out = cv2.resize(img, out_res, interpolation=interp_method)
            if bgr_to_rgb:
                out = out[:, :, ::-1]
            return out
        img = img[h_slice, w_slice, c_slice]
        img = cv2.resize(img, out_res, interpolation=interp_method)
        return img

    return transform


def _load_camera_data_video(base_path):
    timestamp_path = base_path + "_camera_timestamps.json"
    if not os.path.exists(timestamp_path):
        return None
    with open(timestamp_path, "r") as f:
        ts_data = json.load(f)
    timestamps = np.array(ts_data["timestamps"], dtype=np.float64)
    num_cameras = int(ts_data["num_cameras"])
    resolution = tuple(ts_data["resolution"])
    fps = ts_data.get("fps", 15)
    video_paths = [
        f"{base_path}_camera_{cam_idx:02d}.mp4" for cam_idx in range(num_cameras)
    ]
    paths_by_cam_id = {i: p for i, p in enumerate(video_paths)}
    return timestamps, resolution, fps, paths_by_cam_id


def _load_camera_data_avi(camera_root: str, session_id: str):
    camera_dir = os.path.join(camera_root, session_id)
    if not os.path.isdir(camera_dir):
        return None
    timestamp_paths = glob.glob(os.path.join(camera_dir, "cam_*_timestamps.csv"))
    video_paths = glob.glob(os.path.join(camera_dir, "cam_*.avi"))
    if not timestamp_paths or not video_paths:
        return None

    def _cam_index(path: str, suffix: str) -> Optional[int]:
        name = os.path.basename(path)
        if not name.startswith("cam_") or not name.endswith(suffix):
            return None
        idx_str = name[len("cam_") : -len(suffix)]
        if not idx_str.isdigit():
            return None
        return int(idx_str)

    ts_map = {}
    for path in timestamp_paths:
        idx = _cam_index(path, "_timestamps.csv")
        if idx is not None:
            ts_map[idx] = path

    vid_map = {}
    for path in video_paths:
        idx = _cam_index(path, ".avi")
        if idx is not None:
            vid_map[idx] = path

    cam_indices = sorted(set(ts_map.keys()) & set(vid_map.keys()))
    if not cam_indices:
        return None

    # Use the first camera's timestamps as the reference timeline.
    ref_data = np.genfromtxt(
        ts_map[cam_indices[0]], delimiter=",", names=True, dtype=None, encoding=None
    )
    if ref_data is None or ref_data.size == 0:
        return None
    if "timestamp_unix" in ref_data.dtype.names:
        ref_ts = ref_data["timestamp_unix"]
    elif "timestamp" in ref_data.dtype.names:
        ref_ts = ref_data["timestamp"]
    else:
        ref_ts = ref_data[ref_data.dtype.names[-1]]
    timestamps = np.asarray(ref_ts, dtype=np.float64)

    first_video = vid_map[cam_indices[0]]
    cap = cv2.VideoCapture(first_video)
    ok, frame = cap.read()
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if not ok or frame is None:
        return None
    resolution = (frame.shape[1], frame.shape[0])
    paths_by_cam_id = {int(idx): vid_map[idx] for idx in cam_indices}
    return timestamps, resolution, fps, paths_by_cam_id


def _load_camera_data_video_from_dir(camera_root: str, session_id: str):
    """
    Look for *_camera_timestamps.json + *_camera_XX.mp4 inside
    camera_root/session_id and load if found.
    """
    camera_dir = os.path.join(camera_root, session_id)
    if not os.path.isdir(camera_dir):
        return None
    ts_paths = sorted(glob.glob(os.path.join(camera_dir, "*_camera_timestamps.json")))
    if not ts_paths:
        return None
    # Use the first timestamps file found.
    base_path = ts_paths[0].replace("_camera_timestamps.json", "")
    return _load_camera_data_video(base_path)


def _read_video_frames(video_path: str, target_count: Optional[int] = None):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        if target_count is not None and len(frames) >= target_count:
            break
    cap.release()
    return frames


def _read_video_frames_indices(video_path: str, keep_idx: np.ndarray):
    keep_set = set(int(i) for i in keep_idx)
    if not keep_set:
        return []
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_idx = 0
    max_idx = max(keep_set)
    while frame_idx <= max_idx:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx in keep_set:
            frames.append(frame)
        frame_idx += 1
    cap.release()
    return frames


def _decode_and_resize_camera(
    cam_idx: int,
    video_path: str,
    keep_idx: np.ndarray,
    num_frames: int,
    resize_tf,
    out_res: Tuple[int, int],
    expect_h: int,
    expect_w: int,
    image_brightness_gain: float,
):
    if not os.path.exists(video_path):
        return cam_idx, None
    frames = _read_video_frames_indices(video_path, keep_idx)
    frames = frames[:num_frames]
    resized_frames = []
    for frame in frames:
        frame = _ensure_color_frame(frame)
        if frame is None:
            continue
        if frame.shape[:2] != (expect_h, expect_w):
            out = cv2.resize(frame, out_res, interpolation=cv2.INTER_AREA)
        else:
            out = resize_tf(frame)
        out = _apply_brightness_gain(out, image_brightness_gain)
        resized_frames.append(out)
    if not resized_frames:
        return cam_idx, None
    resized = np.stack(resized_frames, axis=0)
    return cam_idx, resized


def _subsample_by_timestamp(
    timestamps: np.ndarray, target_hz: Optional[float]
) -> Tuple[np.ndarray, np.ndarray]:
    if target_hz is None or target_hz <= 0 or timestamps.size == 0:
        idx = np.arange(len(timestamps), dtype=np.int64)
        return timestamps, idx
    step = 1.0 / float(target_hz)
    keep = []
    last_t = -np.inf
    for i, t in enumerate(timestamps):
        if t - last_t >= step:
            keep.append(i)
            last_t = t
    idx = np.asarray(keep, dtype=np.int64)
    return timestamps[idx], idx


def _stack_values(values: np.ndarray) -> np.ndarray:
    if values is None:
        return None
    values = np.asarray(values)
    if values.dtype == object:
        try:
            values = np.stack(values)
        except ValueError:
            values = values.astype(np.float32)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    return values.astype(np.float32)


def _ensure_color_frame(frame: np.ndarray) -> Optional[np.ndarray]:
    if frame is None:
        return None
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if frame.ndim == 3 and frame.shape[2] == 3:
        return frame
    if frame.ndim == 3 and frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    return None


def _apply_brightness_gain(frame: np.ndarray, gain: float) -> np.ndarray:
    if abs(gain - 1.0) < 1e-6:
        return frame
    out = frame.astype(np.float32) * float(gain)
    return np.clip(out, 0, 255).astype(np.uint8)


def _black_camera_stack(num_frames: int, out_res: Tuple[int, int]) -> np.ndarray:
    """RGB or BGR uint8 stack (T, H, W, 3); zeros read as black in either convention."""
    ow, oh = int(out_res[0]), int(out_res[1])
    return np.zeros((num_frames, oh, ow, 3), dtype=np.uint8)


def _finalize_camera_frames_black_pad(
    frames_per_cam: List[Optional[np.ndarray]],
    num_frames: int,
    out_res: Tuple[int, int],
    session_id: str,
) -> List[np.ndarray]:
    """Truncate to num_frames; missing or short streams become black (or black tail)."""
    result: List[np.ndarray] = []
    black_idxs: List[int] = []
    for i, f in enumerate(frames_per_cam):
        if f is None:
            result.append(_black_camera_stack(num_frames, out_res))
            black_idxs.append(i)
        elif len(f) < num_frames:
            tail = _black_camera_stack(num_frames - len(f), out_res)
            result.append(np.concatenate([f, tail], axis=0))
        else:
            result.append(f[:num_frames].copy())
    if black_idxs:
        print(
            f"[PAD] {session_id}: black-filled missing cameras {black_idxs} "
            f"(T={num_frames}, n_cams={len(frames_per_cam)})",
            flush=True,
        )
    return result


def _moving_average_2d(data: np.ndarray, window: int) -> np.ndarray:
    data = np.asarray(data, dtype=np.float32)
    if window <= 1 or data.shape[0] <= 1:
        return data
    if window % 2 == 0:
        window += 1
    radius = window // 2
    padded = np.pad(data, ((radius, radius), (0, 0)), mode="edge")
    out = np.zeros_like(data, dtype=np.float32)
    for i in range(data.shape[0]):
        out[i] = np.mean(padded[i : i + window], axis=0)
    return out


def _compute_motion_keep_mask(
    base_length: np.ndarray,
    joint_angles: np.ndarray,
    eps_base: float,
    eps_joint: float,
) -> np.ndarray:
    n = base_length.shape[0]
    if n <= 1:
        return np.ones((n,), dtype=bool)
    db = np.abs(np.diff(base_length.reshape(n, -1), axis=0)).max(axis=1)
    dj = np.abs(np.diff(joint_angles.reshape(n, -1), axis=0)).max(axis=1)
    motion = (db > float(eps_base)) | (dj > float(eps_joint))
    keep = np.zeros((n,), dtype=bool)
    keep[0] = True
    keep[1:] = motion
    if not np.any(keep):
        keep[0] = True
    return keep


def _save_processed_episode_viz(
    episode_data: Dict[str, np.ndarray],
    session_id: str,
    out_dir: str,
    max_cameras: int = 8,
    image_steps: int = 2,
    max_plot_points: int = 256,
    bgr_to_rgb: bool = True,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        logger.warning("matplotlib not available; skip processed visualization for %s", session_id)
        return

    out_path = os.path.abspath(os.path.expanduser(out_dir))
    os.makedirs(out_path, exist_ok=True)

    rgb_keys = sorted([k for k in episode_data.keys() if k.startswith("camera") and k.endswith("_rgb")])
    lowdim_keys = sorted(
        [
            k
            for k in episode_data.keys()
            if k not in rgb_keys and k != "action"
        ]
    )
    if len(rgb_keys) > 0:
        keys = rgb_keys[: max(1, int(max_cameras))]
        T = int(episode_data[keys[0]].shape[0])
        ncols = max(1, int(image_steps))
        if T <= ncols:
            step_ids = np.arange(T, dtype=np.int64)
            ncols = T
        else:
            step_ids = np.linspace(0, T - 1, ncols, dtype=np.int64)

        fig, axes = plt.subplots(
            nrows=len(keys),
            ncols=ncols,
            figsize=(2.8 * ncols, 2.4 * len(keys)),
            squeeze=False,
        )
        for r, key in enumerate(keys):
            seq = episode_data[key]  # (T,H,W,3) uint8 — RGB if bgr_to_rgb else BGR
            for c, t in enumerate(step_ids):
                img = seq[int(t)]
                # Matplotlib expects RGB; dataset may store BGR when bgr_to_rgb=False.
                if not bgr_to_rgb:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = img.astype(np.float32) / 255.0
                ax = axes[r, c]
                ax.imshow(np.clip(img, 0.0, 1.0))
                if r == 0:
                    ax.set_title(f"t={int(t)}")
                if c == 0:
                    ax.set_ylabel(key, rotation=0, ha="right", va="center")
                ax.set_xticks([])
                ax.set_yticks([])
        fig.suptitle(f"{session_id}: processed RGB slices")
        fig.tight_layout()
        fig.savefig(os.path.join(out_path, f"{session_id}_images.png"), dpi=150)
        plt.close(fig)

    action = episode_data.get("action", None)
    if action is None:
        return
    action = np.asarray(action, dtype=np.float32)
    T = int(action.shape[0])
    if T > max_plot_points:
        keep = np.linspace(0, T - 1, max_plot_points, dtype=np.int64)
    else:
        keep = np.arange(T, dtype=np.int64)
    x = np.arange(len(keep))

    lowdim_flat = []
    for key in lowdim_keys:
        arr = np.asarray(episode_data[key], dtype=np.float32)
        if arr.shape[0] != T:
            continue
        lowdim_flat.append((key, arr.reshape(T, -1)[keep]))
    action_flat = action.reshape(T, -1)[keep]

    n_plots = int(sum(v.shape[1] for _, v in lowdim_flat) + action_flat.shape[1])
    if n_plots <= 0:
        return
    fig, axes = plt.subplots(
        nrows=n_plots,
        ncols=1,
        figsize=(11, max(4, 1.6 * n_plots)),
        squeeze=False,
    )
    axes = axes[:, 0]
    i = 0
    for key, flat in lowdim_flat:
        for d in range(flat.shape[1]):
            ax = axes[i]
            ax.plot(x, flat[:, d], marker="o", linestyle="None", markersize=2)
            ax.set_title(f"{key}[{d}]")
            ax.set_xlabel("time index (subsampled)")
            ax.set_ylabel("value")
            ax.grid(alpha=0.3)
            i += 1
    for d in range(action_flat.shape[1]):
        ax = axes[i]
        ax.plot(x, action_flat[:, d], marker="o", linestyle="None", markersize=2)
        ax.set_title(f"action[{d}]")
        ax.set_xlabel("time index (subsampled)")
        ax.set_ylabel("value")
        ax.grid(alpha=0.3)
        i += 1
    fig.suptitle(f"{session_id}: processed proprio and action")
    fig.tight_layout()
    fig.savefig(os.path.join(out_path, f"{session_id}_signals.png"), dpi=150)
    plt.close(fig)


def _resample_nearest(src_ts: np.ndarray, src_values: np.ndarray, tgt_ts: np.ndarray):
    if src_ts is None or src_values is None or len(src_ts) == 0:
        return None
    src_ts = np.asarray(src_ts, dtype=np.float64)
    order = np.argsort(src_ts)
    src_ts = src_ts[order]
    src_values = src_values[order]

    if src_values.ndim == 1:
        src_values = src_values.reshape(-1, 1)

    valid_mask = np.isfinite(src_ts)
    valid_mask &= np.isfinite(src_values).all(axis=-1)
    if not np.any(valid_mask):
        return None
    src_ts = src_ts[valid_mask]
    src_values = src_values[valid_mask]

    idx = np.searchsorted(src_ts, tgt_ts, side="left")
    idx = np.clip(idx, 1, len(src_ts) - 1)
    left = idx - 1
    right = idx
    left_ts = src_ts[left]
    right_ts = src_ts[right]
    choose_right = (np.abs(tgt_ts - right_ts) < np.abs(tgt_ts - left_ts))
    nearest_idx = np.where(choose_right, right, left)
    return src_values[nearest_idx]


def _align_timestamps(camera_ts: np.ndarray, sensor_ts: np.ndarray):
    if len(camera_ts) == 0 or len(sensor_ts) == 0:
        return camera_ts
    cam_abs = camera_ts[0] > 1e9
    sensor_abs = sensor_ts[0] > 1e9
    if cam_abs and not sensor_abs:
        camera_ts = camera_ts - camera_ts[0] + sensor_ts[0]
    elif sensor_abs and not cam_abs:
        camera_ts = camera_ts + sensor_ts[0]
    return camera_ts


def _load_camera_pose_template(
    num_cameras: int,
    camera_pose_path: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if camera_pose_path is None:
        camera_s = np.linspace(0.0, 1.0, num=num_cameras, dtype=np.float32)
        camera_pos = np.zeros((num_cameras, 3), dtype=np.float32)
        camera_pos[:, 0] = camera_s * 0.05 * (num_cameras - 1)
        camera_ori = np.zeros((num_cameras, 6), dtype=np.float32)
        camera_ori[:, 0] = 1.0
        camera_ori[:, 4] = 1.0
        return camera_pos, camera_ori, camera_s.reshape(-1, 1)

    if camera_pose_path.endswith(".npz"):
        data = np.load(camera_pose_path, allow_pickle=True)
        camera_pos = data["camera_pos"].astype(np.float32)
        camera_ori = data["camera_ori"].astype(np.float32)
        camera_s = data["camera_s"].astype(np.float32)
        data.close()
        return camera_pos, camera_ori, camera_s
    if camera_pose_path.endswith(".json"):
        with open(camera_pose_path, "r") as f:
            data = json.load(f)
        camera_pos = np.array(data["camera_pos"], dtype=np.float32)
        camera_ori = np.array(data["camera_ori"], dtype=np.float32)
        camera_s = np.array(data["camera_s"], dtype=np.float32)
        return camera_pos, camera_ori, camera_s
    raise ValueError(f"Unsupported camera pose file: {camera_pose_path}")


def _select_imu_id(imu_ids: np.ndarray, preferred_id: Optional[int]) -> Optional[int]:
    if imu_ids is None or len(imu_ids) == 0:
        return None
    unique_ids = np.unique(imu_ids)
    if preferred_id is not None and preferred_id in unique_ids:
        return int(preferred_id)
    return int(np.min(unique_ids))


def _sorted_unique_imu_ids(imu_ids: np.ndarray) -> List[int]:
    if imu_ids is None or len(imu_ids) == 0:
        return []
    imu_ids = np.asarray(imu_ids)
    imu_ids = imu_ids[np.isfinite(imu_ids)]
    imu_ids = imu_ids.astype(np.int32, copy=False)
    return sorted(set(int(x) for x in imu_ids.tolist()))


def _resample_encoders_per_imu(
    sensor_ts: np.ndarray,
    data_type: np.ndarray,
    imu_ids: np.ndarray,
    enc1: np.ndarray,
    enc2: np.ndarray,
    target_ts: np.ndarray,
    ordered_imu_ids: List[int],
) -> np.ndarray:
    """
    Returns shape (T, 2 * num_imus) where each imu contributes [enc1, enc2]
    resampled to target_ts.
    """
    if len(ordered_imu_ids) == 0:
        return np.zeros((len(target_ts), 0), dtype=np.float32)

    imu_mask = data_type == 0
    if not np.any(imu_mask):
        return np.zeros((len(target_ts), 2 * len(ordered_imu_ids)), dtype=np.float32)

    imu_ts = sensor_ts[imu_mask]
    imu_ids_raw = imu_ids[imu_mask]
    e1 = enc1[imu_mask]
    e2 = enc2[imu_mask]

    result = []
    for imu_id in ordered_imu_ids:
        this_mask = imu_ids_raw == imu_id
        if not np.any(this_mask):
            result.append(np.zeros((len(target_ts), 2), dtype=np.float32))
            continue
        this_ts = imu_ts[this_mask]
        this_e1 = e1[this_mask]
        this_e2 = e2[this_mask]
        this_enc = np.stack([this_e1, this_e2], axis=-1).astype(np.float32)
        resampled = _resample_nearest(this_ts, this_enc, target_ts)
        if resampled is None:
            resampled = np.zeros((len(target_ts), 2), dtype=np.float32)
        result.append(resampled.astype(np.float32))

    return np.concatenate(result, axis=-1)


def _resample_rpy_per_imu(
    sensor_ts: np.ndarray,
    data_type: np.ndarray,
    imu_ids: np.ndarray,
    roll: np.ndarray,
    pitch: np.ndarray,
    yaw: np.ndarray,
    target_ts: np.ndarray,
    ordered_imu_ids: List[int],
) -> np.ndarray:
    """
    Returns shape (T, num_imus, 3) for roll/pitch/yaw per imu_id.
    """
    if len(ordered_imu_ids) == 0:
        return np.zeros((len(target_ts), 0, 3), dtype=np.float32)

    imu_mask = data_type == 0
    if not np.any(imu_mask):
        return np.zeros((len(target_ts), len(ordered_imu_ids), 3), dtype=np.float32)

    imu_ts = sensor_ts[imu_mask]
    imu_ids_raw = imu_ids[imu_mask]
    r = roll[imu_mask]
    p = pitch[imu_mask]
    y = yaw[imu_mask]

    result = []
    for imu_id in ordered_imu_ids:
        this_mask = imu_ids_raw == imu_id
        if not np.any(this_mask):
            result.append(np.zeros((len(target_ts), 3), dtype=np.float32))
            continue
        this_ts = imu_ts[this_mask]
        this_r = r[this_mask]
        this_p = p[this_mask]
        this_y = y[this_mask]
        this_rpy = np.stack([this_r, this_p, this_y], axis=-1).astype(np.float32)
        resampled = _resample_nearest(this_ts, this_rpy, target_ts)
        if resampled is None:
            resampled = np.zeros((len(target_ts), 3), dtype=np.float32)
        result.append(resampled.astype(np.float32))

    return np.stack(result, axis=1)


def _ensure_filled(data: Optional[np.ndarray], fallback_shape: Tuple[int, int]):
    if data is None:
        return np.full(fallback_shape, 0.0, dtype=np.float32)
    if np.isnan(data).all():
        return np.full(fallback_shape, 0.0, dtype=np.float32)
    return data.astype(np.float32)


def _find_sessions_for_subsystem(base_dir: str) -> List[str]:
    if not os.path.exists(base_dir):
        return []
    sessions = set()
    for item in os.listdir(base_dir):
        p = os.path.join(base_dir, item)
        if os.path.isdir(p):
            f = os.path.join(p, f"{item}_sensors.npz")
            if os.path.exists(f):
                sessions.add(item)
        elif item.endswith("_sensors.npz"):
            sessions.add(item.replace("_sensors.npz", ""))
    return sorted(sessions)


def _find_session_ids(log_root: str) -> List[str]:
    sessions = set()
    for subsystem in ["baseStation", "controller2"]:
        sessions.update(_find_sessions_for_subsystem(os.path.join(log_root, subsystem)))
    return sorted(sessions)


def _find_sensors_file(log_root: str, subsystem: str, session_id: str) -> Optional[str]:
    base_dir = os.path.join(log_root, subsystem)
    f1 = os.path.join(base_dir, session_id, f"{session_id}_sensors.npz")
    if os.path.exists(f1):
        return f1
    f2 = os.path.join(base_dir, f"{session_id}_sensors.npz")
    if os.path.exists(f2):
        return f2
    return None


def _load_filter_list(log_root: str) -> Optional[set]:
    filter_path = os.path.join(log_root, "filter.txt")
    if not os.path.exists(filter_path):
        return None
    keep = set()
    with open(filter_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            keep.add(line)
    return keep


def generate_replay_buffer_from_logs(
    log_root: str,
    output_path: str,
    compression_level: int = 99,
    out_res: Tuple[int, int] = (224, 224),
    imu_id: Optional[int] = None,
    camera_pose_path: Optional[str] = None,
    bgr_to_rgb: bool = True,
    target_fps: Optional[float] = None,
    num_workers: Optional[int] = None,
    action_repr: str = "encoder",
    image_brightness_gain: float = 1.0,
    action_smooth_window: int = 1,
    remove_noop_frames: bool = True,
    noop_eps_base: float = 1e-4,
    noop_eps_joint: float = 1e-4,
    save_processed_viz: bool = False,
    processed_viz_dir: Optional[str] = None,
    processed_viz_max_episodes: int = 20,
    processed_viz_max_cameras: int = 8,
    processed_viz_image_steps: int = 2,
    processed_viz_max_points: int = 256,
    save_imu_in_dataset: bool = True,
    expected_num_cameras_arg: Optional[int] = None,
    num_segments_arg: Optional[int] = None,
):
    if output_path.endswith(".zip") and os.path.exists(output_path):
        os.remove(output_path)
    out_replay_buffer = ReplayBuffer.create_empty_zarr(storage=zarr.MemoryStore())
    img_compressor = JpegXl(level=compression_level, numthreads=1)

    session_ids = _find_session_ids(log_root)
    keep_set = _load_filter_list(log_root)
    if keep_set is not None:
        session_ids = [s for s in session_ids if s in keep_set]
        print(f"Using filter list: {len(session_ids)} sessions kept.")
    print(f"Sessions to process: {len(session_ids)}")
    if len(session_ids) == 0:
        raise FileNotFoundError(f"No *_sensors.npz found under {log_root}")

    camera_root = os.path.join(log_root, "camera")
    expected_num_cameras = (
        int(expected_num_cameras_arg) if expected_num_cameras_arg is not None else None
    )
    if expected_num_cameras is not None and expected_num_cameras < 1:
        raise ValueError("expected_num_cameras must be >= 1 when set")
    expected_imu_ids: Optional[List[int]] = None
    inferred_action_dim: Optional[int] = None
    resize_tf = None

    if num_workers is None:
        num_workers = max(1, multiprocessing.cpu_count() - 1)
    viz_saved = 0

    def report_session_skip(sid: str, reason: str) -> None:
        msg = f"[SKIP] {sid}: {reason}"
        print(msg, flush=True)
        logger.warning("%s", msg)

    for session_id in tqdm(session_ids, desc="sessions"):
        base_sensors_path = _find_sensors_file(log_root, "baseStation", session_id)
        imu_sensors_path = _find_sensors_file(log_root, "controller2", session_id)
        if base_sensors_path is None:
            report_session_skip(session_id, "missing baseStation sensors")
            continue
        if imu_sensors_path is None:
            report_session_skip(session_id, "missing controller2 IMU sensors")
            continue

        sensors_base = np.load(base_sensors_path, allow_pickle=True)
        sensors_imu = np.load(imu_sensors_path, allow_pickle=True)

        sensor_ts_base = sensors_base["timestamp"].astype(np.float64)
        data_type_base = sensors_base["data_type"].astype(np.int64)
        sensor_ts_imu = sensors_imu["timestamp"].astype(np.float64)
        data_type_imu = sensors_imu["data_type"].astype(np.int64)

        video_data = _load_camera_data_video_from_dir(camera_root, session_id)
        cameras_npz_path = None
        for p in [base_sensors_path, imu_sensors_path]:
            if p is None:
                continue
            cand = p.replace("_sensors.npz", "_cameras.npz")
            if os.path.exists(cand):
                cameras_npz_path = cand
                break

        if video_data is None and cameras_npz_path is None:
            video_data = _load_camera_data_avi(camera_root, session_id)

        if video_data is None and cameras_npz_path is None:
            report_session_skip(session_id, "no camera video or cameras.npz")
            sensors_base.close()
            sensors_imu.close()
            continue

        if video_data is not None:
            cam_ts, resolution, _fps, paths_by_id = video_data
            if not paths_by_id:
                report_session_skip(session_id, "no camera video paths")
                sensors_base.close()
                sensors_imu.close()
                continue
            if expected_num_cameras is None:
                expected_num_cameras = max(paths_by_id.keys()) + 1
            # Only decode camera indices [0, expected_num_cameras). Sessions with extra
            # streams are not skipped (extra IDs are ignored), matching npz truncation.
            if resize_tf is None:
                resize_tf = get_image_transform(
                    (resolution[1], resolution[0]), out_res, bgr_to_rgb=bgr_to_rgb
                )
            cam_ts = _align_timestamps(cam_ts, sensor_ts_base)
            cam_ts, keep_idx = _subsample_by_timestamp(cam_ts, target_fps)
            n_ts = len(cam_ts)
            frames_per_cam = [None] * expected_num_cameras
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = []
                for cam_idx in range(expected_num_cameras):
                    path = paths_by_id.get(cam_idx)
                    if path is None or not os.path.isfile(path):
                        continue
                    futures.append(
                        executor.submit(
                            _decode_and_resize_camera,
                            cam_idx,
                            path,
                            keep_idx,
                            n_ts,
                            resize_tf,
                            out_res,
                            resolution[1],
                            resolution[0],
                            image_brightness_gain,
                        )
                    )
                for fut in concurrent.futures.as_completed(futures):
                    cam_idx, resized = fut.result()
                    frames_per_cam[cam_idx] = resized
            valid_lengths = [
                len(frames) for frames in frames_per_cam if frames is not None and len(frames) > 0
            ]
            if not valid_lengths:
                report_session_skip(session_id, "no valid camera frames after decode/resize")
                sensors_base.close()
                sensors_imu.close()
                continue
            num_frames = min(len(cam_ts), min(valid_lengths))
            if num_frames == 0:
                report_session_skip(session_id, "zero frames after timestamp/camera alignment")
                sensors_base.close()
                sensors_imu.close()
                continue
            cam_ts = cam_ts[:num_frames]
            frames_per_cam = _finalize_camera_frames_black_pad(
                frames_per_cam, num_frames, out_res, session_id
            )
            num_cameras = expected_num_cameras
        else:
            cameras_data = np.load(cameras_npz_path, allow_pickle=True)
            if "images" not in cameras_data:
                report_session_skip(session_id, "cameras.npz missing 'images' key")
                cameras_data.close()
                sensors_base.close()
                sensors_imu.close()
                continue
            images = cameras_data["images"]
            cameras_data.close()
            cam_ts = sensor_ts_base.copy()
            n_from_npz = int(images.shape[1])
            if expected_num_cameras is None:
                expected_num_cameras = n_from_npz
            elif n_from_npz > expected_num_cameras:
                images = images[:, :expected_num_cameras, ...]
                n_from_npz = expected_num_cameras
            if resize_tf is None:
                resize_tf = get_image_transform(
                    (images.shape[3], images.shape[2]), out_res, bgr_to_rgb=bgr_to_rgb
                )
            cam_ts, keep_idx = _subsample_by_timestamp(cam_ts, target_fps)
            images = images[keep_idx]
            frames_per_cam = []
            for cam_idx in range(n_from_npz):
                frames = images[:, cam_idx]
                resized_frames = []
                for frame in frames:
                    frame = _ensure_color_frame(frame)
                    if frame is None:
                        continue
                    out = resize_tf(frame)
                    out = _apply_brightness_gain(out, image_brightness_gain)
                    resized_frames.append(out)
                frames_per_cam.append(
                    np.stack(resized_frames, axis=0) if resized_frames else None
                )
            while len(frames_per_cam) < expected_num_cameras:
                frames_per_cam.append(None)
            valid_lengths = [
                len(frames) for frames in frames_per_cam if frames is not None and len(frames) > 0
            ]
            if not valid_lengths:
                report_session_skip(session_id, "no valid camera frames after decode/resize")
                sensors_base.close()
                sensors_imu.close()
                continue
            num_frames = min(len(cam_ts), min(valid_lengths))
            if num_frames == 0:
                report_session_skip(session_id, "zero frames after timestamp/camera alignment")
                sensors_base.close()
                sensors_imu.close()
                continue
            cam_ts = cam_ts[:num_frames]
            frames_per_cam = [
                frames[:num_frames] if frames is not None else None for frames in frames_per_cam
            ]
            frames_per_cam = _finalize_camera_frames_black_pad(
                frames_per_cam, num_frames, out_res, session_id
            )
            num_cameras = expected_num_cameras

        camera_pos, camera_ori, camera_s = _load_camera_pose_template(
            num_cameras, camera_pose_path=camera_pose_path
        )
        camera_pos = np.repeat(camera_pos[None, ...], num_frames, axis=0)
        camera_ori = np.repeat(camera_ori[None, ...], num_frames, axis=0)
        camera_s = np.repeat(camera_s[None, ...], num_frames, axis=0)

        imu_mask = data_type_imu == 0
        imu_ids_all = sensors_imu["imu_id"][imu_mask] if "imu_id" in sensors_imu else None
        session_imu_ids = _sorted_unique_imu_ids(imu_ids_all)
        if save_imu_in_dataset:
            if expected_imu_ids is None or len(expected_imu_ids) == 0:
                if session_imu_ids:
                    expected_imu_ids = session_imu_ids
                else:
                    report_session_skip(session_id, "empty imu_id list (teleop expected)")
                    sensors_base.close()
                    sensors_imu.close()
                    continue
            if expected_imu_ids and session_imu_ids != expected_imu_ids:
                report_session_skip(
                    session_id,
                    f"imu_id mismatch expected={expected_imu_ids} got={session_imu_ids}",
                )
                sensors_base.close()
                sensors_imu.close()
                continue
            ordered_imu_ids = list(expected_imu_ids)
        else:
            if not session_imu_ids:
                report_session_skip(session_id, "empty imu_id list (teleop expected)")
                sensors_base.close()
                sensors_imu.close()
                continue
            ordered_imu_ids = list(session_imu_ids)

        selected_imu_id = _select_imu_id(imu_ids_all, imu_id)
        imu_sel_mask = imu_mask
        if selected_imu_id is not None and "imu_id" in sensors_imu:
            imu_sel_mask = imu_mask & (sensors_imu["imu_id"] == selected_imu_id)

        imu_ts = sensor_ts_imu[imu_sel_mask]
        imu_rpy_single = None
        enc_single = None
        if "roll" in sensors_imu and "pitch" in sensors_imu and "yaw" in sensors_imu:
            imu_rpy_single = np.stack(
                [
                    sensors_imu["roll"][imu_sel_mask],
                    sensors_imu["pitch"][imu_sel_mask],
                    sensors_imu["yaw"][imu_sel_mask],
                ],
                axis=-1,
            ).astype(np.float32)
        if "encoder1" in sensors_imu and "encoder2" in sensors_imu:
            enc_single = np.stack(
                [
                    sensors_imu["encoder1"][imu_sel_mask],
                    sensors_imu["encoder2"][imu_sel_mask],
                ],
                axis=-1,
            ).astype(np.float32)

        pressure_mask = data_type_base == 1
        pressure_ts = sensor_ts_base[pressure_mask]
        pressure = sensors_base["pressure"][pressure_mask] if "pressure" in sensors_base else None

        motor_mask = data_type_base == 2
        motor_ts = sensor_ts_base[motor_mask]
        motor_velocity = (
            sensors_base["motor_velocity"][motor_mask] if "motor_velocity" in sensors_base else None
        )
        motor_pos = (
            sensors_base["motor_pos_data"][motor_mask] if "motor_pos_data" in sensors_base else None
        )
        motor_torque = sensors_base["torque_data"][motor_mask] if "torque_data" in sensors_base else None

        imu_rpy_single = _resample_nearest(imu_ts, imu_rpy_single, cam_ts)
        enc_single = _resample_nearest(imu_ts, enc_single, cam_ts)
        pressure = _resample_nearest(pressure_ts, _stack_values(pressure), cam_ts)
        motor_velocity = _resample_nearest(
            motor_ts, _stack_values(motor_velocity), cam_ts
        )
        motor_pos = _resample_nearest(motor_ts, _stack_values(motor_pos), cam_ts)
        motor_torque = _resample_nearest(motor_ts, _stack_values(motor_torque), cam_ts)

        if motor_velocity is None:
            motor_velocity = np.full((num_frames, 1), 0.0, dtype=np.float32)
        if motor_pos is None:
            motor_pos = np.full((num_frames, 1), 0.0, dtype=np.float32)

        imu_rpy_single = _ensure_filled(imu_rpy_single, (num_frames, 3))
        enc_single = _ensure_filled(enc_single, (num_frames, 2))
        pressure = _ensure_filled(pressure, (num_frames, 1))
        motor_torque = _ensure_filled(motor_torque, (num_frames, 1))

        if "imu_id" in sensors_imu and "encoder1" in sensors_imu and "encoder2" in sensors_imu:
            enc_all = _resample_encoders_per_imu(
                sensor_ts=sensor_ts_imu,
                data_type=data_type_imu,
                imu_ids=sensors_imu["imu_id"],
                enc1=sensors_imu["encoder1"],
                enc2=sensors_imu["encoder2"],
                target_ts=cam_ts,
                ordered_imu_ids=ordered_imu_ids,
            )
        else:
            logger.warning("Missing imu encoder data in %s; filling zeros.", session_id)
            enc_all = np.zeros((num_frames, 2 * len(ordered_imu_ids)), dtype=np.float32)
        if enc_all.shape[1] == 0:
            report_session_skip(
                session_id,
                "empty encoder action vector (base+joint encoders)",
            )
            sensors_base.close()
            sensors_imu.close()
            continue
        if "roll" in sensors_imu and "pitch" in sensors_imu and "yaw" in sensors_imu and "imu_id" in sensors_imu:
            imu_rpy_all = _resample_rpy_per_imu(
                sensor_ts=sensor_ts_imu,
                data_type=data_type_imu,
                imu_ids=sensors_imu["imu_id"],
                roll=sensors_imu["roll"],
                pitch=sensors_imu["pitch"],
                yaw=sensors_imu["yaw"],
                target_ts=cam_ts,
                ordered_imu_ids=ordered_imu_ids,
            )
        else:
            imu_rpy_all = np.zeros((num_frames, len(ordered_imu_ids), 3), dtype=np.float32)

        enc_all = enc_all.reshape(num_frames, -1, 2)
        base_length = encoders_to_length(motor_pos).astype(np.float32)
        enc_flat = enc_all.reshape(num_frames, -1)
        joint_angles = np.stack(
            [enc_list_to_joint_angles(row) for row in enc_flat], axis=0
        ).astype(np.float32)

        # Remove no-op frames where neither base nor joints changed.
        if remove_noop_frames:
            keep_mask = _compute_motion_keep_mask(
                base_length=base_length,
                joint_angles=joint_angles,
                eps_base=noop_eps_base,
                eps_joint=noop_eps_joint,
            )
            if not np.all(keep_mask):
                cam_ts = cam_ts[keep_mask]
                base_length = base_length[keep_mask]
                joint_angles = joint_angles[keep_mask]
                enc_all = enc_all[keep_mask]
                imu_rpy_all = imu_rpy_all[keep_mask]
                pressure = pressure[keep_mask]
                motor_torque = motor_torque[keep_mask]
                camera_pos = camera_pos[keep_mask]
                camera_ori = camera_ori[keep_mask]
                camera_s = camera_s[keep_mask]
                frames_per_cam = [
                    (frames[keep_mask] if frames is not None else None)
                    for frames in frames_per_cam
                ]
                num_frames = int(np.sum(keep_mask))
                if num_frames <= 0:
                    report_session_skip(
                        session_id,
                        "all frames removed as no-op (noop filtering)",
                    )
                    sensors_base.close()
                    sensors_imu.close()
                    continue

        # Smooth base/joint trajectories before writing actions.
        base_length = _moving_average_2d(base_length, action_smooth_window).astype(np.float32)
        joint_angles = _moving_average_2d(joint_angles, action_smooth_window).astype(np.float32)

        if num_segments_arg is not None:
            ns = int(num_segments_arg)
            if ns < 1:
                raise ValueError("num_segments from task config must be >= 1 when set")
            joint_angles = joint_angles[:, :ns].astype(np.float32)

        if action_repr == "angle":
            action = np.concatenate([base_length, joint_angles], axis=-1)
        else:
            enc_flat_full = enc_all.reshape(num_frames, -1)
            if num_segments_arg is not None:
                n_pairs = int(num_segments_arg)
                enc_flat_full = enc_flat_full[:, : 2 * n_pairs]
            action = np.concatenate([base_length, enc_flat_full], axis=-1)
        if inferred_action_dim is None:
            inferred_action_dim = action.shape[1]

        episode_data = {
            "action": action.astype(np.float32),
            "joint_angle": joint_angles.astype(np.float32),
            "base_encoder": base_length.astype(np.float32),
            "pressure": pressure,
            "motor_torque": motor_torque,
            "camera_pos": camera_pos.astype(np.float32),
            "camera_ori": camera_ori.astype(np.float32),
            "camera_s": camera_s.astype(np.float32),
        }
        if save_imu_in_dataset:
            episode_data["imu_rpy"] = imu_rpy_all.astype(np.float32)
            episode_data["encoder"] = enc_all.astype(np.float32)

        compressors = {}
        chunks = {}
        for cam_idx in range(num_cameras):
            frames = frames_per_cam[cam_idx]
            if frames is None:
                continue
            resized = frames[:num_frames]
            key = f"camera{cam_idx}_rgb"
            episode_data[key] = resized.astype(np.uint8)
            compressors[key] = img_compressor
            chunks[key] = (1,) + out_res + (3,)

        # Validate shapes against existing buffer (skip mismatches)
        shape_mismatch = False
        mismatches = []
        for key, value in episode_data.items():
            if key in out_replay_buffer.data:
                if value.shape[1:] != out_replay_buffer.data[key].shape[1:]:
                    mismatches.append(
                        (
                            key,
                            out_replay_buffer.data[key].shape[1:],
                            value.shape[1:],
                        )
                    )
                    shape_mismatch = True
        if shape_mismatch:
            report_session_skip(
                session_id,
                "shape mismatch with existing replay buffer (see log for keys)",
            )
            if mismatches:
                for key, expected_shape, got_shape in mismatches:
                    logger.warning(
                        "Skipping %s due to shape mismatch for %s. expected=%s got=%s",
                        session_id,
                        key,
                        expected_shape,
                        got_shape,
                    )
            sensors_base.close()
            sensors_imu.close()
            continue

        if save_processed_viz and viz_saved < int(processed_viz_max_episodes):
            viz_dir = processed_viz_dir or os.path.join(os.path.dirname(output_path), "processed_viz")
            _save_processed_episode_viz(
                episode_data=episode_data,
                session_id=session_id,
                out_dir=viz_dir,
                max_cameras=processed_viz_max_cameras,
                image_steps=processed_viz_image_steps,
                max_plot_points=processed_viz_max_points,
                bgr_to_rgb=bgr_to_rgb,
            )
            viz_saved += 1

        out_replay_buffer.add_episode(
            episode_data, chunks=chunks, compressors=compressors
        )
        sensors_base.close()
        sensors_imu.close()

    print(f"Saving ReplayBuffer to {output_path}")
    if output_path.endswith(".zip"):
        with zarr.ZipStore(output_path, mode="w") as zip_store:
            out_replay_buffer.save_to_store(store=zip_store)
    else:
        out_replay_buffer.save_to_path(output_path)
    # Summary: number of episodes and dataset shapes
    try:
        rb = zarr.open(output_path, mode="r")
        num_episodes = rb["meta"]["episode_ends"].shape[0]
        print(f"Saved episodes: {num_episodes}")
        if inferred_action_dim is not None:
            print(
                f"Inferred action dim (base + joints): {inferred_action_dim}"
            )
        print("Dataset entries:")
        for key in sorted(rb["data"].keys()):
            print(f"  {key}: {rb['data'][key].shape}")
    except Exception as e:
        print(f"[WARN] Could not read summary from {output_path}: {e}")
    print("Done!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate replay buffer from logs.")
    parser.add_argument("--log-root", default="robot_logs")
    parser.add_argument("--output-path", default="dataset_vine.zarr.zip")
    parser.add_argument("--out-res", nargs=2, type=int, default=[224, 224])
    parser.add_argument("--imu-id", type=int, default=None)
    parser.add_argument(
        "--task-config",
        default=None,
        help=(
            "Optional task YAML (e.g. diffusion_policy/config/task/free_space_vine.yaml). "
            "Reads top-level save_imu_in_dataset, expected_num_cameras, and num_segments when set."
        ),
    )
    parser.add_argument(
        "--no-save-imu-in-dataset",
        action="store_true",
        help=(
            "Omit imu_rpy and encoder from the zarr; use each session's IMU layout only to build actions "
            "(no cross-session IMU-id / encoder-width lock)."
        ),
    )
    parser.add_argument(
        "--expected-num-cameras",
        type=int,
        default=None,
        help=(
            "Fix dataset camera count (e.g. 4 or 19). Only camera indices [0, N) are stored; "
            "missing streams are black-padded; extra streams in logs are ignored. Overrides task YAML."
        ),
    )
    parser.add_argument(
        "--num-segments",
        type=int,
        default=None,
        help=(
            "Export action as base length + this many joint angles (and matching encoder width "
            "when --action-repr encoder). Overrides task YAML when set."
        ),
    )
    parser.add_argument("--camera-pose-path", default=None)
    parser.add_argument("--target-fps", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument(
        "--action-repr",
        choices=["encoder", "angle"],
        default="angle",
        help="Save actions as base length (m) + raw encoders or joint angles.",
    )
    parser.add_argument(
        "--image-brightness-gain",
        type=float,
        default=3.0,
        help="Multiply camera pixel intensities before saving (e.g. 1.1, 1.2).",
    )
    parser.add_argument(
        "--action-smooth-window",
        type=int,
        default=1,
        help="Moving-average window for base/joint action smoothing. 1 disables.",
    )
    parser.add_argument(
        "--keep-noop-frames",
        action="store_true",
        help="Keep frames with no base/joint change (disables no-op filtering).",
    )
    parser.add_argument(
        "--noop-eps-base",
        type=float,
        default=1e-4,
        help="Motion threshold for base length when removing no-op frames.",
    )
    parser.add_argument(
        "--noop-eps-joint",
        type=float,
        default=1e-4,
        help="Motion threshold for joint angles when removing no-op frames.",
    )
    parser.add_argument(
        "--save-processed-viz",
        action="store_true",
        help="Save processed per-episode visualization during dataset generation.",
    )
    parser.add_argument(
        "--processed-viz-dir",
        type=str,
        default=None,
        help="Directory for processed-data visualizations.",
    )
    parser.add_argument(
        "--processed-viz-max-episodes",
        type=int,
        default=20,
        help="Max number of episodes to visualize during generation.",
    )
    parser.add_argument(
        "--processed-viz-max-cameras",
        type=int,
        default=8,
        help="Max number of camera streams to draw per episode viz.",
    )
    parser.add_argument(
        "--processed-viz-image-steps",
        type=int,
        default=2,
        help="Number of timesteps to show in processed RGB visualization.",
    )
    parser.add_argument(
        "--processed-viz-max-points",
        type=int,
        default=256,
        help="Max time points per signal plot (uniform subsample).",
    )
    parser.add_argument(
        "--no-bgr-to-rgb",
        action="store_true",
        help=(
            "Keep OpenCV BGR channel order in the zarr (default: convert to RGB, "
            "matching policy_inference live cameras)."
        ),
    )
    args = parser.parse_args()

    save_imu_in_dataset = True
    expected_num_cameras_arg = None
    num_segments_arg = None
    if args.task_config:
        try:
            import yaml
        except ImportError as e:
            raise RuntimeError(
                "PyYAML is required for --task-config (pip install pyyaml)."
            ) from e
        with open(os.path.abspath(os.path.expanduser(args.task_config)), "r") as f:
            task_cfg = yaml.safe_load(f) or {}
        if "save_imu_in_dataset" in task_cfg:
            save_imu_in_dataset = bool(task_cfg["save_imu_in_dataset"])
        if task_cfg.get("expected_num_cameras") is not None:
            expected_num_cameras_arg = int(task_cfg["expected_num_cameras"])
        if task_cfg.get("num_segments") is not None:
            num_segments_arg = int(task_cfg["num_segments"])
    if args.no_save_imu_in_dataset:
        save_imu_in_dataset = False
    if args.expected_num_cameras is not None:
        expected_num_cameras_arg = int(args.expected_num_cameras)
    if args.num_segments is not None:
        num_segments_arg = int(args.num_segments)

    print(
        f"[INFO] bgr_to_rgb={not bool(args.no_bgr_to_rgb)} "
        "(dataset camera_*_rgb tensors; match policy_inference unless using --no-bgr-to-rgb on both)",
        flush=True,
    )
    print(f"[INFO] save_imu_in_dataset={save_imu_in_dataset}", flush=True)
    if expected_num_cameras_arg is not None:
        print(f"[INFO] expected_num_cameras={expected_num_cameras_arg} (pad missing with black)", flush=True)
    if num_segments_arg is not None:
        print(
            f"[INFO] num_segments={num_segments_arg} "
            "(zarr action width = 1 + num_segments for angle repr)",
            flush=True,
        )
    if not save_imu_in_dataset:
        print(
            "[INFO] IMU tensors omitted from zarr (imu_rpy, encoder); per-session IMU layout is used "
            "only to compute actions and proprio; mixed IMU counts across sessions are allowed.",
            flush=True,
        )

    generate_replay_buffer_from_logs(
        log_root=args.log_root,
        output_path=args.output_path,
        out_res=(args.out_res[0], args.out_res[1]),
        imu_id=args.imu_id,
        camera_pose_path=args.camera_pose_path,
        bgr_to_rgb=not args.no_bgr_to_rgb,
        target_fps=args.target_fps,
        num_workers=args.num_workers,
        action_repr=args.action_repr,
        image_brightness_gain=args.image_brightness_gain,
        action_smooth_window=args.action_smooth_window,
        remove_noop_frames=not args.keep_noop_frames,
        noop_eps_base=args.noop_eps_base,
        noop_eps_joint=args.noop_eps_joint,
        save_processed_viz=args.save_processed_viz,
        processed_viz_dir=args.processed_viz_dir,
        processed_viz_max_episodes=args.processed_viz_max_episodes,
        processed_viz_max_cameras=args.processed_viz_max_cameras,
        processed_viz_image_steps=args.processed_viz_image_steps,
        processed_viz_max_points=args.processed_viz_max_points,
        save_imu_in_dataset=save_imu_in_dataset,
        expected_num_cameras_arg=expected_num_cameras_arg,
        num_segments_arg=num_segments_arg,
    )