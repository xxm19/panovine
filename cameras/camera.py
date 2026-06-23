#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
os.environ["NO_AT_BRIDGE"] = "1"
os.environ["OPENCV_VIDEOIO_PRIORITY_V4L2"] = "1"

import time
from typing import Optional, Protocol, Tuple, List, Dict

import cv2
import numpy as np
from multiprocessing import Process, Event, Queue
from multiprocessing.managers import SharedMemoryManager


class CameraDriver(Protocol):
    def read(
        self,
        img_size: Optional[Tuple[int, int]] = None,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        ...


class USBCamera(CameraDriver):
    """
    Fork-safe USB camera wrapper.

    IMPORTANT: does NOT open cv2.VideoCapture in __init__.
    It opens lazily in read() (inside the worker process).
    """

    def __init__(
        self,
        device_id: str,
        flip: bool = False,
        fps: int = 15,
        resolution: Tuple[int, int] = (480, 640),  # (H, W)
        warmup_reads: int = 20,
        buffer_size: int = 1,
        mjpg: bool = True,
        flush_grabs: int = 30,
        flush_delay_s: float = 0.05,
    ):
        self._device_id = device_id.strip()
        self._flip = flip

        self.req_h, self.req_w = resolution
        self.req_fps = int(max(1, fps))

        self.warmup_reads = int(max(0, warmup_reads))
        self.buffer_size = int(max(1, buffer_size))
        self.use_mjpg = bool(mjpg)

        self.flush_grabs = int(max(0, flush_grabs))
        self.flush_delay_s = float(max(0.0, flush_delay_s))

        self._cap: Optional[cv2.VideoCapture] = None
        self.fail_count = 0

        self.MAX_SOFT_FAILS = 20
        self.MAX_REOPEN_TRIES = 5

    @property
    def device_id(self) -> str:
        return self._device_id

    @property
    def fps(self) -> int:
        return self.req_fps

    @property
    def resolution(self) -> Tuple[int, int]:
        return (self.req_h, self.req_w)

    def _open_cap(self) -> None:
        self.close()

        cap = cv2.VideoCapture(self._device_id, cv2.CAP_V4L2)
        if not cap.isOpened():
            self._cap = None
            raise RuntimeError(f"Failed to open camera device: {self._device_id}")

        if self.use_mjpg:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.req_w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.req_h)
        cap.set(cv2.CAP_PROP_FPS, self.req_fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)

        # Best-effort match old code
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)

        for _ in range(self.flush_grabs):
            cap.grab()
        if self.flush_delay_s > 0:
            time.sleep(self.flush_delay_s)

        ok = False
        for _ in range(self.warmup_reads):
            ret, frame = cap.read()
            if ret and frame is not None:
                ok = True
                break
            time.sleep(0.03)

        if not ok:
            cap.release()
            self._cap = None
            raise RuntimeError(f"Camera opened but no frames during warmup: {self._device_id}")

        self._cap = cap
        self.fail_count = 0

    def _postprocess(self, frame: np.ndarray, img_size: Optional[Tuple[int, int]]) -> np.ndarray:
        if img_size is not None:
            H, W = img_size
            if frame.shape[:2] != (H, W):
                frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)
        else:
            if frame.shape[:2] != (self.req_h, self.req_w):
                frame = cv2.resize(frame, (self.req_w, self.req_h), interpolation=cv2.INTER_AREA)

        if self._flip:
            frame = cv2.rotate(frame, cv2.ROTATE_180)

        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)

        return frame

    def _handle_fail(self) -> Tuple[None, None]:
        self.fail_count += 1

        if self.fail_count >= self.MAX_SOFT_FAILS:
            for _ in range(self.MAX_REOPEN_TRIES):
                try:
                    self._open_cap()
                    ret, frame = self._cap.read()
                    if ret and frame is not None:
                        self.fail_count = 0
                        frame = self._postprocess(frame, img_size=(self.req_h, self.req_w))
                        return frame, None
                except Exception:
                    pass
                time.sleep(0.1)

        return None, None

    def read(self, img_size: Optional[Tuple[int, int]] = None) -> Tuple[Optional[np.ndarray], None]:
        if self._cap is None:
            try:
                self._open_cap()
            except Exception:
                return None, None

        ret = self._cap.grab()
        if not ret:
            return self._handle_fail()

        ret, frame = self._cap.retrieve()
        if (not ret) or (frame is None):
            return self._handle_fail()

        self.fail_count = 0
        frame = self._postprocess(frame, img_size=img_size)
        return frame, None

    def close(self):
        try:
            if self._cap is not None:
                self._cap.release()
        except Exception:
            pass
        self._cap = None


class camServer:
    """
    Shared-memory camera server (multiprocess).

    If allow_partial_start=True:
      - server starts even if some cameras fail to open
      - failed cameras write black frames and keep retrying periodically
    """

    def __init__(
        self,
        shm_manager: SharedMemoryManager,
        cameras: List[USBCamera],
        top_camera: Optional[USBCamera] = None,
        side_camera: Optional[USBCamera] = None,

        allow_partial_start: bool = True,
        start_timeout_s: float = 25.0,

        open_stagger_s: float = 0.20,         # per-index delay before first open attempt
        open_retry_window_s: float = 8.0,     # how long each worker retries first frame at startup
        reopen_retry_period_s: float = 2.0,   # how often a dead camera retries later

        random_disabling: bool = False,
        random_disabling_prob: float = 0.2,
        random_latency: bool = False,
        random_latency_prob: float = 0.2,
        random_latency_range: Tuple[float, float] = (0.0, 1.0),
    ):
        self.cameras = cameras
        self.top_camera = top_camera
        self.side_camera = side_camera

        self.allow_partial_start = bool(allow_partial_start)

        self.stop_event = Event()
        self._status_q: Queue = Queue()

        self.num_cams = len(cameras)
        self.H, self.W = 480, 640

        self.shm = shm_manager.SharedMemory(size=self.num_cams * self.H * self.W * 3)
        self.shm_array = np.ndarray((self.num_cams, self.H, self.W, 3), dtype=np.uint8, buffer=self.shm.buf)
        self.shm_array[:] = 0

        self.shm_top_array = None
        self.shm_side_array = None
        self._has_top = top_camera is not None
        self._has_side = side_camera is not None

        if self._has_top:
            th, tw = top_camera.resolution
            self.shm_top = shm_manager.SharedMemory(size=th * tw * 3)
            self.shm_top_array = np.ndarray((th, tw, 3), dtype=np.uint8, buffer=self.shm_top.buf)
            self.shm_top_array[:] = 0

        if self._has_side:
            sh, sw = side_camera.resolution
            self.shm_side = shm_manager.SharedMemory(size=sh * sw * 3)
            self.shm_side_array = np.ndarray((sh, sw, 3), dtype=np.uint8, buffer=self.shm_side.buf)
            self.shm_side_array[:] = 0

        self.open_stagger_s = float(max(0.0, open_stagger_s))
        self.open_retry_window_s = float(max(0.5, open_retry_window_s))
        self.reopen_retry_period_s = float(max(0.2, reopen_retry_period_s))

        self.random_disabling = random_disabling
        self.random_disabling_prob = random_disabling_prob
        self.random_latency = random_latency
        self.random_latency_prob = random_latency_prob
        self.random_latency_range = random_latency_range

        # status snapshot for main process
        self._alive: Dict[str, bool] = {}
        self._last_err: Dict[str, str] = {}

        self.processes: List[Process] = []
        for i in range(self.num_cams):
            p = Process(target=self._loop_cam, args=(i,))
            p.daemon = True
            p.start()
            self.processes.append(p)
            time.sleep(0.03)

        if self._has_top:
            p = Process(target=self._loop_top)
            p.daemon = True
            p.start()
            self.processes.append(p)

        if self._has_side:
            p = Process(target=self._loop_side)
            p.daemon = True
            p.start()
            self.processes.append(p)

        # Wait for startup messages (but don't necessarily fail)
        self._collect_startup(timeout_s=start_timeout_s)

    def _collect_startup(self, timeout_s: float):
        expected = self.num_cams + (1 if self._has_top else 0) + (1 if self._has_side else 0)
        seen = 0
        t0 = time.time()

        while seen < expected and (time.time() - t0) < timeout_s:
            try:
                kind, name, msg = self._status_q.get(timeout=0.2)
            except Exception:
                continue

            if kind == "READY":
                self._alive[name] = True
                self._last_err[name] = ""
                seen += 1
            elif kind == "DEAD":
                self._alive[name] = False
                self._last_err[name] = msg or ""
                seen += 1
            elif kind == "WARN":
                # runtime warnings: update last error but don't change alive unless specified
                if name:
                    self._last_err[name] = msg or ""

        if not self.allow_partial_start:
            # strict mode: require all READY at startup
            not_ready = [k for k, v in self._alive.items() if not v]
            if seen < expected or len(not_ready) > 0:
                self.end()
                raise RuntimeError(f"[FATAL] Startup failed. Dead cams: {not_ready}. Errors: {self._last_err}")

    def get_status(self) -> Dict[str, Dict[str, object]]:
        """
        Non-blocking: drain status queue and return latest snapshot.
        """
        while True:
            try:
                kind, name, msg = self._status_q.get_nowait()
            except Exception:
                break

            if kind == "READY":
                self._alive[name] = True
                self._last_err[name] = ""
            elif kind == "DEAD":
                self._alive[name] = False
                self._last_err[name] = msg or ""
            elif kind == "WARN":
                if name:
                    self._last_err[name] = msg or ""

        out = {}
        for k in sorted(self._alive.keys()):
            out[k] = {"alive": bool(self._alive[k]), "last_error": self._last_err.get(k, "")}
        return out

    def _startup_open_with_retries(self, cam: USBCamera, img_size: Tuple[int, int], name: str) -> Optional[np.ndarray]:
        deadline = time.time() + self.open_retry_window_s
        img = None
        last_err = ""
        while (img is None) and (time.time() < deadline) and (not self.stop_event.is_set()):
            try:
                img, _ = cam.read(img_size=img_size)
            except Exception as e:
                last_err = str(e)
                img = None
            if img is None:
                time.sleep(0.2)
        if img is None and last_err:
            self._status_q.put(("WARN", name, last_err))
        return img

    def _loop_cam(self, camera_idx: int):
        H, W = self.H, self.W
        black = np.zeros((H, W, 3), dtype=np.uint8)

        cam = self.cameras[camera_idx]
        name = f"cam{camera_idx:02d}"

        # Stagger OPEN attempts (important for many USB cams)
        time.sleep(self.open_stagger_s * camera_idx)

        # Startup: try to get first frame
        img = self._startup_open_with_retries(cam, img_size=(H, W), name=name)
        if img is None:
            self._status_q.put(("DEAD", name, f"startup no-frame dev={cam.device_id}"))
            alive = False
        else:
            self._status_q.put(("READY", name, ""))
            alive = True
            self.shm_array[camera_idx] = img

        tick_sleep = 1.0 / max(1, cam.fps)
        next_retry = time.time() + self.reopen_retry_period_s

        while not self.stop_event.is_set():
            if self.random_disabling and np.random.rand() < self.random_disabling_prob:
                self.shm_array[camera_idx] = black
                time.sleep(tick_sleep)
                continue

            if alive:
                img, _ = cam.read(img_size=(H, W))
                if img is None:
                    self.shm_array[camera_idx] = black
                    alive = False
                    self._status_q.put(("DEAD", name, f"runtime read fail dev={cam.device_id}"))
                    next_retry = time.time() + self.reopen_retry_period_s
                    time.sleep(0.005)
                else:
                    self.shm_array[camera_idx] = img
                    time.sleep(tick_sleep)
            else:
                # dead: keep black, retry periodically
                self.shm_array[camera_idx] = black
                if time.time() >= next_retry:
                    img2 = self._startup_open_with_retries(cam, img_size=(H, W), name=name)
                    if img2 is not None:
                        self.shm_array[camera_idx] = img2
                        alive = True
                        self._status_q.put(("READY", name, "recovered"))
                        time.sleep(tick_sleep)
                    else:
                        next_retry = time.time() + self.reopen_retry_period_s
                time.sleep(0.02)

            if self.random_latency and np.random.rand() < self.random_latency_prob:
                time.sleep(np.random.uniform(*self.random_latency_range))

        try:
            cam.close()
        except Exception:
            pass

    def _loop_top(self):
        cam = self.top_camera
        th, tw = cam.resolution
        black = np.zeros((th, tw, 3), dtype=np.uint8)
        name = "top"

        img = self._startup_open_with_retries(cam, img_size=(th, tw), name=name)
        if img is None:
            self._status_q.put(("DEAD", name, f"startup no-frame dev={cam.device_id}"))
            alive = False
        else:
            self._status_q.put(("READY", name, ""))
            alive = True
            self.shm_top_array[:] = img

        tick_sleep = 1.0 / max(1, cam.fps)
        next_retry = time.time() + self.reopen_retry_period_s

        while not self.stop_event.is_set():
            if alive:
                img, _ = cam.read(img_size=(th, tw))
                if img is None:
                    self.shm_top_array[:] = black
                    alive = False
                    self._status_q.put(("DEAD", name, f"runtime read fail dev={cam.device_id}"))
                    next_retry = time.time() + self.reopen_retry_period_s
                    time.sleep(0.005)
                else:
                    self.shm_top_array[:] = img
                    time.sleep(tick_sleep)
            else:
                self.shm_top_array[:] = black
                if time.time() >= next_retry:
                    img2 = self._startup_open_with_retries(cam, img_size=(th, tw), name=name)
                    if img2 is not None:
                        self.shm_top_array[:] = img2
                        alive = True
                        self._status_q.put(("READY", name, "recovered"))
                        time.sleep(tick_sleep)
                    else:
                        next_retry = time.time() + self.reopen_retry_period_s
                time.sleep(0.02)

        try:
            cam.close()
        except Exception:
            pass

    def _loop_side(self):
        cam = self.side_camera
        sh, sw = cam.resolution
        black = np.zeros((sh, sw, 3), dtype=np.uint8)
        name = "side"

        img = self._startup_open_with_retries(cam, img_size=(sh, sw), name=name)
        if img is None:
            self._status_q.put(("DEAD", name, f"startup no-frame dev={cam.device_id}"))
            alive = False
        else:
            self._status_q.put(("READY", name, ""))
            alive = True
            self.shm_side_array[:] = img

        tick_sleep = 1.0 / max(1, cam.fps)
        next_retry = time.time() + self.reopen_retry_period_s

        while not self.stop_event.is_set():
            if alive:
                img, _ = cam.read(img_size=(sh, sw))
                if img is None:
                    self.shm_side_array[:] = black
                    alive = False
                    self._status_q.put(("DEAD", name, f"runtime read fail dev={cam.device_id}"))
                    next_retry = time.time() + self.reopen_retry_period_s
                    time.sleep(0.005)
                else:
                    self.shm_side_array[:] = img
                    time.sleep(tick_sleep)
            else:
                self.shm_side_array[:] = black
                if time.time() >= next_retry:
                    img2 = self._startup_open_with_retries(cam, img_size=(sh, sw), name=name)
                    if img2 is not None:
                        self.shm_side_array[:] = img2
                        alive = True
                        self._status_q.put(("READY", name, "recovered"))
                        time.sleep(tick_sleep)
                    else:
                        next_retry = time.time() + self.reopen_retry_period_s
                time.sleep(0.02)

        try:
            cam.close()
        except Exception:
            pass

    def get_data(self):
        top = self.shm_top_array.copy() if self.shm_top_array is not None else None
        side = self.shm_side_array.copy() if self.shm_side_array is not None else None
        return self.shm_array.copy(), top, side

    def end(self):
        self.stop_event.set()
        for p in getattr(self, "processes", []):
            try:
                p.join(timeout=2.0)
            except Exception:
                pass
        for cam in self.cameras:
            try:
                cam.close()
            except Exception:
                pass

    def join(self):
        for p in getattr(self, "processes", []):
            p.join()


# errno=19 (No such device) usually means:
# - camera unplugged / USB reset / device disappeared
# - hub/controller hiccup
# With allow_partial_start=True, server will keep running, and that cam stays black
# and keeps retrying every reopen_retry_period_s.


if __name__ == "__main__":
    device_ids = [
        "/dev/v4l/by-path/pci-0000:00:14.0-usb-0:7:1.0-video-index0",
        "/dev/v4l/by-path/pci-0000:00:14.0-usb-0:8:1.0-video-index0",
    ]

    usbcams = [USBCamera(device_id=idx, fps=15) for idx in device_ids]

    shm_manager = SharedMemoryManager()
    shm_manager.start()

    server = camServer(shm_manager, usbcams, allow_partial_start=True)

    for i in range(50):
        data, _, _ = server.get_data()
        status = server.get_status()
        if i % 10 == 0:
            print("status:", status)
        time.sleep(0.1)

    server.end()
    server.join()
    shm_manager.shutdown()

