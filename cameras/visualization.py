#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Monitor node:
- Subscribes to:
    /motor_encoder  (std_msgs/Float32)
    /encoders_raw   (std_msgs/Int32MultiArray)

- Converts motor_encoder -> length (m) using a lookup table
- Converts selected encoders -> joint angles (rad) via calibration lookup
- Real-time 3D visualization of the growing robot trajectory:

    * Robot has 6 segments (link lengths L1..L6)
    * Active segments depend on current length:
        - segment 1: active if length > 0
        - segment 2: active if length > L1
        - segment 3: active if length > L1+L2
        - ...
        - segment 6: active if length > L1+...+L5

    * Rotation axes per segment:
        1: horizontal (about Z)
        2: vertical   (about Y)
        3: horizontal (about Z)
        4: vertical   (about Y)
        5: horizontal (about Z)
        6: vertical   (about Y)

    * Forward kinematics assumes base at origin and initial direction along +X.
      IMPORTANT: joints are placed at the END of each link:
        - translate along link i
        - then apply joint i rotation, which affects link i+1 and beyond.
      This keeps link 1 as a fixed base; later steering only bends later links.
"""

import rospy
import numpy as np
import threading
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from std_msgs.msg import Float32, Int32MultiArray

# ================================
# Shared state (protected by lock)
# ================================
state_lock = threading.Lock()
latest_motor_encoder = None   # float
latest_motor_length  = None   # float (meters)
latest_encoders_raw  = []     # list[int]

# ================================
# 1) encoder -> length lookup
# ================================

RAW_ENCODER_TABLE = np.array([
    0.0,
    -4.098,
    -4.66,
    -6.90,
    -8.96,
    -10.02,
    -14.31,
    -24.29,
    -38.99,
    -53.03,
    -68.95,
    -82.72,
    -93.47,
    -100.10,
    -106.58,
    -115.09,
    -125.49,
    -130.64,
    -140.22,
    -150.05,
    -160.34,
    -170.39,
    -180.02,
    -190.25,



], dtype=float)

RAW_LENGTH_TABLE = np.array([
    0.0,
    0.25,
    0.30,
    0.40,
    0.50,
    0.55,
    0.74,
    1.10,
    1.49,
    1.79,
    2.10,
    2.37,
    2.57,
    2.70,
    2.83,
    3.00,
    3.17,
    3.20,
    3.40,
    3.58,
    3.74,
    3.91,
    4.05,
    4.23


], dtype=float)


sort_idx_len = np.argsort(RAW_ENCODER_TABLE)
ENCODER_TABLE_LEN = RAW_ENCODER_TABLE[sort_idx_len]
LENGTH_TABLE      = RAW_LENGTH_TABLE[sort_idx_len]

def encoder_to_length(enc_value):
    """Convert motor encoder reading -> robot length in meters."""
    theta_min = ENCODER_TABLE_LEN[0]
    theta_max = ENCODER_TABLE_LEN[-1]

    if enc_value <= theta_min:
        return float(LENGTH_TABLE[0])
    if enc_value >= theta_max:
        return float(LENGTH_TABLE[-1])

    return float(np.interp(enc_value, ENCODER_TABLE_LEN, LENGTH_TABLE))


# ================================
# 2) encoder -> joint angle mapping (lookup-table interpolation)
# ================================

ENCODER_INDEX_PER_SEGMENT = [0, 2, 4, 6, 8, 10]
NUM_SEGMENTS = 7

# Encoder tick → measured bending angle (deg)
RAW_ENCODER_ANGLE_TICKS = np.array([
    -49888, -39916, -29944, -19972,
         0,
     19972,  29944,  39916,  49888
], dtype=float)

RAW_ENCODER_ANGLE_DEG = np.array([
     57,   50,   41,   30,
      0,
    -30,  -41,  -50,  -57
], dtype=float)


# sort to ensure monotonic input for interpolation
_sort_idx = np.argsort(RAW_ENCODER_ANGLE_TICKS)
ENC_TBL = RAW_ENCODER_ANGLE_TICKS[_sort_idx]
ANG_TBL = RAW_ENCODER_ANGLE_DEG[_sort_idx]


def encoder_tick_to_angle_rad(enc_value):
    """
    Convert encoder ticks -> joint angle (radians) using interpolation.
    Values beyond the table range are clipped.
    """
    if enc_value <= ENC_TBL[0]:
        angle_deg = ANG_TBL[0]
    elif enc_value >= ENC_TBL[-1]:
        angle_deg = ANG_TBL[-1]
    else:
        angle_deg = np.interp(enc_value, ENC_TBL, ANG_TBL)

    return np.deg2rad(angle_deg)


def extract_joint_angles_from_encoders(enc_list):
    """Pick encoders [0,2,4,6,8,10] and convert to joint angles."""
    angles = np.zeros(NUM_SEGMENTS, dtype=float)

    for seg_idx, enc_idx in enumerate(ENCODER_INDEX_PER_SEGMENT):
        if enc_idx < len(enc_list):
            angles[seg_idx] = encoder_tick_to_angle_rad(enc_list[enc_idx])
        else:
            angles[seg_idx] = 0.0

    return angles


# ================================
# 3) Growing robot geometry
# ================================
LINK_LENGTHS = np.array([0.37, 0.20, 1.2, 0.20, 1.2, 0.20, 0.88], dtype=float)  # meters

SEG_CUMSUM = np.cumsum(LINK_LENGTHS)

# Prefix lengths: sum of previous segments
# segment 0 active if length > 0
# segment 1 active if length > L1
# segment 2 active if length > L1 + L2
SEG_PREFIX = np.concatenate(([0.0], np.cumsum(LINK_LENGTHS[:-1])))

# Rotation axes per segment (horizontal/vertical pattern):
#   - "horizontal": rotation about Z axis (turning in XY plane)
#   - "vertical"  : rotation about Y axis (pitching in XZ plane)
SEGMENT_AXES = ["horizontal", "vertical",
                "horizontal", "vertical",
                "horizontal", "vertical"]


def compute_segment_effective_lengths(total_length):
    """
    Given the current total grown length (meters),
    compute the effective length of each segment (0..LINK_LENGTHS[i]):

    - Early segments fully grown if there's enough length.
    - The next segment gets the remaining length (partial).
    - Later segments are 0.
    """
    effective = np.zeros(NUM_SEGMENTS, dtype=float)
    remaining = total_length

    for i in range(NUM_SEGMENTS):
        if remaining <= 0.0:
            break
        seg_len = min(LINK_LENGTHS[i], remaining)
        effective[i] = seg_len
        remaining -= seg_len

    return effective


def mask_inactive_joints(total_length, joint_angles):
    """
    Zero out joint angles for segments that should not be active yet.

    Rule (1-based indexing in description):
      - segment 1: active if length > 0
      - segment 2: active if length > L1
      - segment 3: active if length > L1 + L2
      - ...
    Implementation (0-based):
      - segment i active if total_length > SEG_PREFIX[i]
    """
    active = joint_angles.copy()
    for i in range(NUM_SEGMENTS):
        if total_length <= SEG_PREFIX[i]:
            active[i] = 0.0
    return active


def rot_z(theta):
    """Rotation matrix about Z axis by theta (rad)."""
    c = np.cos(-theta)
    s = np.sin(-theta)
    return np.array([
        [ c, -s, 0.0],
        [ s,  c, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=float)


def rot_y(theta):
    """Rotation matrix about Y axis by theta (rad)."""
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([
        [ c, 0.0,  s],
        [0.0, 1.0, 0.0],
        [-s, 0.0,  c]
    ], dtype=float)


def forward_kinematics_3d(segment_lengths, joint_angles):
    """
    Compute 3D points of the growing robot centerline.

    Base at origin, initial direction +X.

    IMPORTANT: Joint i is at the END of link i:
        - Translate along link i (using current orientation)
        - Then apply joint i rotation, which affects link i+1 and beyond.

    This keeps link 1 as a fixed base: later steering only changes later links.
    """
    points = [np.array([0.0, 0.0, 0.0], dtype=float)]
    R_cur = np.eye(3)

    for i in range(NUM_SEGMENTS):
        L_i = segment_lengths[i]
        if L_i <= 1e-6:
            # No more grown length in this (or later) segments
            # We can safely break since effective lengths are front-filled.
            break

        # 1) translate along current orientation by L_i
        p_prev = points[-1]
        step = R_cur @ np.array([L_i, 0.0, 0.0])
        p_new = p_prev + step
        points.append(p_new)

        # 2) apply joint i rotation at the END of this link
        theta = joint_angles[i]
        axis = SEGMENT_AXES[i]

        if axis == "horizontal":
            R_seg = rot_z(theta)
        else:
            R_seg = rot_y(theta)

        R_cur = R_cur @ R_seg

    return np.vstack(points)


# ================================
# ROS Callbacks
# ================================
def motor_encoder_callback(msg: Float32):
    global latest_motor_encoder, latest_motor_length
    enc = msg.data
    L = encoder_to_length(enc)

    with state_lock:
        latest_motor_encoder = enc
        latest_motor_length = L


def encoders_raw_callback(msg: Int32MultiArray):
    global latest_encoders_raw
    with state_lock:
        latest_encoders_raw = list(msg.data)


# ================================
# Periodic log (text)
# ================================
def status_timer(_event):
    with state_lock:
        enc_val = latest_motor_encoder
        length  = latest_motor_length
        encs    = latest_encoders_raw.copy()

    if enc_val is None or length is None:
        return

    enc_str = f"motor={enc_val:.3f}"
    len_str = f"L={length:.3f} m"

    pair_str = ""
    if encs:
        parts = []
        for i in range(0, len(encs), 2):
            j = i // 2 + 1
            e1 = encs[i]
            e2 = encs[i + 1] if i + 1 < len(encs) else 0
            parts.append(f"J{j}=({e1},{e2})")
        pair_str = " | " + "  ".join(parts)

    rospy.loginfo_throttle(0.5, f"[Monitor] {enc_str}  {len_str}{pair_str}")


# ================================
# Real-time 3D trajectory plotter
# ================================
def plot_3d_trajectory():
    """
    Real-time 3D trajectory visualization.
    Uses:
      - latest_motor_length
      - latest_encoders_raw
    """
    plt.ion()
    fig = plt.figure("Growing Robot 3D Trajectory")
    ax = fig.add_subplot(111, projection='3d')

    while not rospy.is_shutdown() and plt.fignum_exists(fig.number):
        with state_lock:
            length = latest_motor_length
            encs   = latest_encoders_raw.copy()

        ax.cla()

        if length is None or length <= 1e-6:
            ax.set_title("Waiting for length data...")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            ax.set_xlim(-1, 1)
            ax.set_ylim(-4, 1)
            ax.set_zlim(-1, 1)
            plt.pause(0.05)
            continue

        # Compute per-segment effective lengths (partial growth handled)
        seg_lengths = compute_segment_effective_lengths(length)

        # Extract joint angles from encoders (using indices 0,2,4,6,8,10)
        joint_angles_raw = extract_joint_angles_from_encoders(encs)

        # Zero out joints that should not be active yet based on total length
        joint_angles = mask_inactive_joints(length, joint_angles_raw)

        # Forward kinematics to get 3D points
        pts = forward_kinematics_3d(seg_lengths, joint_angles)

        xs, ys, zs = pts[:, 0], pts[:, 1], pts[:, 2]

        ax.plot(xs, ys, zs, "-o")
        ax.scatter(xs[-1], ys[-1], zs[-1], s=50)  # tip marker

        ax.set_xlabel("X (forward)")
        ax.set_ylabel("Y (horizontal)")
        ax.set_zlabel("Z (vertical)")
        ax.set_title("Growing Robot 3D Trajectory")

        # Set symmetric-ish limits
        span = max(1.0, np.sum(LINK_LENGTHS))
        ax.set_xlim(0, span)
        ax.set_ylim(-span/2, span/2)
        ax.set_zlim(-span/2, span/2)

        plt.draw()
        plt.pause(0.05)


# ================================
# Main
# ================================
def main():
    rospy.init_node("encoder_monitor", anonymous=True)

    rospy.Subscriber("/motor_encoder", Float32, motor_encoder_callback)
    rospy.Subscriber("/encoders_raw", Int32MultiArray, encoders_raw_callback)

    # Text status
    rospy.Timer(rospy.Duration(0.2), status_timer)  # 5 Hz

    # Start visualization thread
    t = threading.Thread(target=plot_3d_trajectory, daemon=True)
    t.start()

    rospy.loginfo("[encoder_monitor] Listening on /motor_encoder and /encoders_raw, plotting 3D trajectory.")
    rospy.spin()


if __name__ == "__main__":
    main()