import numpy as np

# Copied from cameras/visualization.py (line 181-191)
ENCODER_INDEX_PER_SEGMENT = [0, 2, 4, 6, 8, 10]
NUM_SEGMENTS = 6

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

_sort_idx = np.argsort(RAW_ENCODER_ANGLE_TICKS)
ENC_TBL = RAW_ENCODER_ANGLE_TICKS[_sort_idx]
ANG_TBL = RAW_ENCODER_ANGLE_DEG[_sort_idx]

# For angle->tick interpolation, ensure the angle axis is strictly increasing
_ang_sort_idx = np.argsort(ANG_TBL)
ANG_TBL_INC = ANG_TBL[_ang_sort_idx]
ENC_TBL_FOR_ANG = ENC_TBL[_ang_sort_idx]


def encoder_tick_to_angle_rad(enc_value: float) -> float:
    if enc_value <= ENC_TBL[0]:
        angle_deg = ANG_TBL[0]
    elif enc_value >= ENC_TBL[-1]:
        angle_deg = ANG_TBL[-1]
    else:
        angle_deg = np.interp(enc_value, ENC_TBL, ANG_TBL)
    return np.deg2rad(angle_deg)


def encoder_ticks_to_angle_rad(enc_values: np.ndarray) -> np.ndarray:
    enc_values = np.asarray(enc_values, dtype=float)
    angle_deg = np.interp(enc_values, ENC_TBL, ANG_TBL, left=ANG_TBL[0], right=ANG_TBL[-1])
    return np.deg2rad(angle_deg)


def angle_rad_to_encoder_tick(angle_rad: float) -> float:
    angle_deg = np.rad2deg(angle_rad)
    if angle_deg <= ANG_TBL_INC[0]:
        enc = ENC_TBL_FOR_ANG[0]
    elif angle_deg >= ANG_TBL_INC[-1]:
        enc = ENC_TBL_FOR_ANG[-1]
    else:
        enc = np.interp(angle_deg, ANG_TBL_INC, ENC_TBL_FOR_ANG)
    return float(enc)


def angles_rad_to_encoder_ticks(angles_rad: np.ndarray) -> np.ndarray:
    angles_rad = np.asarray(angles_rad, dtype=float)
    angles_deg = np.rad2deg(angles_rad)
    enc = np.interp(angles_deg, ANG_TBL_INC, ENC_TBL_FOR_ANG,
                    left=ENC_TBL_FOR_ANG[0], right=ENC_TBL_FOR_ANG[-1])
    return enc


def angles_rad_to_encoder_tick_pairs(angles_rad: np.ndarray) -> np.ndarray:
    """
    Map joint angles (rad) to encoder tick pairs (enc1, enc2) per joint.
    Convention: enc2 is the negative of enc1.
    Returns shape (..., 2) where last dim is (enc1, enc2).
    """
    ticks = angles_rad_to_encoder_ticks(angles_rad)
    ticks = np.asarray(ticks, dtype=float)
    return np.stack([ticks, -ticks], axis=-1)


def enc_list_to_joint_angles(enc_list: np.ndarray) -> np.ndarray:
    enc_list = np.asarray(enc_list, dtype=float)
    angles = np.zeros(NUM_SEGMENTS, dtype=float)
    for seg_idx, enc_idx in enumerate(ENCODER_INDEX_PER_SEGMENT):
        if enc_idx < len(enc_list):
            angles[seg_idx] = encoder_tick_to_angle_rad(enc_list[enc_idx])
        else:
            angles[seg_idx] = 0.0
    return angles
