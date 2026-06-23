import numpy as np

# Copied from cameras/visualization.py (encoder -> length table)
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
    4.23,
], dtype=float)

_sort_by_encoder = np.argsort(RAW_ENCODER_TABLE)
ENCODER_TABLE = RAW_ENCODER_TABLE[_sort_by_encoder]
LENGTH_TABLE_BY_ENCODER = RAW_LENGTH_TABLE[_sort_by_encoder]

_sort_by_length = np.argsort(RAW_LENGTH_TABLE)
LENGTH_TABLE = RAW_LENGTH_TABLE[_sort_by_length]
ENCODER_TABLE_BY_LENGTH = RAW_ENCODER_TABLE[_sort_by_length]


def _interp_array(values: np.ndarray, xp: np.ndarray, fp: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    flat = values.reshape(-1)
    out = np.interp(flat, xp, fp, left=fp[0], right=fp[-1])
    return out.reshape(values.shape)


def encoder_to_length(enc_value: float) -> float:
    """Convert base encoder tick reading -> robot length in meters."""
    return float(
        np.interp(
            enc_value,
            ENCODER_TABLE,
            LENGTH_TABLE_BY_ENCODER,
            left=LENGTH_TABLE_BY_ENCODER[0],
            right=LENGTH_TABLE_BY_ENCODER[-1],
        )
    )


def encoders_to_length(enc_values: np.ndarray) -> np.ndarray:
    """Vectorized encoder -> length conversion (meters)."""
    return _interp_array(enc_values, ENCODER_TABLE, LENGTH_TABLE_BY_ENCODER)


def length_to_encoder(length_m: float) -> float:
    """Convert length in meters -> base encoder tick reading."""
    return float(
        np.interp(
            length_m,
            LENGTH_TABLE,
            ENCODER_TABLE_BY_LENGTH,
            left=ENCODER_TABLE_BY_LENGTH[0],
            right=ENCODER_TABLE_BY_LENGTH[-1],
        )
    )


def lengths_to_encoder(lengths_m: np.ndarray) -> np.ndarray:
    """Vectorized length (meters) -> encoder ticks conversion."""
    return _interp_array(lengths_m, LENGTH_TABLE, ENCODER_TABLE_BY_LENGTH)
