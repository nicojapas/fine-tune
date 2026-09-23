import numpy as np

FLAT_THRESH = 8
EDGE_THRESH = 40
MAX_COLORS = 16
MAX_SOFT_RATIO = 0.05
MAX_BLOCK_STD = 3.0
COLOR_BUCKET = 16


def edge_deltas(image):
    arr = np.asarray(image.convert("L"), dtype=np.int16)
    row_deltas = np.abs(np.diff(arr, axis=1))
    col_deltas = np.abs(np.diff(arr, axis=0))
    return row_deltas, col_deltas


def soft_ratio(image):
    row_deltas, col_deltas = edge_deltas(image)
    deltas = np.concatenate([row_deltas.ravel(), col_deltas.ravel()])
    soft = (deltas > FLAT_THRESH) & (deltas < EDGE_THRESH)
    return soft.mean()


def color_count(image):
    arr = np.asarray(image.convert("RGB"))
    quantized = (arr // COLOR_BUCKET) * COLOR_BUCKET
    return len(np.unique(quantized.reshape(-1, 3), axis=0))


def block_size_std(image):
    row_deltas, _ = edge_deltas(image)
    edge_positions = [np.flatnonzero(row >= EDGE_THRESH) for row in row_deltas]
    gaps = [np.diff(p) for p in edge_positions if len(p) > 1]
    if not gaps:
        return 0.0
    return float(np.std(np.concatenate(gaps)))


def passes(image):
    return (
        soft_ratio(image) <= MAX_SOFT_RATIO
        and color_count(image) <= MAX_COLORS
        and block_size_std(image) <= MAX_BLOCK_STD
    )
