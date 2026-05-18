"""Per-card descriptors: dominant color + grayscale symbol signature for matching.

Also exposes ``is_dark_token`` — a quick test that distinguishes a solid
dark domino-shaped token (no internal colour) from a wild / draw_4 card
(dark frame with a rainbow oval inside).
"""
import cv2
import numpy as np
from skimage.color import rgb2hsv


COLOR_NAMES = ["r", "y", "g", "b"]


def _center_hsv(crop):
    """Mean HSV over the inner area (drop the 15% border) to avoid background bleed."""
    h, w = crop.shape[:2]
    inner = crop[int(0.15 * h):int(0.85 * h), int(0.15 * w):int(0.85 * w)]
    return rgb2hsv(inner)


def card_color(crop):
    """Return one of 'r','y','g','b' or 'k' (black background for wild/draw_4)."""
    hsv = _center_hsv(crop)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    sat_mask = s > 0.45
    dark_mask = (v < 0.45) & (s < 0.4)
    if dark_mask.mean() > 0.30 and sat_mask.mean() < 0.30:
        return "k"
    if sat_mask.sum() == 0:
        return "k"
    hh = h[sat_mask]
    scores = {
        "r": ((hh < 0.06) | (hh > 0.94)).mean(),
        "y": ((hh > 0.08) & (hh < 0.20)).mean(),
        "g": ((hh > 0.22) & (hh < 0.45)).mean(),
        "b": ((hh > 0.48) & (hh < 0.70)).mean(),
    }
    return max(scores, key=scores.get)


def is_dark_token(crop):
    """A dark token (domino-like block) is a solid dark rectangle with no
    colored content. Wild and draw_4 cards are also dark but contain a
    rainbow oval or colored '+4' design — those have many saturated pixels.

    We look across the full crop (not just a tight central window) because the
    distinctive coloured marks of wild / draw_4 reach close to the card edges.
    """
    hsv = rgb2hsv(crop)
    s, v = hsv[:, :, 1], hsv[:, :, 2]
    H, W = v.shape
    iv = v[int(0.10 * H):int(0.90 * H), int(0.10 * W):int(0.90 * W)]
    is_ = s[int(0.10 * H):int(0.90 * H), int(0.10 * W):int(0.90 * W)]
    dark_frac = (iv < 0.55).mean()
    sat_frac = (is_ > 0.30).mean()
    return dark_frac > 0.80 and sat_frac < 0.05


def symbol_signature(crop, size=96):
    """Binarized symbol mask from the central oval region — captures the digit/icon shape only."""
    h, w = crop.shape[:2]
    inner = crop[int(0.22 * h):int(0.78 * h), int(0.20 * w):int(0.80 * w)]
    hsv = rgb2hsv(inner)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    # The symbol is the saturated (colored or dark) part inside the white oval.
    sym = (sat > 0.30) | (val < 0.40)
    resized = cv2.resize(sym.astype(np.uint8), (size, size), interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32)


def best_match_score(sig, ref_sig):
    """Return max(NCC) across upright and 180-rotated orientations."""
    a = sig - sig.mean()
    sa = np.sqrt((a * a).sum()) + 1e-8
    best = -1.0
    for ref in (ref_sig, ref_sig[::-1, ::-1]):
        b = ref - ref.mean()
        sb = np.sqrt((b * b).sum()) + 1e-8
        score = float((a * b).sum() / (sa * sb))
        if score > best:
            best = score
    return best
