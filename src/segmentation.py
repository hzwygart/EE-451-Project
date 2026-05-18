"""Segmentation: HSV thresholding -> per-card oriented crops.

Pipeline
--------
1. Apply per-color HSV thresholds (red, yellow, green, blue) plus a dark-mask
   for wild / draw_4 cards, all gated by a global saturation cut to suppress
   the background. Light closing smooths the masks.
2. Each card's color ring and inner number often appear as several
   neighbouring fragments. Single-linkage cluster the fragments so that
   pieces less than ~200 px apart merge into one card.
3. For every cluster, take the union of its masks, fit a rotated bounding
   box, filter by area and aspect ratio, and warp the box upright.
"""
import cv2
import numpy as np
from skimage.color import rgb2hsv
from skimage.measure import label, regionprops
from skimage.morphology import binary_closing, disk, remove_small_objects
from scipy.cluster.hierarchy import fcluster, linkage

DOWNSAMPLE = 2             # process at 1/DOWNSAMPLE resolution for speed
CLUSTER_EPS = 200 / DOWNSAMPLE          # px gap that still belongs to the same card
CARD_MIN_AREA = 16000 // (DOWNSAMPLE ** 2)
CARD_ASPECT_OK = (1.10, 2.20)
TOKEN_AREA = (500 // (DOWNSAMPLE ** 2), 4500 // (DOWNSAMPLE ** 2))
TOKEN_ASPECT_MAX = 1.7


def _hsv(img):
    h = rgb2hsv(img)
    return h[:, :, 0], h[:, :, 1], h[:, :, 2]


def _hsv_thresholds(h, s, v):
    red    = ((h < 0.05) | (h > 0.95)) & (s > 0.65) & (v > 0.85)
    yellow = (h > 0.10) & (h < 0.16) & (v > 0.60)
    green  = (h > 0.25) & (h < 0.42) & (v > 0.60) & (v < 0.88) & (s > 0.20)
    blue   = (h > 0.50) & (h < 0.65) & (s > 0.40) & (v > 0.30)
    dark   = (v < 0.55) & (s < 0.31) & (h < 0.12)
    m = (red | yellow | green | blue | dark) & ((s > 0.46) | dark)
    return m


def hsv_card_mask(img):
    """Solid mask of colored + dark card pixels (no morphological closing yet).

    Convenience wrapper kept for the notebook; downstream code uses the cached
    HSV path in ``detect_cards``.
    """
    h, s, v = _hsv(img)
    m = _hsv_thresholds(h, s, v)
    m = remove_small_objects(m.astype(bool), min_size=60 // (DOWNSAMPLE ** 2))
    m = binary_closing(m, disk(max(2, 9 // DOWNSAMPLE)))
    m = remove_small_objects(m, min_size=4000 // (DOWNSAMPLE ** 2))
    return m


def _cluster_fragments(props, eps=CLUSTER_EPS):
    """Single-linkage cluster fragment centroids by Euclidean distance."""
    if len(props) <= 1:
        return [[i] for i in range(len(props))]
    centroids = np.array([p.centroid for p in props])
    Z = linkage(centroids, method="single", metric="euclidean")
    labels = fcluster(Z, t=eps, criterion="distance")
    clusters = {}
    for i, c in enumerate(labels):
        clusters.setdefault(c, []).append(i)
    return list(clusters.values())


def _cluster_rect(mask, props, indices):
    union = np.zeros_like(mask, dtype=np.uint8)
    for i in indices:
        union[tuple(props[i].coords.T)] = 255
    cnts, _ = cv2.findContours(union, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    pts = np.vstack(cnts).reshape(-1, 2)
    return cv2.minAreaRect(pts)


def _is_card_rect(rect, total_area):
    (_, _), (w, h), _ = rect
    short, long_ = sorted([w, h])
    if short < 1:
        return False
    ar = long_ / short
    if not (CARD_ASPECT_OK[0] <= ar <= CARD_ASPECT_OK[1]):
        return False
    if total_area < CARD_MIN_AREA:
        return False
    return True


def _warp_card(img, rect, scale=1.06, out_w=160, out_h=240):
    (cx, cy), (w, h), angle = rect
    if w > h:
        w, h = h, w
        angle = angle + 90
    big_w = w * scale
    big_h = h * scale
    a = np.deg2rad(angle)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    half = np.array([[-big_w / 2, -big_h / 2], [big_w / 2, -big_h / 2],
                     [big_w / 2,  big_h / 2], [-big_w / 2,  big_h / 2]])
    src = (half @ R.T + [cx, cy]).astype(np.float32)
    dst = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (out_w, out_h))


OVAL_AREA = (5000 // (DOWNSAMPLE ** 2), 25000 // (DOWNSAMPLE ** 2))
OVAL_ASPECT = (1.10, 3.0)
SINGLE_CARD_AREA = 28000 // (DOWNSAMPLE ** 2)


def _find_ovals_inside(white_global, rect, expand=1.15):
    """Detect white oval centroids inside a rotated rect — one per stacked colored card.

    ``white_global`` is the precomputed white-pixel mask for the whole image.
    """
    (cx, cy), (w, h), angle = rect
    big_w = w * expand
    big_h = h * expand
    a = np.deg2rad(angle)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    half = np.array([[-big_w / 2, -big_h / 2], [big_w / 2, -big_h / 2],
                     [big_w / 2,  big_h / 2], [-big_w / 2,  big_h / 2]])
    corners = (half @ R.T + [cx, cy]).astype(np.int32)
    rect_mask = np.zeros(white_global.shape, dtype=np.uint8)
    cv2.fillConvexPoly(rect_mask, corners, 1)
    white = white_global & rect_mask.astype(bool)
    white = remove_small_objects(white, min_size=OVAL_AREA[0])
    centroids = []
    for p in regionprops(label(white)):
        if not (OVAL_AREA[0] <= p.area <= OVAL_AREA[1]):
            continue
        if p.minor_axis_length < 1:
            continue
        ar = p.major_axis_length / p.minor_axis_length
        if not (OVAL_ASPECT[0] <= ar <= OVAL_ASPECT[1]):
            continue
        centroids.append((p.centroid[0], p.centroid[1]))
    return centroids


def _downsample(img):
    if DOWNSAMPLE == 1:
        return img
    return img[::DOWNSAMPLE, ::DOWNSAMPLE]


def detect_cards(img, out_w=160, out_h=240):
    """Return list of dicts with keys: centroid (y,x), rect, crop.

    The image is processed at 1/DOWNSAMPLE resolution; crops are warped from
    the original full-resolution image so classification still sees crisp
    symbols. Per cluster: if >=2 white ovals are found inside the cluster
    rect, treat each oval as one stacked card; otherwise emit a single card.
    """
    small = _downsample(img)
    mask = hsv_card_mask(small)
    h_, s_, v_ = _hsv(small)
    white_global = (v_ > 0.82) & (s_ < 0.22)
    lbl = label(mask)
    props = regionprops(lbl)
    cards = []
    for cluster in _cluster_fragments(props):
        total_area = sum(props[i].area for i in cluster)
        rect = _cluster_rect(mask, props, cluster)
        if rect is None or not _is_card_rect(rect, total_area):
            continue
        (cx, cy), (rw, rh), ang = rect
        likely_multi = total_area > 1.5 * SINGLE_CARD_AREA
        ovals = _find_ovals_inside(white_global, rect) if likely_multi else []
        # Rescale rect back to original-image coordinates for warping.
        D = DOWNSAMPLE
        full_rect = ((cx * D, cy * D), (rw * D, rh * D), ang)
        if len(ovals) >= 2:
            per_w = min(rw, rh) * D
            per_h = max(rw, rh) * D
            for oy, ox in ovals:
                per_rect = ((ox * D, oy * D), (per_w, per_h), ang)
                crop = _warp_card(img, per_rect, scale=1.10, out_w=out_w, out_h=out_h)
                cards.append({"centroid": (oy * D, ox * D), "rect": per_rect, "crop": crop})
        else:
            crop = _warp_card(img, full_rect, scale=1.06, out_w=out_w, out_h=out_h)
            cards.append({"centroid": (cy * D, cx * D), "rect": full_rect, "crop": crop})
    return cards


def _card_exclusion_mask(shape, cards, expand=30):
    """Mask covering all detected-card pixels, slightly expanded, to exclude tokens inside cards."""
    mask = np.zeros(shape[:2], dtype=np.uint8)
    for c in cards:
        rect = c["rect"]
        (cx, cy), (w, h), ang = rect
        bigger = ((cx, cy), (w + expand, h + expand), ang)
        box = cv2.boxPoints(bigger).astype(np.int32)
        cv2.fillConvexPoly(mask, box, 1)
    return mask.astype(bool)


def detect_dark_token(img, cards=None):
    """Find a solid dark token block (used on the white-table images).

    The token is a domino-sized dark grey rectangle with no internal colour.
    Our card mask uses a hue cut that excludes neutral grays, so it can miss
    this token entirely — this detector uses a hue-free dark threshold and
    requires the blob to be card-sized, rectangular, and uncoloured.
    """
    h, s, v = _hsv(img)
    dark = (v < 0.45) & (s < 0.30)
    if cards is not None:
        dark = dark & ~_card_exclusion_mask(img.shape, cards, expand=20)
    dark = binary_closing(dark, disk(4))
    dark = remove_small_objects(dark, min_size=8000)
    cnts, _ = cv2.findContours(dark.astype(np.uint8) * 255,
                                cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cnts:
        a = cv2.contourArea(c)
        if not (8000 < a < 80000):
            continue
        rect = cv2.minAreaRect(c)
        (cx, cy), (w, hh), _ = rect
        if min(w, hh) < 1:
            continue
        aspect = max(w, hh) / min(w, hh)
        if aspect > 2.2:
            continue
        rect_area = w * hh
        if a / rect_area < 0.75:
            continue
        # Reject if there is saturated colour inside the bbox (wild / draw_4).
        y0, y1 = max(0, int(cy - hh / 2)), min(img.shape[0], int(cy + hh / 2))
        x0, x1 = max(0, int(cx - w / 2)), min(img.shape[1], int(cx + w / 2))
        if x1 > x0 and y1 > y0:
            inner_s = s[y0:y1, x0:x1]
            if (inner_s > 0.30).mean() > 0.05:
                continue
        out.append({"centroid": (cy, cx), "area": a, "kind": "dark"})
    return out


def detect_yellow_token(img, cards=None):
    """Find the yellow disc token (used on the leaf-pattern background).

    The disc is a saturated, near-round yellow blob sitting on its own —
    distinct from yellow cards (much larger and accompanied by a white oval).
    """
    h, s, v = _hsv(img)
    yellow = (h > 0.11) & (h < 0.18) & (s > 0.70) & (v > 0.70)
    if cards is not None:
        yellow = yellow & ~_card_exclusion_mask(img.shape, cards, expand=20)
    yellow = remove_small_objects(yellow, min_size=2000)
    yellow = binary_closing(yellow, disk(3))
    cnts, _ = cv2.findContours(yellow.astype(np.uint8) * 255,
                                cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cnts:
        a = cv2.contourArea(c)
        if not (2500 < a < 25000):
            continue
        rect = cv2.minAreaRect(c)
        (cx, cy), (w, hh), _ = rect
        if min(w, hh) < 1:
            continue
        aspect = max(w, hh) / min(w, hh)
        if aspect > 1.35:
            continue
        fill = a / (w * hh)
        if fill < 0.70:
            continue
        out.append({"centroid": (cy, cx), "area": a, "kind": "yellow"})
    return out


# Backwards-compatible alias used by older code / the notebook.
def detect_tokens(img, cards=None):
    return detect_yellow_token(img, cards)


def detect_scene(img, out_w=160, out_h=240):
    cards = detect_cards(img, out_w, out_h)
    return {"cards": cards, "tokens": detect_tokens(img, cards=cards)}
