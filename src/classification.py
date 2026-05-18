"""Build reference templates from the 4 reference images and classify cards."""
import os
import numpy as np
from PIL import Image

from .segmentation import detect_cards
from .description import card_color, symbol_signature, best_match_score


# Hardcoded card identity in each reference image. Each entry: (image_id, [(symbol, row, col_color_order), ...])
# Reference layouts (verified visually):
#   L1000765: rows top->bottom = 3,2,1  cols left->right = y,r,b,g  (12 cards)
#   L1000766: rows = 6,5,4              cols = y,r,b,g              (12 cards)
#   L1000767: 3x4 grid of 9,8,7 in y,r,b,g + 2 standalone (draw_4, wild) on the left
#   L1000768: rows = reverse, skip, [draw_2 x4 | 0 x4]  cols = y,r,b,g
REF_LAYOUTS = {
    "L1000765": {"grid": [("3", 3), ("2", 3), ("1", 3)], "cols": ["y", "r", "b", "g"]},
    "L1000766": {"grid": [("6", 3), ("5", 3), ("4", 3)], "cols": ["y", "r", "b", "g"]},
    "L1000767": {"grid": [("9", 3), ("8", 3), ("7", 3)], "cols": ["y", "r", "b", "g"],
                  "extras": [("draw_4", "k"), ("wild", "k")]},
    "L1000768": {"grid": [("reverse", 3), ("skip", 3)], "cols": ["y", "r", "b", "g"],
                  "bottom_row": [("draw_2", ["y", "b", "r", "g"]), ("0", ["y", "r", "b", "g"])]},
}


def _grid_cluster(cards, n_rows, n_cols):
    """Cluster cards into rows by y-centroid, then sort each row by x. Returns 2D list of card dicts."""
    cs = sorted(cards, key=lambda c: c["centroid"][0])
    rows = []
    for i in range(n_rows):
        start = i * n_cols
        row = sorted(cs[start:start + n_cols], key=lambda c: c["centroid"][1])
        rows.append(row)
    return rows


def _label_reference(ref_id, cards):
    """Assign labels (e.g. 'r_3') to detected cards based on REF_LAYOUTS."""
    layout = REF_LAYOUTS[ref_id]
    labeled = []
    if ref_id == "L1000767":
        # Right side: 3 rows of 4 cards each (9, 8, 7 in y,r,b,g). Left side: 2 extra cards.
        cs = sorted(cards, key=lambda c: c["centroid"][1])
        left = sorted(cs[:2], key=lambda c: c["centroid"][0])
        right = cs[2:]
        labeled.append(("draw_4", left[0]))
        labeled.append(("wild", left[1]))
        rows = _grid_cluster(right, 3, 4)
        for (sym, _), row in zip(layout["grid"], rows):
            for col_color, card in zip(layout["cols"], row):
                labeled.append((f"{col_color}_{sym}", card))
        return labeled
    if ref_id == "L1000768":
        # Top 2 rows = reverse, skip in 4 colors. Bottom row = 4 draw_2 then 4 zeros.
        cs = sorted(cards, key=lambda c: c["centroid"][0])
        top8 = cs[:8]
        bot8 = cs[8:]
        rows = _grid_cluster(top8, 2, 4)
        for (sym, _), row in zip(layout["grid"], rows):
            for col_color, card in zip(layout["cols"], row):
                labeled.append((f"{col_color}_{sym}", card))
        bot8 = sorted(bot8, key=lambda c: c["centroid"][1])
        left4, right4 = bot8[:4], bot8[4:]
        for col_color, card in zip(layout["bottom_row"][0][1], left4):
            labeled.append((f"{col_color}_draw_2", card))
        for col_color, card in zip(layout["bottom_row"][1][1], right4):
            labeled.append((f"{col_color}_0", card))
        return labeled
    # Generic n_rows x n_cols grid
    n_cols = len(layout["cols"])
    rows = _grid_cluster(cards, len(layout["grid"]), n_cols)
    for (sym, _), row in zip(layout["grid"], rows):
        for col_color, card in zip(layout["cols"], row):
            labeled.append((f"{col_color}_{sym}", card))
    return labeled


def build_templates(ref_dir="reference_images"):
    """Return list of (label, color_key, signature) for every reference card."""
    templates = []
    for ref_id in REF_LAYOUTS:
        path = os.path.join(ref_dir, ref_id + ".jpg")
        img = np.array(Image.open(path))
        cards = detect_cards(img)
        # Sort by area desc and keep the expected number
        layout = REF_LAYOUTS[ref_id]
        expected = sum(n for _, n in layout["grid"]) * len(layout["cols"])
        if "extras" in layout:
            expected += len(layout["extras"])
        if "bottom_row" in layout:
            expected += sum(len(cols) for _, cols in layout["bottom_row"])
        cards = sorted(cards, key=lambda c: -(c["rect"][1][0] * c["rect"][1][1]))[:expected]
        labeled = _label_reference(ref_id, cards)
        for label, c in labeled:
            color = label.split("_", 1)[0] if label.startswith(("r_", "y_", "g_", "b_")) else "k"
            sig = symbol_signature(c["crop"])
            templates.append((label, color, sig))
    return templates


DARK_REJECT_SCORE = 0.55
GENERIC_REJECT_SCORE = 0.50  # All cards must clear this NCC threshold or they are dropped.


def classify_card(crop, templates):
    """Return (label, color_key, score). ``label=None`` if no template matched well enough."""
    color = card_color(crop)
    sig = symbol_signature(crop)
    candidates = [t for t in templates if t[1] == color] or templates
    best_label, best_score = None, -1.0
    for label, _, ref_sig in candidates:
        score = best_match_score(sig, ref_sig)
        if score > best_score:
            best_score = score
            best_label = label
    if best_score < GENERIC_REJECT_SCORE:
        return None, color, best_score
    if color == "k" and best_score < DARK_REJECT_SCORE:
        return None, color, best_score
    return best_label, color, best_score
