"""End-to-end pipeline: image -> game state dict matching the CSV columns."""
import os
import numpy as np
from PIL import Image

from .segmentation import detect_cards, detect_yellow_token, detect_dark_token
from .description import is_dark_token
from .classification import classify_card, build_templates
from .game_state import assign_game_state


_TEMPLATE_CACHE = None


def get_templates(ref_dir="reference_images"):
    global _TEMPLATE_CACHE
    if _TEMPLATE_CACHE is None:
        _TEMPLATE_CACHE = build_templates(ref_dir)
    return _TEMPLATE_CACHE


def predict_image(img, templates):
    raw = detect_cards(img)
    classified, tokens = [], []
    # Some dark tokens already show up as a "card" cluster (they are card-
    # sized). Split each candidate into either a token or a real card first.
    for c in raw:
        if is_dark_token(c["crop"]):
            tokens.append({"centroid": c["centroid"], "area": c["rect"][1][0] * c["rect"][1][1],
                            "kind": "dark"})
            continue
        label = classify_card(c["crop"], templates)[0]
        if label is not None:
            classified.append({**c, "label": label})
    # Dedicated detectors for tokens that the card pass misses entirely:
    # the hue-cut dark mask in the card pipeline skips neutral grays, so the
    # dark domino sometimes isn't picked up there.
    tokens.extend(detect_dark_token(img, cards=classified))
    tokens.extend(detect_yellow_token(img, cards=classified))
    # Drop duplicates that landed within a few px of each other.
    dedup = []
    for t in tokens:
        if any(abs(t["centroid"][0] - u["centroid"][0]) + abs(t["centroid"][1] - u["centroid"][1]) < 60 for u in dedup):
            continue
        dedup.append(t)
    state = assign_game_state(classified, dedup, img.shape)
    center = state["center"]["label"] if state["center"] is not None else "EMPTY"
    hands = {}
    for p in (1, 2, 3, 4):
        labels = [c["label"] for c in state["hands"][p]]
        hands[p] = ";".join(labels) if labels else "EMPTY"
    return {
        "center_card": center,
        "active_player": state["active"],
        "player_1_cards": hands[1],
        "player_2_cards": hands[2],
        "player_3_cards": hands[3],
        "player_4_cards": hands[4],
    }


def predict_file(path, templates):
    img = np.array(Image.open(path))
    return predict_image(img, templates)
