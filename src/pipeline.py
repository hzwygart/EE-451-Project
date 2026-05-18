"""End-to-end pipeline: image -> game state dict matching the CSV columns."""
import os
import numpy as np
from PIL import Image

from .segmentation import detect_cards, detect_tokens
from .classification import classify_card, build_templates
from .game_state import assign_game_state


_TEMPLATE_CACHE = None


def get_templates(ref_dir="reference_images"):
    global _TEMPLATE_CACHE
    if _TEMPLATE_CACHE is None:
        _TEMPLATE_CACHE = build_templates(ref_dir)
    return _TEMPLATE_CACHE


def predict_image(img, templates):
    cards = detect_cards(img)
    # Classify each detected card and keep only those that match a template confidently.
    classified = []
    for c in cards:
        label = classify_card(c["crop"], templates)[0]
        if label is not None:
            classified.append({**c, "label": label})
    # Tokens are detected after the phantom rejection so that a misclassified
    # token (initially seen as a card and then dropped) is not excluded.
    tokens = detect_tokens(img, cards=classified)
    state = assign_game_state(classified, tokens, img.shape)
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
