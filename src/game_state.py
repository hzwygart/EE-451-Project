"""Assemble the full game state from detected cards + tokens.

Conventions:
    player_1 -> bottom of the image
    player_2 -> right
    player_3 -> top
    player_4 -> left  (players are ordered counter-clockwise)

Each card is assigned to a player by the angle of its centroid measured from
the image centre. The card closest to the image centre is the center card.
The active player is the one whose hand sits closest to the detected token.
"""
import numpy as np


def _assign_player(cy, cx, h, w):
    dy = cy - h / 2
    dx = cx - w / 2
    angle = np.arctan2(dy, dx)
    if -np.pi / 4 <= angle <= np.pi / 4:
        return 2  # right
    if np.pi / 4 < angle <= 3 * np.pi / 4:
        return 1  # bottom
    if -3 * np.pi / 4 <= angle < -np.pi / 4:
        return 3  # top
    return 4  # left


def assign_game_state(cards, tokens, img_shape):
    h, w = img_shape[:2]
    ic = np.array([h / 2, w / 2])
    if not cards:
        return {"center": None, "hands": {1: [], 2: [], 3: [], 4: []}, "active": "p1"}

    # Center card: closest to image centre. We require it sits reasonably close
    # (within ~30% of the image diagonal) to avoid stealing a player's card.
    dists = [np.linalg.norm(np.array(c["centroid"]) - ic) for c in cards]
    diag = np.hypot(h, w)
    center_idx = int(np.argmin(dists))
    center_card = cards[center_idx] if dists[center_idx] < 0.35 * diag else None

    hands = {1: [], 2: [], 3: [], 4: []}
    for i, c in enumerate(cards):
        if center_card is not None and i == center_idx:
            continue
        p = _assign_player(c["centroid"][0], c["centroid"][1], h, w)
        hands[p].append(c)

    active = _active_player(hands, tokens, img_shape)
    return {"center": center_card, "hands": hands, "active": active}


def _active_player(hands, tokens, img_shape):
    h, w = img_shape[:2]
    if not tokens:
        # Fall back to the player with the most cards (rare scenario, no token detected).
        non_empty = [p for p, cs in hands.items() if cs]
        return f"p{non_empty[0]}" if non_empty else "p1"
    # Choose the token closest to *any* hand (filters spurious dark blobs that aren't tokens).
    best_tok = None
    best_dist = float("inf")
    for tok in tokens:
        for cs in hands.values():
            for c in cs:
                d = np.linalg.norm(np.array(tok["centroid"]) - np.array(c["centroid"]))
                if d < best_dist:
                    best_dist = d
                    best_tok = tok
    if best_tok is None:
        # No card detected at all — assign token to a quadrant directly.
        tc_y, tc_x = best_tok["centroid"] if best_tok else (h / 2, w / 2)
        return f"p{_assign_player(tc_y, tc_x, h, w)}"
    # Active player = the one whose mean centroid is closest to the chosen token.
    tc = np.array(best_tok["centroid"])
    best_p, best = None, float("inf")
    for p, cs in hands.items():
        if not cs:
            continue
        mean_pos = np.mean([c["centroid"] for c in cs], axis=0)
        d = np.linalg.norm(mean_pos - tc)
        if d < best:
            best = d
            best_p = p
    if best_p is None:
        return f"p{_assign_player(tc[0], tc[1], h, w)}"
    return f"p{best_p}"
