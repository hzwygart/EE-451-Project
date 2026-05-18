"""Evaluate the pipeline on train.csv (multiset comparison for hands)."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from PIL import Image
from collections import Counter
from src.pipeline import get_templates, predict_image


def card_set(s):
    if s == "EMPTY" or pd.isna(s):
        return Counter()
    return Counter(s.split(";"))


def main(train_dir="train_images", csv="train.csv", limit=None):
    truth = pd.read_csv(csv).set_index("image_id")
    templates = get_templates("reference_images")
    rows = []
    ids = list(truth.index)
    if limit:
        ids = ids[:limit]
    t0 = time.time()
    for i, image_id in enumerate(ids):
        img = np.array(Image.open(os.path.join(train_dir, image_id + ".jpg")))
        pred = predict_image(img, templates)
        rows.append({"image_id": image_id, "pred": pred, "truth": truth.loc[image_id].to_dict()})
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(ids)} ({time.time()-t0:.1f}s)")
    cc = ca = 0
    hand_correct = hand_total = 0
    card_tp = card_fn = card_fp = 0
    for r in rows:
        p, t = r["pred"], r["truth"]
        if p["center_card"] == t["center_card"]:
            cc += 1
        if p["active_player"] == t["active_player"]:
            ca += 1
        for k in ("player_1_cards", "player_2_cards", "player_3_cards", "player_4_cards"):
            ps, ts = card_set(p[k]), card_set(t[k])
            if ps == ts:
                hand_correct += 1
            hand_total += 1
            # multiset intersection
            inter = sum((ps & ts).values())
            card_tp += inter
            card_fn += sum(ts.values()) - inter
            card_fp += sum(ps.values()) - inter
    n = len(rows)
    prec = card_tp / max(1, card_tp + card_fp)
    rec = card_tp / max(1, card_tp + card_fn)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    print(f"\nN={n}")
    print(f"center_card acc: {cc}/{n} = {100*cc/n:.1f}%")
    print(f"active_player acc: {ca}/{n} = {100*ca/n:.1f}%")
    print(f"hand exact-match: {hand_correct}/{hand_total} = {100*hand_correct/hand_total:.1f}%")
    print(f"card-level precision={prec:.3f}, recall={rec:.3f}, f1={f1:.3f}")
    return rows


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    main(limit=args.limit)
