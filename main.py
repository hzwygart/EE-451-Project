"""Generate the Kaggle submission CSV by running the classical pipeline on every test image."""
import os
import argparse
import pandas as pd

from src.pipeline import get_templates, predict_file


COLUMNS = ["image_id", "center_card", "active_player",
           "player_1_cards", "player_2_cards", "player_3_cards", "player_4_cards"]


def _empty_row(image_id):
    return {"image_id": image_id, "center_card": "EMPTY", "active_player": "p1",
            "player_1_cards": "EMPTY", "player_2_cards": "EMPTY",
            "player_3_cards": "EMPTY", "player_4_cards": "EMPTY"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_dir", default="test_images")
    ap.add_argument("--ref_dir", default="reference_images")
    ap.add_argument("--sample", default="sample_submission.csv",
                    help="CSV providing the authoritative list of image_ids to predict.")
    ap.add_argument("--out", default="submission.csv")
    args = ap.parse_args()

    templates = get_templates(args.ref_dir)
    if os.path.exists(args.sample):
        image_ids = list(pd.read_csv(args.sample)["image_id"])
    else:
        image_ids = sorted(os.path.splitext(f)[0] for f in os.listdir(args.test_dir) if f.lower().endswith(".jpg"))

    rows = []
    for i, image_id in enumerate(image_ids):
        path = os.path.join(args.test_dir, image_id + ".jpg")
        if os.path.exists(path):
            pred = predict_file(path, templates)
            pred["image_id"] = image_id
            rows.append({k: pred[k] for k in COLUMNS})
        else:
            rows.append(_empty_row(image_id))
        if (i + 1) % 10 == 0 or i + 1 == len(image_ids):
            print(f"  {i+1}/{len(image_ids)}", flush=True)

    pd.DataFrame(rows, columns=COLUMNS).to_csv(args.out, index=False)
    print(f"Wrote {args.out} with {len(rows)} rows.")


if __name__ == "__main__":
    main()
