Our task is to recover the game state from the snapshots of a multiplayer UNO game. The corresponding Kaggle page is here: https://www.kaggle.com/competitions/iapr-26-uno-vision-challenge

The dataset consists of images of UNO game scenes together with structured annotations describing the game state. Each image captures a tabletop setup where multiple players are holding cards. The goal is to extract the center card, the active player, and the cards held by each player. The dataset is split into a training set (images + ground truth annotations) and a test set (images only).

## Approach

We use a classical image-processing pipeline (no learning models, no pretrained weights, no external data). Three stages, each in its own module under `src/`:

| Stage | Module | What it does |
|---|---|---|
| Segmentation | `src/segmentation.py` | Per-color HSV thresholding + morphology + single-linkage clustering of fragments into one rotated crop per card. Multi-card hands are split by detecting individual white ovals inside the cluster. Tokens come from a small-blob pass outside any detected card. |
| Object description | `src/description.py` | Each crop is reduced to a 5-way colour key (r/y/g/b/k) and a 96×96 binary signature of the central oval — the shape of the digit/icon, lighting-independent. |
| Classification | `src/classification.py` | Templates are built once from the 4 reference images (54 templates: 4 colours × 13 values + wild + draw_4). Each card is matched to the same-colour templates via normalised cross-correlation across both upright and 180°-rotated orientations. NCC thresholds drop phantom dark blobs (shadows misread as wild cards). |
| Game-state inference | `src/game_state.py` | The center card is the one closest to the image centre (if inside 35% of the diagonal). Remaining cards are assigned to p1/p2/p3/p4 by angle. The active player is the one whose hand sits closest to the detected token. |

## Project Structure

```
EE-451-Project/
├── reference_images/      4 reference images, one per row of UNO classes
├── test_images/           160 test images (1 image missing → emitted as EMPTY)
├── train_images/          81 annotated training images
├── src/
│   ├── segmentation.py
│   ├── description.py
│   ├── classification.py
│   ├── game_state.py
│   └── pipeline.py        end-to-end orchestrator (image → game-state dict)
├── scripts/
│   └── eval_train.py      multiset evaluation over train.csv
├── main.py                produces submission.csv (uses sample_submission.csv as image_id list)
├── notebook.ipynb         the Jupyter report
├── sample_submission.csv
├── train.csv
└── README.md
```

## Running

```bash
python main.py --test_dir test_images --ref_dir reference_images --out submission.csv
```

This reproduces the submission file uploaded to Kaggle.

**Author 1 (sciper):** Turcanu Magdalena (416457)  
**Author 2 (sciper):** Jakub Jan Kielar (423372)   
**Author 3 (sciper):** Zwygart Hanna (333423)
