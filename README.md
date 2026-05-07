Our task is to recover the game state from the snapshots of a multiplayer UNO game. 
The corresponding kaggle page is here : https://www.kaggle.com/competitions/iapr-26-uno-vision-challengeThe dataset consists of images of UNO game scenes together with structured annotations describing the game state.

Each image captures a tabletop setup where multiple players are holding cards. The goal is to extract the center card, the active player, and the cards held by each player.

The dataset is split into:
  - Training set: images + ground truth annotations
  - Test set: images only (labels hidden)

## Project Structure

```
EE-451-Project/
├── reference_images/
├── test_images/
├── train_images/
├── src/
├── main.py
├── notebook.ipynb
├── sample_submission.csv
├── train.csv
└── README.md
```
**Author 1 (sciper):** Turcanu Magdalena (416457)  
**Author 2 (sciper):** Jakub Jan Kielar (423372)   
**Author 3 (sciper):** Zwygart Hanna (333423)
