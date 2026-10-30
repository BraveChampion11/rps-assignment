# Rock-Paper-Scissors: custom CNN + live GUI (Applied Computer Vision)

## Files
| File | Purpose |
|---|---|
| `collect_data.py` | Webcam tool to build the ImageFolder dataset (`rps/rock`, `rps/paper`, `rps/scissors`) |
| `model.py` | Custom from-scratch CNN (`RPSNet`, ~1.2M params) + augmentations |
| `train.py` | Stratified 70/15/15 split, class-weighted loss, W&B logging, test report |
| `RPS_Training_Colab.ipynb` | Same pipeline for Colab GPU (writes model.py/train.py itself) |
| `gui_app.py` | Tkinter live-camera classifier |

## 1. Build the dataset (Windows, VS Code terminal)
```
pip install -r requirements.txt
python collect_data.py --out rps --subject me
```
Hold your hand in the green square, press `r`/`p`/`s` to start/stop recording that class, `q` to quit.
Rerun with `--subject friend1`, `--subject sister`, etc. Vary lighting, background, distance, left/right hand.

Requirements: >= 500 per class, >= 2000 total, **deliberately imbalanced**. Suggested target:
rock ~800, paper ~1000, scissors ~600. Check counts on the collector overlay.

## 2. Train
Local: `python train.py --data rps --epochs 30 --wandb_project rps-applied-cv`
Colab: zip `rps/` as `rps.zip`, open the notebook, choose GPU runtime, run all.
Use `--wandb_mode offline` if you don't want to log in. Outputs go to `outputs/`:
`rps_best.pt`, `results.json`, `training_summary.png`.

W&B logs: train/val loss, accuracy, macro-F1, learning rate, gradients, class distribution,
test confusion matrix, model artifact.

## 3. Run the GUI
`python gui_app.py --ckpt outputs/rps_best.pt`

Show a sign inside the green box; label, confidence and per-class bars update live.
Controls: min-confidence slider, 5-frame smoothing, mirror toggle, camera index, load model, snapshot.

## Notes for the report
- Split is stratified and disjoint (no leakage); augmentation applies to train only.
- Imbalance handled with inverse-frequency class weights; evaluate with macro-F1, not just accuracy.
- Best checkpoint chosen by validation macro-F1; test set touched once at the end.
- Collector and GUI both mirror frames and crop the same centered square, so train/inference match.