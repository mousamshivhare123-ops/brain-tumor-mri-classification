# Brain Tumor MRI Image Classification

This project follows the supplied brief: inspect and preprocess the MRI dataset, train a custom CNN, compare it with transfer learning, evaluate the models, and deploy an interactive Streamlit demo.

## Dataset note

The uploaded archive is named `tumor.zip.zip`. Its `Tumour/` directory has `Train/`, `valid/`, and `test/` folders. The class directory spellings are inconsistent (`pituitary`/`Pituitary`, `no_tumor`/`No_tumor`, `Giloma`/`Glioma`). `prepare_data.py` normalizes these to four labels: `Glioma`, `Meningioma`, `No_Tumor`, and `Pituitary`.

The ZIP currently contains about 599 image files (200 train, 200 validation, 199 test), not the 2,443 images mentioned in the embedded README. Always use the counts printed by `prepare_data.py` as the archive's actual counts, and describe this discrepancy in the report.

## 1. Setup

Use Python 3.10 or 3.11 in a virtual environment. Install packages:

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Prepare the uploaded ZIP

```bash
python prepare_data.py --zip /path/to/tumor.zip.zip --out data
```

Replace `/path/to/tumor.zip.zip` with the actual path to the archive. Check printed class counts and inspect some images before training.

## 3. Train both models

```bash
python train.py --data data --epochs 20
```

Optional MobileNetV2 fine-tuning:

```bash
python train.py --data data --epochs 20 --fine-tune
```

The script saves models in `models/` and a basic test accuracy comparison in `results/model_comparison.json`. Compare validation results when choosing a model; the test set should be used only for final evaluation.

## 4. Evaluate the selected model

```bash
python evaluate.py --data data --model models/mobilenetv2_final.keras
```

Outputs include a per-class precision/recall/F1 report and confusion matrix. Add training-history accuracy/loss plots to the final report if required.

## 5. Launch Streamlit

```bash
streamlit run app.py
```

The app expects `models/mobilenetv2_final.keras` and `models/class_names.json`.

## Suggested 14-day schedule

1. Day 1: inspect dataset, class labels and image quality.
2. Day 2: clean/normalize folder names and document class counts.
3. Day 3: resize/normalize images and build training pipelines.
4. Days 4–5: train and tune custom CNN.
5. Days 6–7: train MobileNetV2 transfer-learning model.
6. Day 8: optional fine-tuning and model selection using validation data.
7. Days 9–10: evaluate with accuracy, precision, recall, F1 and confusion matrix.
8. Day 11: compare models and save the selected model.
9. Day 12: build and test Streamlit interface.
10. Day 13: write README/report and organize code.
11. Day 14: final checks, reproducibility notes and submission.

## Important limitations

- This is an educational classifier, not a clinically validated diagnostic system.
- A dataset folder label is not independent proof of medical ground truth.
- A small dataset can lead to overfitting and unreliable estimates. Do not claim clinical accuracy or safety based only on this test split.
- Check for near-duplicate or related images across train/validation/test before trusting the metrics.
- The archive's embedded README states CC BY 4.0; retain attribution to the original dataset creator/source in public documentation.
