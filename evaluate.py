"""Evaluate a trained model on the held-out test split."""
import argparse, json
from pathlib import Path
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from train import CLASS_NAMES, IMG_SIZE, BATCH_SIZE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data')
    ap.add_argument('--model', default='models/mobilenetv2_final.keras')
    args = ap.parse_args()
    test_dir = Path(args.data) / 'test'
    ds = tf.keras.utils.image_dataset_from_directory(
        test_dir, labels='inferred', label_mode='int', class_names=CLASS_NAMES,
        image_size=IMG_SIZE, batch_size=BATCH_SIZE, shuffle=False)
    model = tf.keras.models.load_model(args.model)
    y_true = np.concatenate([y.numpy() for _, y in ds])
    probs = model.predict(ds, verbose=1)
    y_pred = probs.argmax(axis=1)
    report = classification_report(y_true, y_pred, labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    Path('results').mkdir(exist_ok=True)
    Path('results/classification_report.json').write_text(json.dumps(report, indent=2))
    print(classification_report(y_true, y_pred, labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES, zero_division=0))
    disp = ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES)))),
        display_labels=CLASS_NAMES)
    disp.plot(xticks_rotation=30); plt.tight_layout(); plt.savefig('results/confusion_matrix.png', dpi=160); plt.close()
    print('Saved results/classification_report.json and results/confusion_matrix.png')

if __name__ == '__main__':
    main()
