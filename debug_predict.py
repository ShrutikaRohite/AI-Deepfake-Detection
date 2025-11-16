# debug_predict.py
import os
import numpy as np
import tensorflow as tf
from data_generator import create_data_generators
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# ----- Config (change paths if needed) -----
TRAIN_DIR = "data/final_splits/train"
VAL_DIR   = "data/final_splits/val"
TEST_DIR  = "data/final_splits/test"
MODEL_PATH = "models/deepfake_model.h5"
IMG_SIZE = 224
BATCH = 16
THRESHOLD = 0.5

def main():
    print("=== Debug Predict Script ===\n")

    # Check files exist
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: model file not found at: {MODEL_PATH}")
        return
    if not os.path.exists(TEST_DIR):
        print(f"ERROR: test directory not found at: {TEST_DIR}")
        return

    # Load generators
    print("Loading data generators...")
    try:
        _, _, test_gen = create_data_generators(TRAIN_DIR, VAL_DIR, TEST_DIR, img_size=IMG_SIZE, batch_size=BATCH)
    except Exception as e:
        print("ERROR creating data generators. Make sure create_data_generators exists and paths are correct.")
        print("Exception:", e)
        return

    print("\nclass_indices mapping (folder -> label):")
    print(test_gen.class_indices)

    # Load model
    print("\nLoading model...")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded from:", MODEL_PATH)

    # Predict on entire test set
    print("\nRunning predictions on test set (this may take a bit)...")
    # Ensure steps = ceil(n_samples / batch)
    steps = int(np.ceil(test_gen.samples / float(test_gen.batch_size)))
    y_scores = model.predict(test_gen, steps=steps, verbose=1)
    y_scores = np.array(y_scores).reshape(-1)  # flatten

    y_true = test_gen.classes  # ground truth ints in the same order as filenames
    filenames = np.array(test_gen.filenames)

    # Predicted classes using threshold
    y_pred = (y_scores > THRESHOLD).astype(int)

    # Metrics
    print("\n=== Metrics ===")
    try:
        cm = confusion_matrix(y_true, y_pred)
        print("\nConfusion matrix (rows=true, cols=pred):")
        print(cm)
        print("\nClassification report:")
        # derive target names in order of class index keys sorted by value
        idx_to_name = {v: k for k, v in test_gen.class_indices.items()}
        target_names = [idx_to_name[i] for i in sorted(idx_to_name.keys())]
        print(classification_report(y_true, y_pred, target_names=target_names))
    except Exception as e:
        print("Error computing metrics:", e)

    acc = accuracy_score(y_true, y_pred)
    print(f"Overall accuracy at threshold {THRESHOLD}: {acc*100:.2f}%")

    # Show some sample predictions
    print("\nSample predictions (first 30):")
    nshow = min(30, len(filenames))
    for i in range(nshow):
        fname = filenames[i]
        true_label_idx = int(y_true[i])
        pred_score = float(y_scores[i])
        pred_label_idx = int(y_pred[i])
        print(f"{i:02d} | {fname} | true:{true_label_idx} | score:{pred_score:.4f} | pred:{pred_label_idx} "
              f"| true_name:{idx_to_name[true_label_idx]} | pred_name:{idx_to_name[pred_label_idx]}")

    # Heuristic: check if predictions appear swapped
    print("\nHeuristic check for swapped labels:")
    if cm.shape == (2,2):
        tn, fp, fn, tp = cm.ravel()
        print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")
        # If many off-diagonals > diagonals, likely swapped
        if (fp + fn) > (tp + tn):
            print("\n>>> Warning: Many predictions are off-diagonal. It may indicate label swap or model inverted output.")
            print("Possible quick fixes (pick one):")
            print("1) In Streamlit app invert the model score: pred_inverted = 1 - pred")
            print("   Then use pred_inverted for thresholding and label logic.")
            print("2) Check your folder names and reorder so that flow_from_directory maps correctly. Example mapping shown above.")
            print("3) Retrain ensuring training generator had correct folder->label mapping.")
        else:
            print("No obvious swap detected from the confusion matrix.")
    else:
        print("Confusion matrix not 2x2 — check your classes and generator settings.")

    # Option: Save CSV of predictions for manual inspection
    try:
        import pandas as pd
        out_df = pd.DataFrame({
            "filename": filenames,
            "true_label_idx": y_true,
            "true_label_name": [idx_to_name[i] for i in y_true],
            "score": y_scores,
            "pred_label_idx": y_pred,
            "pred_label_name": [idx_to_name[i] for i in y_pred]
        })
        out_csv = "debug_predictions.csv"
        out_df.to_csv(out_csv, index=False)
        print(f"\nSaved detailed predictions to: {out_csv}")
    except Exception:
        print("\n(pandas not available or failed to save CSV)")

    print("\nDone.")

if __name__ == "__main__":
    main()
