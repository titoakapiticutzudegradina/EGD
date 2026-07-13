import numpy as np
import pandas as pd

#load data from train and test paths
def load_data(train_path, test_path):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df

#early detection from predictions
def early_detection_from_preds(
    df,
    preds,
    checkpoints=None,
    *,
    probs=None,
    thresholds_by_progress=None,
    default_threshold=0.5,
):

    if checkpoints is None:
        checkpoints = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
        labels = df["label"].values
        progress = df["progress"].values
        results = []
        #iterate over checkpoints
        for cp in checkpoints: 
            #create mask for progress <= cp
            mask = progress <= cp
            if not mask.any():
                continue
            #get threshold for current checkpoint
            thr = thresholds_by_progress.get(float(cp), default_threshold)
            if thr is None:
                thr = thresholds_by_progress.get(cp, default_threshold)
            #create binary mask for probs >= thr
            binary = (probs[mask] >= thr).astype(int)
            #calculate f1 score
            f1 = f1_score(labels[mask], binary)
            results.append((cp, f1))
        return results

    #early detection from predictions
    preds = np.asarray(preds).reshape(-1)
    results = []
    for cp in checkpoints:
        mask = df["progress"] <= cp
        if not mask.any():
            continue
        f1 = f1_score(df.loc[mask, "label"], preds[mask])
        results.append((cp, f1))

    return results