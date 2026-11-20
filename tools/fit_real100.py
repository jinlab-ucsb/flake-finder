"""Fit the real-flake-or-glue model of flakeid.real100 to 100x pictures labelled by eye.

    python tools/fit_real100.py LABELS.json -o flakeid/profiles/real100.json

LABELS.json: [{"folder": "<piece>_100x", "default": "real" | "glue", "real": [picture numbers], "glue": [picture numbers]}, ...]
(picture numbers: the order column of the folder's index; pictures not listed take the default).  Prints a 5-fold
cross-validation and writes the model (a logistic regression on the standardised measures).
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flakeid import real100  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("labels")
    ap.add_argument("-o", "--output", required=True)
    a = ap.parse_args()
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    X, y, names = [], [], []
    for s in json.loads(Path(a.labels).read_text()):
        folder = Path(s["folder"])
        rows = list(csv.DictReader(open(next(folder.glob("*_100x_index.csv")))))
        fr = {}
        for r in rows:
            f = folder / r["file"]
            if not f.is_file():
                f = folder / "rejected" / r["file"]
            fr[int(r["order"])] = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
        sub = np.median(np.array([real100.frame_mode(v) for v in fr.values() if v is not None]), axis=0)
        for n, img in fr.items():
            if img is None:
                continue
            lab = 1 if n in s.get("real", []) else 0 if n in s.get("glue", []) else int(s.get("default", "real") == "real")
            f = real100.features(img, sub)
            X.append(real100.vector(f)); y.append(lab); names.append((folder.name, n, f))
    X, y = np.array(X), np.array(y)
    mean, std = X.mean(axis=0), X.std(axis=0) + 1e-9
    Z = (X - mean) / std
    lr = LogisticRegression(C=1.0, class_weight="balanced")
    p = cross_val_predict(lr, Z, y, cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
    full = np.array([f["cover"] >= real100.FULL_COVER and f["flat_frac"] >= real100.FULL_FLAT for _, _, f in names])
    p = np.where(full, 0.99, p)
    print(f"{len(y)} pictures: {int(y.sum())} real, {int((1 - y).sum())} glue; cross-validated at 0.5: real kept "
          f"{int((p[y == 1] >= .5).sum())}/{int(y.sum())}, glue rejected {int((p[y == 0] < .5).sum())}/{int((1 - y).sum())}")
    for i in np.flatnonzero((p >= 0.5) != (y == 1)):
        print(f"   misjudged: {names[i][0]} #{names[i][1]:03d}  labelled {'real' if y[i] else 'glue'}, p_real {p[i]:.2f}")
    lr.fit(Z, y)
    Path(a.output).write_text(json.dumps(dict(features=list(real100.FEATURES), mean=mean.tolist(), std=std.tolist(), coef=lr.coef_[0].tolist(),
                                              intercept=float(lr.intercept_[0]), cut=0.5, n=int(len(y)), n_real=int(y.sum()),
                                              source=str(a.labels)), indent=1))
    print("wrote", a.output)


if __name__ == "__main__":
    main()
