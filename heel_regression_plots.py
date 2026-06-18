"""
Classification performance plots for the heel regression pipeline.
The regression predicts QUS T-score, which is then thresholded into:
  normal / osteopenia / osteoporosis

All plots focus on whether the final 3-class output is correct.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import (classification_report, confusion_matrix,
                             ConfusionMatrixDisplay, f1_score)

# ── Thresholds ─────────────────────────────────────────────────────────────────
QUS_OSTEOPOROSIS_THRESHOLD = -2.725
QUS_OSTEOPENIA_THRESHOLD   = -1.0
LABEL_ORDER = ["normal", "osteopenia", "osteoporosis"]
DX_COLORS   = {"normal": "#2ecc71", "osteopenia": "#f39c12", "osteoporosis": "#e74c3c"}

def tscore_to_diagnosis(t):
    if t >= QUS_OSTEOPENIA_THRESHOLD:     return "normal"
    elif t >= QUS_OSTEOPOROSIS_THRESHOLD: return "osteopenia"
    else:                                 return "osteoporosis"

# ── Load & prepare ─────────────────────────────────────────────────────────────
df = pd.read_csv("dataset_heel.csv")
df["QUS_T"] = pd.to_numeric(df["QUS_T"], errors="coerce")
df = df.dropna(subset=["QUS_T"]).reset_index(drop=True)
df["diagnosis"] = df["QUS_T"].apply(tscore_to_diagnosis)

raw_features = ["BUA", "SOS", "Age", "sexValue", "BMI"]
X = df[raw_features].values
y = df["QUS_T"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# ── Train regressors, convert predictions → 3-class labels ────────────────────
reg_models = {
    "Linear Regression":  LinearRegression(),
    "Random Forest":      RandomForestRegressor(n_estimators=200, random_state=42),
    "Gradient Boosting":  GradientBoostingRegressor(n_estimators=200, random_state=42),
}

true_labels = [tscore_to_diagnosis(t) for t in y_test]
true_tscores = y_test

results = {}
for name, model in reg_models.items():
    model.fit(X_train_s, y_train)
    pred_tscores = model.predict(X_test_s)
    pred_labels  = [tscore_to_diagnosis(t) for t in pred_tscores]
    f1 = f1_score(true_labels, pred_labels, labels=LABEL_ORDER, average="weighted")
    results[name] = {
        "model":        model,
        "pred_tscores": pred_tscores,
        "pred_labels":  pred_labels,
        "f1":           f1,
    }
    print(f"{name:25s}  F1={f1:.4f}")
    print(classification_report(true_labels, pred_labels, labels=LABEL_ORDER))

best_name = max(results, key=lambda n: results[n]["f1"])
best      = results[best_name]
print(f"Best: {best_name}  F1={best['f1']:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Confusion matrices for all 3 models
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(17, 5))
fig.suptitle("Confusion matrices — regression → threshold → 3-class diagnosis",
             fontsize=13, fontweight="bold")

for ax, (name, res) in zip(axes, results.items()):
    cm = confusion_matrix(true_labels, res["pred_labels"], labels=LABEL_ORDER)
    disp = ConfusionMatrixDisplay(cm, display_labels=LABEL_ORDER)
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"{name}\nWeighted F1 = {res['f1']:.3f}")
    ax.tick_params(axis="x", rotation=15)

plt.tight_layout()
plt.savefig("cls_plot1_confusion_matrices.png", dpi=150, bbox_inches="tight")
print("Saved → cls_plot1_confusion_matrices.png")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Per-class precision / recall / F1 bar chart (best model)
# ══════════════════════════════════════════════════════════════════════════════
from sklearn.metrics import precision_score, recall_score

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle(f"Per-class precision / recall / F1 — {best_name}", fontsize=13, fontweight="bold")

metrics = {
    "Precision": precision_score(true_labels, best["pred_labels"], labels=LABEL_ORDER, average=None),
    "Recall":    recall_score(true_labels,    best["pred_labels"], labels=LABEL_ORDER, average=None),
    "F1":        f1_score(true_labels,        best["pred_labels"], labels=LABEL_ORDER, average=None),
}

for ax, (metric_name, values) in zip(axes, metrics.items()):
    bars = ax.bar(LABEL_ORDER, values,
                  color=[DX_COLORS[d] for d in LABEL_ORDER],
                  edgecolor="white", alpha=0.85)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.2f}", ha="center", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.15)
    ax.set_ylabel(metric_name)
    ax.set_title(metric_name)
    ax.axhline(0.8, color="gray", linestyle="--", linewidth=0.8, alpha=0.6, label="0.8 target")
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("cls_plot2_per_class_metrics.png", dpi=150, bbox_inches="tight")
print("Saved → cls_plot2_per_class_metrics.png")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Where do misclassifications land on the T-score axis?
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 5))

correct_mask = [t == p for t, p in zip(true_labels, best["pred_labels"])]
pred_t = best["pred_tscores"]

# Plot all test points on the T-score axis, coloured by true class
for i, (actual_t, pred_t_val, true_dx, is_correct) in enumerate(
        zip(true_tscores, pred_t, true_labels, correct_mask)):
    marker = "o" if is_correct else "X"
    size   = 30 if is_correct else 80
    ax.scatter(actual_t, pred_t_val,
               color=DX_COLORS[true_dx],
               marker=marker, s=size,
               alpha=0.6 if is_correct else 0.9,
               edgecolors="black" if not is_correct else "none",
               linewidths=0.8)

# Threshold lines
ax.axvline(QUS_OSTEOPENIA_THRESHOLD,   color="gray",    linestyle="--", linewidth=1.5,
           label="−1.0 threshold (actual)")
ax.axvline(QUS_OSTEOPOROSIS_THRESHOLD, color="#cc0000", linestyle="--", linewidth=1.5,
           label="−2.725 threshold (actual)")
ax.axhline(QUS_OSTEOPENIA_THRESHOLD,   color="gray",    linestyle=":",  linewidth=1.5,
           label="−1.0 threshold (predicted)")
ax.axhline(QUS_OSTEOPOROSIS_THRESHOLD, color="#cc0000", linestyle=":",  linewidth=1.5,
           label="−2.725 threshold (predicted)")

# Diagonal
lim = [min(true_tscores.min(), pred_t.min()) - 0.2,
       max(true_tscores.max(), pred_t.max()) + 0.2]
ax.plot(lim, lim, "k--", linewidth=0.8, alpha=0.4)
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("Actual QUS T-score")
ax.set_ylabel("Predicted QUS T-score")
ax.set_title(f"Misclassifications on T-score axis — {best_name}\n"
             f"Circle = correct class  |  ✕ = wrong class  |  colour = true diagnosis")

patches = [mpatches.Patch(color=DX_COLORS[d], label=d) for d in LABEL_ORDER]
cross   = plt.scatter([], [], marker="X", color="black", s=60, label="Misclassified")
circle  = plt.scatter([], [], marker="o", color="gray",  s=20, alpha=0.6, label="Correct")
ax.legend(handles=patches + [circle, cross], fontsize=8, loc="upper left")

plt.tight_layout()
plt.savefig("cls_plot3_misclassification_map.png", dpi=150, bbox_inches="tight")
print("Saved → cls_plot3_misclassification_map.png")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Model F1 comparison bar
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 4))
names  = list(results.keys())
scores = [results[n]["f1"] for n in names]
colors = ["#e74c3c" if n == best_name else "#3498db" for n in names]
bars   = ax.barh(names, scores, color=colors, edgecolor="white")
ax.set_xlim(0, 1.1)
ax.set_xlabel("Weighted F1 (3-class classification)")
ax.set_title("Classification performance — regression → threshold pipeline\n(red = best)")
for bar, score in zip(bars, scores):
    ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
            f"{score:.4f}", va="center", fontsize=11, fontweight="bold")
plt.tight_layout()
plt.savefig("cls_plot4_model_comparison.png", dpi=150, bbox_inches="tight")
print("Saved → cls_plot4_model_comparison.png")

plt.show()
print("\nAll 4 plots saved.")
