import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# ── 1. Load data ──────────────────────────────────────────────────────────────
df = pd.read_csv("patient_data_cleaned.csv")
print("Shape:", df.shape)
print(df.head())
print("\nDiagnosis counts:\n", df["diagnosis"].value_counts())
print("\nT-score stats:\n", df["tscore"].describe())

# ── 2. Encode target ──────────────────────────────────────────────────────────
label_order = ["normal", "osteopenia", "osteoporosis"]
le = LabelEncoder()
le.fit(label_order)
df["label"] = le.transform(df["diagnosis"])

X = df[["tscore"]].values
y = df["label"].values

# ── 3. Train / test split ─────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain: {len(X_train)}  Test: {len(X_test)}")

# ── 4. Train three models ─────────────────────────────────────────────────────
models = {
    "Logistic Regression": LogisticRegression(multi_class="multinomial", max_iter=500),
    "Decision Tree":       DecisionTreeClassifier(max_depth=4, random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
}

print("\n── Cross-validation accuracy (5-fold) ──")
best_model, best_score = None, 0
for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
    print(f"  {name:25s}  mean={scores.mean():.4f}  std={scores.std():.4f}")
    if scores.mean() > best_score:
        best_score, best_model = scores.mean(), (name, model)

# ── 5. Evaluate ALL models on held-out test set ───────────────────────────────
fitted_models = {}
predictions = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    fitted_models[name] = model
    predictions[name] = y_pred
    print(f"\n── Test-set results: {name} ──")
    print(classification_report(y_test, y_pred, target_names=label_order))

# ── 6. Visualizations ─────────────────────────────────────────────────────────
model_names = list(models.keys())
colors = {"normal": "#2ecc71", "osteopenia": "#f39c12", "osteoporosis": "#e74c3c"}
color_map = colors

fig, axes = plt.subplots(3, 3, figsize=(16, 14))
fig.suptitle("Bone Density Predictive Model — All Models", fontsize=14, fontweight="bold")

tscore_range = np.linspace(df["tscore"].min() - 0.1, df["tscore"].max() + 0.1, 500).reshape(-1, 1)
jitter = np.random.default_rng(0).uniform(-0.3, 0.3, len(df))

for row, name in enumerate(model_names):
    model = fitted_models[name]
    y_pred = predictions[name]

    # Col 0: T-score distribution
    for diag in label_order:
        subset = df[df["diagnosis"] == diag]["tscore"]
        axes[row, 0].hist(subset, bins=15, alpha=0.6, label=diag, color=colors[diag])
    axes[row, 0].axvline(-1.0, color="gray", linestyle="--", linewidth=1, label="threshold −1.0")
    axes[row, 0].axvline(-2.5, color="gray", linestyle=":",  linewidth=1, label="threshold −2.5")
    axes[row, 0].set_title(f"{name}\nT-score Distribution")
    axes[row, 0].set_xlabel("T-score")
    axes[row, 0].set_ylabel("Count")
    axes[row, 0].legend(fontsize=7)

    # Col 1: Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_order)
    disp.plot(ax=axes[row, 1], colorbar=False, cmap="Blues")
    axes[row, 1].set_title(f"{name}\nConfusion Matrix")
    axes[row, 1].tick_params(axis="x", rotation=15)

    # Col 2: Decision boundary
    preds = model.predict(tscore_range)
    pred_labels = le.inverse_transform(preds)
    step = tscore_range[1, 0] - tscore_range[0, 0]
    for ts, pl in zip(tscore_range.flatten(), pred_labels):
        axes[row, 2].axvspan(ts, ts + step, color=color_map[pl], alpha=0.25)
    for diag in label_order:
        mask = df["diagnosis"] == diag
        axes[row, 2].scatter(df.loc[mask, "tscore"], jitter[mask], label=diag,
                             color=colors[diag], s=20, alpha=0.7, edgecolors="none")
    axes[row, 2].set_title(f"{name}\nDecision Boundary")
    axes[row, 2].set_xlabel("T-score")
    axes[row, 2].set_yticks([])
    axes[row, 2].legend(fontsize=7)

plt.tight_layout()
plt.savefig("bone_density_model.png", dpi=150, bbox_inches="tight")
print("\nPlot saved → bone_density_model.png")
plt.show()

# ── 7. Simple prediction function (uses best model) ───────────────────────────
best_name, best_model = best_model
best_model.fit(X_train, y_train)

def predict_diagnosis(tscore_value):
    pred = best_model.predict([[tscore_value]])[0]
    prob = best_model.predict_proba([[tscore_value]])[0]
    label = le.inverse_transform([pred])[0]
    prob_dict = dict(zip(label_order, prob))
    print(f"\nT-score {tscore_value:+.2f}  →  {label.upper()}  (via {best_name})")
    for cls, p in prob_dict.items():
        bar = "█" * int(p * 30)
        print(f"  {cls:15s} {p:.1%}  {bar}")

predict_diagnosis(-0.5)
predict_diagnosis(-1.8)
predict_diagnosis(-2.9)
