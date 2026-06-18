"""
Advanced 3-class bone density classifier using:
  - HistGradientBoostingClassifier (sklearn's LightGBM-style, no extra installs)
  - RandomizedSearchCV tuning
  - Soft VotingClassifier ensemble
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import (train_test_split, cross_val_score,
                                     StratifiedKFold, RandomizedSearchCV)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor,
                              HistGradientBoostingClassifier, VotingClassifier)
from sklearn.metrics import (classification_report, confusion_matrix,
                             ConfusionMatrixDisplay, mean_absolute_error,
                             r2_score, f1_score)
from scipy.stats import randint, uniform

# ── Thresholds (Youden's Index for QUS calcaneus, consistent with app.py) ─────
QUS_OSTEOPOROSIS_THRESHOLD = -2.725
QUS_OSTEOPENIA_THRESHOLD   = -1.0
LABEL_ORDER = ["normal", "osteopenia", "osteoporosis"]

def tscore_to_diagnosis(t):
    if t >= QUS_OSTEOPENIA_THRESHOLD:
        return "normal"
    elif t >= QUS_OSTEOPOROSIS_THRESHOLD:
        return "osteopenia"
    else:
        return "osteoporosis"

# ── 1. Load & prepare data ─────────────────────────────────────────────────────
df = pd.read_csv("dataset_heel.csv")
df["QUS_T"] = pd.to_numeric(df["QUS_T"], errors="coerce")
df = df.dropna(subset=["QUS_T"]).reset_index(drop=True)
df["diagnosis"] = df["QUS_T"].apply(tscore_to_diagnosis)

print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)
print(f"Shape: {df.shape}")
print(f"\n3-class diagnosis counts:\n{df['diagnosis'].value_counts()}")

le = LabelEncoder()
le.fit(LABEL_ORDER)
y_cls = le.transform(df["diagnosis"])

raw_features = ["BUA", "SOS", "Age", "sexValue", "BMI"]
X_raw = df[raw_features].values
y_reg = df["QUS_T"].values

X_raw_train, X_raw_test, y_train, y_test = train_test_split(
    X_raw, y_cls, test_size=0.2, random_state=42, stratify=y_cls)
X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
    X_raw, y_reg, test_size=0.2, random_state=42)

scaler_raw = StandardScaler()
scaler_reg = StandardScaler()
X_raw_train_s = scaler_raw.fit_transform(X_raw_train)
X_raw_test_s  = scaler_raw.transform(X_raw_test)
X_reg_train_s = scaler_reg.fit_transform(X_reg_train)
X_reg_test_s  = scaler_reg.transform(X_reg_test)

print(f"\nTrain: {len(X_raw_train)}  |  Test: {len(X_raw_test)}")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ══════════════════════════════════════════════════════════════════════════════
# PART A — CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PART A: 3-CLASS CLASSIFICATION — normal / osteopenia / osteoporosis")
print("=" * 60)

fitted_clf = {}
clf_scores = {}
best_clf_name, best_clf_score = None, 0

# ── A1. HistGradientBoosting tuned with RandomizedSearchCV ────────────────────
print("\n[1/3] HistGradientBoostingClassifier + RandomizedSearchCV (60 iterations)...")

hgb_param_dist = {
    "max_iter":        randint(100, 600),
    "max_depth":       randint(3, 10),
    "learning_rate":   uniform(0.01, 0.29),
    "min_samples_leaf": randint(5, 50),
    "l2_regularization": uniform(0.0, 1.0),
    "max_bins":        randint(64, 255),
}

hgb_search = RandomizedSearchCV(
    HistGradientBoostingClassifier(random_state=42),
    param_distributions = hgb_param_dist,
    n_iter              = 60,
    scoring             = "f1_weighted",
    cv                  = cv,
    n_jobs              = -1,
    random_state        = 42,
    verbose             = 0,
)
hgb_search.fit(X_raw_train_s, y_train)
best_hgb = hgb_search.best_estimator_

y_pred_hgb = best_hgb.predict(X_raw_test_s)
f1_hgb = f1_score(y_test, y_pred_hgb, average="weighted")
print(f"  Best CV F1 : {hgb_search.best_score_:.4f}")
print(f"  Test F1    : {f1_hgb:.4f}")
print(f"  Best params: {hgb_search.best_params_}")
print(classification_report(y_test, y_pred_hgb, target_names=LABEL_ORDER, digits=3))
fitted_clf["HistGradBoost (tuned)"] = best_hgb
clf_scores["HistGradBoost (tuned)"] = f1_hgb
if f1_hgb > best_clf_score:
    best_clf_score, best_clf_name = f1_hgb, "HistGradBoost (tuned)"

# ── A2. Random Forest tuned with RandomizedSearchCV ───────────────────────────
print("\n[2/3] RandomForestClassifier + RandomizedSearchCV (60 iterations)...")

rf_param_dist = {
    "n_estimators":      randint(100, 500),
    "max_depth":         [None, 5, 8, 12, 16],
    "min_samples_split": randint(2, 20),
    "min_samples_leaf":  randint(1, 10),
    "max_features":      ["sqrt", "log2", 0.5, 0.7],
}

rf_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    param_distributions = rf_param_dist,
    n_iter              = 60,
    scoring             = "f1_weighted",
    cv                  = cv,
    n_jobs              = -1,
    random_state        = 42,
    verbose             = 0,
)
rf_search.fit(X_raw_train_s, y_train)
best_rf = rf_search.best_estimator_

y_pred_rf = best_rf.predict(X_raw_test_s)
f1_rf = f1_score(y_test, y_pred_rf, average="weighted")
print(f"  Best CV F1 : {rf_search.best_score_:.4f}")
print(f"  Test F1    : {f1_rf:.4f}")
print(classification_report(y_test, y_pred_rf, target_names=LABEL_ORDER, digits=3))
fitted_clf["Random Forest (tuned)"] = best_rf
clf_scores["Random Forest (tuned)"] = f1_rf
if f1_rf > best_clf_score:
    best_clf_score, best_clf_name = f1_rf, "Random Forest (tuned)"

# ── A3. Soft Voting Ensemble ───────────────────────────────────────────────────
print("\n[3/3] Soft VotingClassifier (HGB + RF + Logistic)...")

voter = VotingClassifier(
    estimators=[
        ("hgb", HistGradientBoostingClassifier(**hgb_search.best_params_, random_state=42)),
        ("rf",  RandomForestClassifier(**rf_search.best_params_, random_state=42, n_jobs=-1)),
        ("lr",  LogisticRegression(multi_class="multinomial", max_iter=1000, C=1.0)),
    ],
    voting="soft",
    n_jobs=-1,
)
voter.fit(X_raw_train_s, y_train)
y_pred_vote = voter.predict(X_raw_test_s)
f1_vote = f1_score(y_test, y_pred_vote, average="weighted")
cv_vote = cross_val_score(voter, X_raw_train_s, y_train, cv=cv, scoring="f1_weighted")
print(f"  CV F1  : {cv_vote.mean():.4f} ± {cv_vote.std():.4f}")
print(f"  Test F1: {f1_vote:.4f}")
print(classification_report(y_test, y_pred_vote, target_names=LABEL_ORDER, digits=3))
fitted_clf["Soft Voting Ensemble"] = voter
clf_scores["Soft Voting Ensemble"] = f1_vote
if f1_vote > best_clf_score:
    best_clf_score, best_clf_name = f1_vote, "Soft Voting Ensemble"

# ── Summary ────────────────────────────────────────────────────────────────────
print("\n── Model comparison (Test weighted F1) ──")
for name, score in sorted(clf_scores.items(), key=lambda x: -x[1]):
    marker = "  ← best" if name == best_clf_name else ""
    print(f"  {name:30s}  {score:.4f}{marker}")

# ══════════════════════════════════════════════════════════════════════════════
# PART B — REGRESSION
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PART B: REGRESSION — Predicting QUS T-score from raw signals")
print("=" * 60)

reg_models = {
    "Linear Regression":       LinearRegression(),
    "Random Forest Regressor": RandomForestRegressor(n_estimators=200, random_state=42),
    "HistGradBoost Regressor": HistGradientBoostingClassifier(random_state=42),  # reuse tuned params
}

# Use a proper regressor for HGB
from sklearn.ensemble import HistGradientBoostingRegressor
reg_models = {
    "Linear Regression":       LinearRegression(),
    "Random Forest Regressor": RandomForestRegressor(**rf_search.best_params_, random_state=42),
    "HistGradBoost Regressor": HistGradientBoostingRegressor(**hgb_search.best_params_, random_state=42),
}

best_reg_mae, best_reg_name = 999, None
fitted_reg = {}

for name, model in reg_models.items():
    model.fit(X_reg_train_s, y_reg_train)
    preds = model.predict(X_reg_test_s)
    mae   = mean_absolute_error(y_reg_test, preds)
    r2    = r2_score(y_reg_test, preds)
    print(f"  {name:35s}  MAE={mae:.4f}  R²={r2:.4f}")
    fitted_reg[name] = (model, preds)
    if mae < best_reg_mae:
        best_reg_mae, best_reg_name = mae, name

print(f"\n  Best regressor: {best_reg_name}  (MAE={best_reg_mae:.4f})")

best_reg_model, best_reg_preds = fitted_reg[best_reg_name]
pred_dx = [tscore_to_diagnosis(t) for t in best_reg_preds]
true_dx = [tscore_to_diagnosis(t) for t in y_reg_test]
print(f"\n  3-class via regression → threshold:")
print(classification_report(true_dx, pred_dx, target_names=LABEL_ORDER))

# ══════════════════════════════════════════════════════════════════════════════
# PART C — VISUALIZATIONS
# ══════════════════════════════════════════════════════════════════════════════
dx_colors = {"normal": "#2ecc71", "osteopenia": "#f39c12", "osteoporosis": "#e74c3c"}

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle(
    "Calcaneus QUS — HistGradientBoosting + Soft Voting Ensemble (no extra installs)",
    fontsize=13, fontweight="bold"
)

# C1: Model F1 bar
ax = axes[0, 0]
names  = list(clf_scores.keys())
scores = [clf_scores[n] for n in names]
colors = ["#e74c3c" if n == best_clf_name else "#3498db" for n in names]
bars   = ax.barh(names, scores, color=colors)
ax.set_xlim(0, 1)
ax.set_xlabel("Weighted F1")
ax.set_title("Model comparison (red = best)")
for bar, score in zip(bars, scores):
    ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
            f"{score:.4f}", va="center", fontsize=9)

# C2: BUA distribution
ax = axes[0, 1]
for dx in LABEL_ORDER:
    ax.hist(df[df["diagnosis"] == dx]["BUA"], bins=20, alpha=0.6,
            label=dx, color=dx_colors[dx])
ax.set_title("BUA distribution by class")
ax.set_xlabel("BUA"); ax.set_ylabel("Count"); ax.legend()

# C3: QUS T-score with thresholds
ax = axes[0, 2]
for dx in LABEL_ORDER:
    ax.hist(df[df["diagnosis"] == dx]["QUS_T"], bins=20, alpha=0.6,
            label=dx, color=dx_colors[dx])
ax.axvline(QUS_OSTEOPENIA_THRESHOLD,   color="gray",    linestyle="--", linewidth=1.2)
ax.axvline(QUS_OSTEOPOROSIS_THRESHOLD, color="#cc0000", linestyle=":",  linewidth=1.2)
ax.set_title("QUS T-score distribution\nwith thresholds")
ax.set_xlabel("QUS T-score"); ax.legend(fontsize=8)

# C4: Confusion matrix — best model
ax = axes[1, 0]
cm = confusion_matrix(y_test, fitted_clf[best_clf_name].predict(X_raw_test_s))
disp = ConfusionMatrixDisplay(cm, display_labels=LABEL_ORDER)
disp.plot(ax=ax, colorbar=False, cmap="Blues")
ax.set_title(f"Confusion matrix\n{best_clf_name}")
ax.tick_params(axis="x", rotation=15)

# C5: HGB feature importance (via permutation proxy — HGB doesn't expose feature_importances_)
ax = axes[1, 1]
rf_imp = pd.Series(best_rf.feature_importances_, index=raw_features).sort_values()
rf_imp.plot(kind="barh", ax=ax, color="#e67e22")
ax.set_title("Feature importance\n(Random Forest tuned)")
ax.set_xlabel("Importance")

# C6: Regression predicted vs actual
ax = axes[1, 2]
_, preds = fitted_reg[best_reg_name]
ax.scatter(y_reg_test, preds, alpha=0.4, s=15, color="#9b59b6")
lim = [min(y_reg_test.min(), preds.min()) - 0.2,
       max(y_reg_test.max(), preds.max()) + 0.2]
ax.plot(lim, lim, "k--", linewidth=1)
ax.axvline(QUS_OSTEOPENIA_THRESHOLD,   color="gray",    linestyle="--", linewidth=0.8, alpha=0.7)
ax.axvline(QUS_OSTEOPOROSIS_THRESHOLD, color="#cc0000", linestyle=":",  linewidth=0.8, alpha=0.7)
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_title(f"Regression: predicted vs actual\n{best_reg_name}  MAE={best_reg_mae:.3f}")
ax.set_xlabel("Actual QUS T-score"); ax.set_ylabel("Predicted QUS T-score")

plt.tight_layout()
plt.savefig("osteoporosis_hgb_pipeline.png", dpi=150, bbox_inches="tight")
print("\nPlot saved → osteoporosis_hgb_pipeline.png")
plt.show()

# ══════════════════════════════════════════════════════════════════════════════
# PART D — SHOE DEVICE SIMULATOR
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PART D: SHOE DEVICE SIMULATOR")
print("=" * 60)
print(f"Classifier : {best_clf_name}")
print(f"Regressor  : {best_reg_name}\n")

best_clf_model = fitted_clf[best_clf_name]

def predict_from_shoe(BUA, SOS, age, sex, bmi):
    """sex: 0 = female, 1 = male"""
    raw_input  = np.array([[BUA, SOS, age, sex, bmi]])
    raw_scaled = scaler_raw.transform(raw_input)
    reg_scaled = scaler_reg.transform(raw_input)

    pred_tscore = best_reg_model.predict(reg_scaled)[0]
    diagnosis   = tscore_to_diagnosis(pred_tscore)
    probs       = best_clf_model.predict_proba(raw_scaled)[0]
    prob_dict   = dict(zip(LABEL_ORDER, probs))

    print(f"  Input  →  BUA={BUA}  SOS={SOS}  Age={age}  Sex={'M' if sex==1 else 'F'}  BMI={bmi}")
    print(f"  Predicted QUS T-score  : {pred_tscore:+.2f}")
    print(f"  Diagnosis (via T-score): {diagnosis.upper()}")
    for cls in LABEL_ORDER:
        p   = prob_dict[cls]
        bar = "█" * int(p * 30) + "░" * (30 - int(p * 30))
        print(f"  {cls:15s} {p:.1%}  [{bar}]")
    print()

print("── Patient 1: Younger woman, healthy signals ──")
predict_from_shoe(BUA=70, SOS=1500, age=55, sex=0, bmi=24)

print("── Patient 2: Older woman, borderline signals ──")
predict_from_shoe(BUA=52, SOS=1360, age=72, sex=0, bmi=22)

print("── Patient 3: Older woman, low signals (high risk) ──")
predict_from_shoe(BUA=38, SOS=1280, age=80, sex=0, bmi=19)
