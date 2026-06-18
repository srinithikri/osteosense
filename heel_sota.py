import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, StackingClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             ConfusionMatrixDisplay, mean_absolute_error,
                             r2_score, f1_score)
import xgboost as xgb
import catboost as cb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

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

# ── 1. Load data ───────────────────────────────────────────────────────────────
df = pd.read_csv("dataset_heel.csv")
df["QUS_T"] = pd.to_numeric(df["QUS_T"], errors="coerce")
df = df.dropna(subset=["QUS_T"]).reset_index(drop=True)

df["diagnosis"] = df["QUS_T"].apply(tscore_to_diagnosis)

print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)
print(f"Shape: {df.shape}")
print(f"\n3-class diagnosis counts:\n{df['diagnosis'].value_counts()}")
print(f"\nQUS_T range: {df['QUS_T'].min():.2f} to {df['QUS_T'].max():.2f}")

# ── 2. Encode 3-class target ───────────────────────────────────────────────────
le = LabelEncoder()
le.fit(LABEL_ORDER)
y_cls = le.transform(df["diagnosis"])   # 0=normal 1=osteopenia 2=osteoporosis

# ── 3. Feature sets ────────────────────────────────────────────────────────────
raw_features  = ["BUA", "SOS", "Age", "sexValue", "BMI"]
full_features = ["BUA", "SOS", "QUS_T", "Age", "sexValue", "BMI"]

X_raw  = df[raw_features].values
X_full = df[full_features].values
y_reg  = df["QUS_T"].values

# ── 4. Train/test split ────────────────────────────────────────────────────────
X_raw_train,  X_raw_test,  y_train, y_test = train_test_split(
    X_raw,  y_cls, test_size=0.2, random_state=42, stratify=y_cls)

X_full_train, X_full_test, _, _ = train_test_split(
    X_full, y_cls, test_size=0.2, random_state=42, stratify=y_cls)

X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
    X_raw, y_reg, test_size=0.2, random_state=42)

print(f"\nTrain size: {len(X_raw_train)}  |  Test size: {len(X_raw_test)}")

# ── 5. Scale features ──────────────────────────────────────────────────────────
scaler_raw  = StandardScaler()
scaler_full = StandardScaler()
scaler_reg  = StandardScaler()

X_raw_train_s  = scaler_raw.fit_transform(X_raw_train)
X_raw_test_s   = scaler_raw.transform(X_raw_test)
X_full_train_s = scaler_full.fit_transform(X_full_train)
X_full_test_s  = scaler_full.transform(X_full_test)
X_reg_train_s  = scaler_reg.fit_transform(X_reg_train)
X_reg_test_s   = scaler_reg.transform(X_reg_test)

# ══════════════════════════════════════════════════════════════════════════════
# PART A — STATE-OF-THE-ART 3-CLASS CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PART A: SOTA 3-CLASS CLASSIFICATION — normal / osteopenia / osteoporosis")
print("=" * 60)

cv_splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fitted_clf  = {}
clf_scores  = {}
best_clf_name, best_clf_score = None, 0

# ── A1. Optuna-tuned XGBoost ───────────────────────────────────────────────────
print("\n[1/4] Tuning XGBoost with Optuna (50 trials)...")

def xgb_objective(trial):
    params = dict(
        n_estimators     = trial.suggest_int("n_estimators", 100, 600),
        max_depth        = trial.suggest_int("max_depth", 3, 8),
        learning_rate    = trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        subsample        = trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.6, 1.0),
        min_child_weight = trial.suggest_int("min_child_weight", 1, 10),
        reg_alpha        = trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        reg_lambda       = trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        eval_metric      = "mlogloss",
        random_state     = 42,
        n_jobs           = -1,
    )
    model = xgb.XGBClassifier(**params)
    return cross_val_score(model, X_raw_train_s, y_train,
                           cv=cv_splitter, scoring="f1_weighted").mean()

xgb_study = optuna.create_study(direction="maximize")
xgb_study.optimize(xgb_objective, n_trials=50, show_progress_bar=False)

best_xgb = xgb.XGBClassifier(
    **xgb_study.best_params,
    eval_metric="mlogloss",
    random_state=42,
    n_jobs=-1,
)
best_xgb.fit(X_raw_train_s, y_train)
y_pred_xgb = best_xgb.predict(X_raw_test_s)
f1_xgb = f1_score(y_test, y_pred_xgb, average="weighted")
print(f"  XGBoost (Optuna)  best CV F1={xgb_study.best_value:.4f}  Test F1={f1_xgb:.4f}")
print(classification_report(y_test, y_pred_xgb, target_names=LABEL_ORDER, digits=3))
fitted_clf["XGBoost (Optuna)"] = best_xgb
clf_scores["XGBoost (Optuna)"] = f1_xgb
if f1_xgb > best_clf_score:
    best_clf_score, best_clf_name = f1_xgb, "XGBoost (Optuna)"

# ── A2. Optuna-tuned CatBoost ─────────────────────────────────────────────────
print("\n[2/4] Tuning CatBoost with Optuna (50 trials)...")

def cat_objective(trial):
    params = dict(
        iterations          = trial.suggest_int("iterations", 100, 600),
        depth               = trial.suggest_int("depth", 3, 8),
        learning_rate       = trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        l2_leaf_reg         = trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
        bagging_temperature = trial.suggest_float("bagging_temperature", 0.0, 1.0),
        random_strength     = trial.suggest_float("random_strength", 1e-3, 10.0, log=True),
        loss_function       = "MultiClass",
        random_seed         = 42,
        verbose             = False,
    )
    model = cb.CatBoostClassifier(**params)
    return cross_val_score(model, X_raw_train_s, y_train,
                           cv=cv_splitter, scoring="f1_weighted").mean()

cat_study = optuna.create_study(direction="maximize")
cat_study.optimize(cat_objective, n_trials=50, show_progress_bar=False)

best_cat = cb.CatBoostClassifier(
    **cat_study.best_params,
    loss_function="MultiClass",
    random_seed=42,
    verbose=False,
)
best_cat.fit(X_raw_train_s, y_train)
y_pred_cat = best_cat.predict(X_raw_test_s).flatten()
f1_cat = f1_score(y_test, y_pred_cat, average="weighted")
print(f"  CatBoost (Optuna)  best CV F1={cat_study.best_value:.4f}  Test F1={f1_cat:.4f}")
print(classification_report(y_test, y_pred_cat, target_names=LABEL_ORDER, digits=3))
fitted_clf["CatBoost (Optuna)"] = best_cat
clf_scores["CatBoost (Optuna)"] = f1_cat
if f1_cat > best_clf_score:
    best_clf_score, best_clf_name = f1_cat, "CatBoost (Optuna)"

# ── A3. TabPFN (pre-trained transformer for small tabular data) ────────────────
print("\n[3/4] TabPFN (pre-trained transformer, no training loop)...")
tabpfn_available = False
try:
    from tabpfn import TabPFNClassifier
    tabpfn = TabPFNClassifier(device="cpu", N_ensemble_configurations=32)
    tabpfn.fit(X_raw_train_s, y_train)
    y_pred_pfn = tabpfn.predict(X_raw_test_s)
    f1_pfn = f1_score(y_test, y_pred_pfn, average="weighted")
    cv_pfn = cross_val_score(tabpfn, X_raw_train_s, y_train,
                             cv=cv_splitter, scoring="f1_weighted")
    print(f"  TabPFN  CV F1={cv_pfn.mean():.4f} ± {cv_pfn.std():.4f}  Test F1={f1_pfn:.4f}")
    print(classification_report(y_test, y_pred_pfn, target_names=LABEL_ORDER, digits=3))
    fitted_clf["TabPFN"] = tabpfn
    clf_scores["TabPFN"] = f1_pfn
    if f1_pfn > best_clf_score:
        best_clf_score, best_clf_name = f1_pfn, "TabPFN"
    tabpfn_available = True
except ImportError:
    print("  TabPFN not installed — run: pip install tabpfn")
    print("  Skipping TabPFN.")

# ── A4. Stacking ensemble ──────────────────────────────────────────────────────
print("\n[4/4] Stacking ensemble (XGBoost + CatBoost + RF → Logistic meta-learner)...")

base_estimators = [
    ("xgb", xgb.XGBClassifier(**xgb_study.best_params,
                               eval_metric="mlogloss", random_state=42, n_jobs=-1)),
    ("cat", cb.CatBoostClassifier(**cat_study.best_params,
                                  loss_function="MultiClass", random_seed=42, verbose=False)),
    ("rf",  RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)),
]
if tabpfn_available:
    base_estimators.append(("pfn", TabPFNClassifier(device="cpu", N_ensemble_configurations=32)))

stack = StackingClassifier(
    estimators      = base_estimators,
    final_estimator = LogisticRegression(multi_class="multinomial", max_iter=1000),
    cv              = cv_splitter,
    stack_method    = "predict_proba",
    n_jobs          = -1,
)
stack.fit(X_raw_train_s, y_train)
y_pred_stack = stack.predict(X_raw_test_s)
f1_stack = f1_score(y_test, y_pred_stack, average="weighted")
cv_stack = cross_val_score(stack, X_raw_train_s, y_train,
                           cv=cv_splitter, scoring="f1_weighted")
print(f"  Stacking  CV F1={cv_stack.mean():.4f} ± {cv_stack.std():.4f}  Test F1={f1_stack:.4f}")
print(classification_report(y_test, y_pred_stack, target_names=LABEL_ORDER, digits=3))
fitted_clf["Stacking Ensemble"] = stack
clf_scores["Stacking Ensemble"] = f1_stack
if f1_stack > best_clf_score:
    best_clf_score, best_clf_name = f1_stack, "Stacking Ensemble"

# ── Summary ────────────────────────────────────────────────────────────────────
print("\n── Model comparison (Test weighted F1) ──")
for name, score in sorted(clf_scores.items(), key=lambda x: -x[1]):
    marker = "  ← best" if name == best_clf_name else ""
    print(f"  {name:30s}  {score:.4f}{marker}")

# Compare best model with vs without QUS_T (using RF as a fair baseline)
print("\n── Raw signals vs Raw + QUS_T (RF baseline) ──")
rf_raw  = RandomForestClassifier(n_estimators=300, random_state=42)
rf_full = RandomForestClassifier(n_estimators=300, random_state=42)
rf_raw.fit(X_raw_train_s,  y_train)
rf_full.fit(X_full_train_s, y_train)
f1_raw  = f1_score(y_test, rf_raw.predict(X_raw_test_s),   average="weighted")
f1_full = f1_score(y_test, rf_full.predict(X_full_test_s), average="weighted")
print(f"  RF — Raw signals only : F1 = {f1_raw:.4f}")
print(f"  RF — With QUS_T added : F1 = {f1_full:.4f}")
print(f"  Gap                   : {f1_full - f1_raw:+.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# PART B — REGRESSION (predicting QUS T-score)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PART B: REGRESSION — Predicting QUS T-score from raw signals")
print("=" * 60)

reg_models = {
    "Linear Regression":       LinearRegression(),
    "Random Forest Regressor": RandomForestRegressor(n_estimators=200, random_state=42),
    "XGBoost Regressor":       xgb.XGBRegressor(n_estimators=300, learning_rate=0.05,
                                                 max_depth=4, random_state=42, n_jobs=-1),
}

best_reg_mae, best_reg_name = 999, None
fitted_reg = {}

for name, model in reg_models.items():
    model.fit(X_reg_train_s, y_reg_train)
    preds = model.predict(X_reg_test_s)
    mae   = mean_absolute_error(y_reg_test, preds)
    r2    = r2_score(y_reg_test, preds)
    print(f"  {name:30s}  MAE={mae:.4f}  R²={r2:.4f}")
    fitted_reg[name] = (model, preds)
    if mae < best_reg_mae:
        best_reg_mae, best_reg_name = mae, name

print(f"\n  Best regressor: {best_reg_name}  (MAE={best_reg_mae:.4f} T-score units)")

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
    "Calcaneus QUS — SOTA 3-Class Pipeline  (XGBoost + CatBoost + TabPFN + Stacking)",
    fontsize=13, fontweight="bold"
)

# C1: Model F1 comparison bar
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
            f"{score:.4f}", va="center", fontsize=8)

# C2: BUA distribution
ax = axes[0, 1]
for dx in LABEL_ORDER:
    ax.hist(df[df["diagnosis"] == dx]["BUA"], bins=20, alpha=0.6,
            label=dx, color=dx_colors[dx])
ax.set_title("BUA distribution by class")
ax.set_xlabel("BUA"); ax.set_ylabel("Count"); ax.legend()

# C3: QUS T-score distribution with thresholds
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

# C5: XGBoost feature importance
ax = axes[1, 1]
importances = pd.Series(best_xgb.feature_importances_, index=raw_features).sort_values()
importances.plot(kind="barh", ax=ax, color="#e67e22")
ax.set_title("Feature importance\n(XGBoost Optuna-tuned)")
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
plt.savefig("osteoporosis_sota_pipeline.png", dpi=150, bbox_inches="tight")
print("\nPlot saved → osteoporosis_sota_pipeline.png")
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
    if probs.ndim > 1:
        probs = probs[0]
    prob_dict = dict(zip(LABEL_ORDER, probs))

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
