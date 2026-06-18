import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score #split for training, cross val for checking how good model is
from sklearn.preprocessing import StandardScaler #scales everything so same range
from sklearn.linear_model import LogisticRegression, LinearRegression #logistic --> clasfication, linear is prediction 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor)
from sklearn.metrics import (classification_report, confusion_matrix, ConfusionMatrixDisplay, mean_absolute_error, r2_score)

# first we loaded our data from ""
df = pd.read_csv("dataset_heel.csv")

# because we could not find any dataset that contained both DXA T-scores and QUS features, we had to fabricate the DXA t-scores using # a published calcaneal QUS → femoral neck DXA correlation: r = 0.55 (PMC10577968) with systematic offset of -0.4 (QUS reads higher than DXA), # in order to use coefficent, we don't want the same perfect data for each data point, as in real life there are some variations # r^2 used to find how much QUS features can explain DXA accurately r^2 = 0.3025 // 30% explainable, 70% not explainable # use residual formula: # residual_std = sqrt(unexplained variance fraction) × DXA population std # = sqrt(0.70) × 1.25 # = 0.8367 × 1.25 # = 1.047 --> we decided to do half of this because for demo, data can get messy 
# #fabricating our DXA scores -- for DEMO only, in future we will use InterSystem database or clinical trials (replace y_reg with real DXA data)

# r=0.55 from published calcaneal QUS → femoral neck DXA correlation (PMC10577968) we chose to use a noise_std of 0.6 (in betweent the r^2 residual calculations)
# to not get the exact same values each time 

def fabricate_dxa_tscore(qus_t, r=0.55, offset=-0.4, noise_std=0.6, seed=42):
    np.random.seed(seed) #same numbers every run
    noise = np.random.normal(0, noise_std, size=len(qus_t)) #random variation
    return (r * qus_t) + offset + noise #final formula we will use 

df["DXA_T"] = fabricate_dxa_tscore(df["QUS_T"].values) #creates the new feature column!

# the 3 classes labeling for diagnosis (we will comapre our QUS to this!) -- covnerting tscore to diagnosis
def tscore_to_diagnosis(t):
    if t > -1.0:    return "normal"
    elif t > -2.5:  return "osteopenia"
    else:           return "osteoporosis"

df["diagnosis"] = df["DXA_T"].apply(tscore_to_diagnosis)

#the features we are using from our dataset 
features = ["BUA", "SOS", "Age", "sexValue", "BMI"]


X = df[features].values
y_cls = df["diagnosis"].values   # this is the 3-class classification target
y_reg = df["DXA_T"].values       # this is our continuous regression target (prediction modeling -- accuracy)

# training the model here (splitting 80% test and 20% train)
X_cls_train, X_cls_test, y_cls_train, y_cls_test = train_test_split(
    X, y_cls, test_size=0.2, random_state=42, stratify=y_cls)

X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
    X, y_reg, test_size=0.2, random_state=42)

#scaling 
scaler_cls = StandardScaler()
scaler_reg = StandardScaler()

X_cls_train_s = scaler_cls.fit_transform(X_cls_train)
X_cls_test_s  = scaler_cls.transform(X_cls_test)
X_reg_train_s = scaler_reg.fit_transform(X_reg_train)
X_reg_test_s  = scaler_reg.transform(X_reg_test)

class_names = ["normal", "osteopenia", "osteoporosis"]


#classifying the QUS scores 
print("\n" + "=" * 60)
print("CLASSIFICATION — normal / osteopenia / osteoporosis")
print("=" * 60)

# all the classfication models we are using 
clf_models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree":DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "Gradient Boosting":GradientBoostingClassifier(n_estimators=200, random_state=42),
}

best_clf_score = 0
fitted_clf = {}

for name, model in clf_models.items(): #loop through each model
    cv = cross_val_score(model, X_cls_train_s, y_cls_train, cv=5, scoring="accuracy")
    model.fit(X_cls_train_s, y_cls_train)
    y_pred = model.predict(X_cls_test_s)
    acc = (y_pred == y_cls_test).mean()
    print(f"\n  {name}")
    print(f"    CV Acc={cv.mean():.4f} ± {cv.std():.4f}  |  Test Acc={acc:.4f}")
    print(classification_report(y_cls_test, y_pred, target_names=class_names, digits=3))
    fitted_clf[name] = model
    if acc > best_clf_score:
        best_clf_score, best_clf_name = acc, name

print(f"\n  Best classifier: {best_clf_name}  (Acc={best_clf_score:.4f})")

# QUS signals to DXA! this is where our prediction comes in 

print("\n" + "=" * 60)
print("REGRESSION — QUS signals → predicted DXA T-score")
print("=" * 60)

#the regression models we are using
reg_models = {
    "Linear Regression": LinearRegression(),
    "Random Forest Regressor": RandomForestRegressor(n_estimators=200, random_state=42),
    "Gradient Boosting Reg": GradientBoostingRegressor(n_estimators=200, random_state=42),
}

best_reg_mae = 999
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

print(f"\n  Best regressor: {best_reg_name}  (MAE={best_reg_mae:.4f})")
print(f"  Clinical note: MAE < 0.5 is considered diagnostically useful")

best_reg_model, best_reg_preds = fitted_reg[best_reg_name]
pred_dx = [tscore_to_diagnosis(t) for t in best_reg_preds]
true_dx = [tscore_to_diagnosis(t) for t in y_reg_test]

print(f"\n  QUS signals → DXA T-score → WHO threshold → diagnosis:")
print(classification_report(true_dx, pred_dx, target_names=class_names, digits=3))

# visualizing the data trained corrrelations 
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Calcaneus QUS → DXA Equivalent Pipeline", fontsize=15, fontweight="bold")

colors = {"normal": "#2ecc71", "osteopenia": "#f39c12", "osteoporosis": "#e74c3c"}

# graph 1: DXA T-score distribution --> checking if our fabricated is medically as accurate as it can be
ax = axes[0, 0]
for cls in class_names:
    ax.hist(df[df["diagnosis"] == cls]["DXA_T"], bins=20,
            alpha=0.6, label=cls, color=colors[cls])
ax.axvline(-1.0, color="black", linestyle="--", linewidth=1)
ax.axvline(-2.5, color="black", linestyle=":",  linewidth=1)
ax.set_title("DXA T-score by diagnosis")
ax.set_xlabel("DXA T-score")
ax.legend()

# graph 2: confusion matrix -- which model is best
ax = axes[0, 1]
cm = confusion_matrix(y_cls_test, fitted_clf[best_clf_name].predict(X_cls_test_s),
                      labels=class_names)
ConfusionMatrixDisplay(cm, display_labels=class_names).plot(
    ax=ax, colorbar=False, cmap="Blues")
ax.set_title(f"Confusion matrix\n{best_clf_name}")
ax.tick_params(axis="x", rotation=15)

# graph 3: which feature contributes the most to prediction? (we originally though only T-score does!) 
ax = axes[1, 0]
rf = fitted_clf["Random Forest"]
pd.Series(rf.feature_importances_, index=features).sort_values().plot(
    kind="barh", ax=ax, color="#3498db")
ax.set_title("Feature importance (Random Forest)")
ax.set_xlabel("Importance")

# our linear regression predicted vs actual DXA T-score!
ax = axes[1, 1]
_, preds = fitted_reg[best_reg_name]
ax.scatter(y_reg_test, preds, alpha=0.4, s=15, color="#9b59b6")
lim = [min(y_reg_test.min(), preds.min()) - 0.2,
       max(y_reg_test.max(), preds.max()) + 0.2]
ax.plot(lim, lim, "k--", linewidth=1)
ax.axvline(-1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
ax.axvline(-2.5, color="gray", linestyle=":",  linewidth=0.8, alpha=0.6)
ax.axhline(-1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.3)
ax.axhline(-2.5, color="gray", linestyle=":",  linewidth=0.8, alpha=0.3)
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_title(f"Predicted vs Actual DXA T-score\n{best_reg_name}  MAE={best_reg_mae:.3f}")
ax.set_xlabel("Actual DXA T-score")
ax.set_ylabel("Predicted DXA T-score")

plt.tight_layout()
plt.savefig("osteoporosis_pipeline_final.png", dpi=150, bbox_inches="tight")
print("\nPlot saved → osteoporosis_pipeline_final.png")
plt.show()

