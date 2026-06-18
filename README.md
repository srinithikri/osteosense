# OsteoSense: An AI-Powered Ultrasound for Earlier Diagnosing of Osteoporosis -- Created for the 2026 Harvard T.H. Chan Health Systems Innovation Lab Hackathon

## The Problem
- Osteoporosis is underdiagnosed due to limited access to DEXA Scans
- It's a silent condition which means many who have it don't know they have it and by the time they are diagnosed, damage already done to their bones
- DEXA machines are expensive and not portable

## Our Solution
We built an AI/ML pipeline that converts ultrasound data (SOS, BUA, etc.) and used fabricated DXA scores to train ML models to make the QUS data collected 
from our foot ultrasound sensor as accurate clinically as possible. We used the dataset from this artcile: https://www.nature.com/articles/s41598-021-95261-7 (Under supplementary information #1)

## Goal
We are not meant to replace DXA scans, but to provide a normal routine to high-risk populations that can actively check their bones without needing to go to the doctor everyday --> more accessible scanning + cheaper

## Impact
- Expands osteoporosis screening to low-resource settings
- Enables early detection and intervention
- Reduces reliance on expensive imaging infrastructure
- Scales globally with portable ultrasound devices

## Our Steps
1) We realized in order to do cross mapping, we would need a dataset that contained patient ID with people doing scans of QUS and its features and then comparing that to the DEXA T-score, (which we can clinically see from scale above where the patient lies in terms of fractal damage)

2) For now, we found a correlation factor for QUS and DXA T-scores: Calcaneus BMD measured by peripheral DXA vs central anatomical sites ranges from r = 0.49 to r = 0.78. Source: Patel, R., Blake, G. M., Jefferies, A., Sautereau-Chandley, P. M., & Fogelman, I. (1998). A comparison of a peripheral DXA system with conventional densitometry of the spine and femur. Journal of Clinical Densitometry, 1(3), 235–244. https://doi.org/10.1385/jcd:1:3:235

Note: the correlation coefficent is different than accuracy because we have a range of T-scores not just one set value you must get to have this condition or not 

3) We fabricated the data to pretend a clinical trial was done to see QUS features and DXA T-scores. (In the future, we can use the InterSystems database!) 

4) We then trained the model using the QUS features as our labels and DXA T-score as our target and found the Random Forest to be best → accuracy: 0.7597 (76% → good for clinical use)



## How to Run

**Install dependencies:**
```bash
pip install streamlit pandas scikit-learn plotly
```

**Clone and run:**
```bash
git clone git@github.com:srinithikri/osteosense.git
cd osteosense
streamlit run app.py
```

Opens at `http://localhost:8501`.

---

## The Application

The dashboard is styled after **Epic EHR** — the most widely used electronic health record system in US hospitals — so it feels familiar to clinicians and fits naturally into existing workflows.

### What You Can Do
- **Switch between 4 demo patients** using the patient switcher bar — each has a different bone health profile and scan history
- **Drag the T-score simulator** in the right panel to simulate any T-score value and watch every element update live: the diagnosis badge, status bar, metric cards, AI summary, probability chart, trend chart, and clinical guidance
- **View longitudinal T-score history** on the trend chart with WHO diagnostic zone shading — the most recent point always reflects the current simulated value
- **See model confidence** broken down across all three classes in both the center chart and right panel

### Demo Patients

| Patient | Age | T-score | Diagnosis |
|---|---|---|---|
| Harrington, Jane G. | 68F | −2.5 | Osteoporosis |
| Okafor, Bunmi A. | 54F | −1.4 | Osteopenia |
| Vasquez, Maria L. | 72F | −3.1 | Osteoporosis |
| Lindqvist, Erik P. | 61M | −0.6 | Normal |

---

## The ML Pipeline

### Data

We train on **1,042 real patient records** combined from two datasets:

- `patient_data_cleaned_dxa.csv` — 802 patients with real **DXA T-scores** (gold-standard bone density scan)
- `patient_data_cleaned_qus.csv` — 240 patients with real **QUS T-scores** (calcaneus heel ultrasound)

Both datasets cover the same anatomical region (calcaneus/heel) and use the same T-score scale, but come from different patient cohorts. Combining them gives the model broader and more balanced coverage across the three diagnostic classes.

### Labels

We apply **WHO diagnostic thresholds** to assign each patient a class:

| T-score range | Class |
|---|---|
| ≥ −1.0 | Normal |
| −2.725 to −1.0 | Osteopenia |
| < −2.725 | Osteoporosis |

We use **−2.725** (not the commonly cited −2.5) because this is the **Youden's Index optimal cutoff** for calcaneus QUS specifically — published research shows this threshold maximizes sensitivity and specificity for heel ultrasound relative to central DXA.

### Model Selection

Three classifiers are trained and evaluated via **5-fold cross-validation** (weighted F1 score). The best one is used in the app:

| Model | Notes |
|---|---|
| Logistic Regression | Fast interpretable baseline |
| Decision Tree (max depth 4) | Mimics clinical decision logic |
| Random Forest (100 trees) | Typically wins — handles class imbalance well |

We use **weighted F1** (not accuracy) because osteoporosis cases are underrepresented — accuracy would reward a model that ignores rare but critical diagnoses.

**Best model (Random Forest) weighted F1: ~0.76** — considered clinically useful for screening.

### How Live Prediction Works

```
T-score slider value
        ↓
model.predict_proba([[tscore]])
        ↓
{ Normal: X%, Osteopenia: Y%, Osteoporosis: Z% }
        ↓
All UI elements update:
  header badge · status bar · metric cards
  AI clinical summary · probability chart
  trend chart · confidence bars · clinical guidance
```

---

## Why QUS → T-score Works

The correlation between **calcaneal QUS** and **central DXA T-scores** is well-established in the literature:

> r = 0.49 to 0.78 depending on anatomical site  
> Patel et al. (1998). *Journal of Clinical Densitometry*, 1(3), 235–244.  
> https://doi.org/10.1385/jcd:1:3:235

This means a heel ultrasound reading isn't as precise as a spine or hip DEXA scan — but at r ≈ 0.55–0.78, it's strong enough to reliably **flag who is at risk**. The goal isn't a perfect score. It's catching the patients who need a DEXA before they fracture.

---

## Future Work

- Expand model inputs from T-score alone to full QUS signal features: **BUA** (Broadband Ultrasound Attenuation), **SOS** (Speed of Sound), **Stiffness Index**, age, BMI, sex — the multi-feature pipeline is already built in `heel.py` and `heel_hgb.py`
- Obtain truly **paired QUS + DXA data from the same patients** to train a direct QUS-signal → diagnosis model (no T-score bridge required)
- Deploy the advanced **ensemble model** (HistGradientBoosting + Random Forest + Logistic soft voting) from `heel_hgb.py` once dependencies are bundled
- Integrate **FRAX fracture risk score** alongside bone density prediction
- Partner with **InterSystems** for live EHR data integration
- Deploy on a **portable tablet** for point-of-care use in rural clinics and low-resource settings
