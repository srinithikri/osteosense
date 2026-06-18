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

