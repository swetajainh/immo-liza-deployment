# immo-liza-deployment
[![forthebadge made-with-python](https://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)
## 🏢 Description

The real estate company Immo Eliza asked us to create to create an API so their web developers can access the predictions whenever they need to. This model is deployed using Render.

## 🧬 Project structure

```
immo-liza-deployment/
│
├── data/
│   └── properties.csv                --- data: training dataset
│                      
│
├── api/                    
│   └── train.ipynb
│   └── predict.py
|   └── trained_model.pkl
|   └── train1_copy.ipynb
|   └── Dockerfile
|   └── cat_model.pkl
|   └── requirements.txt
|
|                   
├── .gitignore
├── requirements.txt
└── README.md
```
## Usage

Python Libraries: scikit-learn, CatBoost,random forest regressor, fastapi,uvicorn
Data Preprocessing Libraries: pandas, numpy
To deploy using the trained model -

![image](https://github.com/swetajainh/immo-liza-deployment/assets/158171729/70541dd1-4e7e-46f4-91d9-7f7864d8f8da)

![image](https://github.com/swetajainh/immo-liza-deployment/assets/158171729/79e01c0d-a1f2-4903-bd0d-57834108a7f3)

Link - https://property-price-predict-api.onrender.com/docs#/house/predict_house__post


## ⏱️ Timeline

This project took five days for completion.
