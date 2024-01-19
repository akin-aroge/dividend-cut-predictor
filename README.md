[![Codespaces Prebuilds](https://github.com/akin-aroge/dividend-cut-predictor/actions/workflows/codespaces/create_codespaces_prebuilds/badge.svg)](https://github.com/akin-aroge/dividend-cut-predictor/actions/workflows/codespaces/create_codespaces_prebuilds)

# Predicting Likelihood of Dividend Cut

An AI solution to improve investment decision-making by predicting the likelihood of future dividend cuts for companies in the S & P 500.


#### -- Project Status: [Active]

## Project Description

**The problem:** For investors, the decision to buy, hold, or sell a stock is often influenced by the company's dividend policy. If there is a high probability of a dividend cut, investors may reassess their investment thesis, risk tolerance, and overall portfolio strategy.

**The proposed solution:** The idea is to predict the likelihood that companies in the S & P would cut dividend based on relevant data such as financial statements, fundamentals and macro-economic data.


**The dataset:**
The raw dataset consists of financial statements, company fundamentals and macro-economic data obtain via APIs provided by [Financial Modeling Prep](https://site.financialmodelingprep.com/developer) and [Federal Reserve Economic Data](https://fred.stlouisfed.org/docs/api/fred/). Other company relevant information was also obtained from [wikipedia](https://en.wikipedia.org/wiki/List_of_S%26P_500_companies) and the yahoo finance pacakge.



### Model Training

![Model Traing Schematic](./reports/project-schematic.png?raw=true)

I have used:
* logistic regression
* random forest
* xgboost

The XGBoost model performs best based on the ROC-AUC metric.

### Technologies

* Numpy
* Pandas
* Scikit-learn

### Project Organization
```
.
├── artifacts
├── config
├── data
│   ├── interim
│   ├── processed
│   └── raw
├── models
│   ├── artifacts
│   └── random_forest
├── notebooks
├── reports
└── src
    ├── data_preprocessing
    ├── data_retrieval
    ├── inference
    ├── modelling
    ├── tools
    └── utils
```