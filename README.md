# financial_crime

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Financial Crime detection using Machine Learning

This projects contains both acadmic data science work and production ready ML systems

Major topics covered:
- Academic data science work on exploratory data analysis and modeling can be found [here](papers\README.md)
- Reusable feature generation, training/inference pipelines, model persistance
- Model serving with FastAPI + Docker
- Feature Store [TODO]
- Real-time Kafka Streaming for Model Inference [TODO]
- Batch PySpark Pipelines for Historical Feature Generation
- Model Versioning [TODO]
- Monitoring (Drift, latency, distributions) [TODO]
- Automated tests [TODO]
- CI/CD [TODO]



Todos:
- add work on graph based detection using synthetic data: https://github.com/SantanderAI/gen-fraud-graph
- add work on real time monitoring vs the batch techniques done in these projects


                    ┌──────────────┐
Transactions ──────►│ Kafka / Queue│
                    └──────┬───────┘
                           │
                           ▼
                 ┌──────────────────┐
                 │ Feature Pipeline  │
                 └────────┬─────────┘
                          │
             ┌────────────┴────────────┐
             ▼                         ▼
       Batch features             Online features
             │                         │
             └────────────┬────────────┘
                          ▼
                   ┌──────────────┐
                   │ ML Inference │
                   │   Service    │
                   └──────┬───────┘
                          │
             ┌────────────┴────────────┐
             ▼                         ▼
        Risk score                 Decision
             │                         │
             ▼                         ▼
       Monitoring              Alert / Review


## Setup
### Download Source Data
- Download the required input dataset from https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml
- Copy the `HI-Small_accounts.csv`, `HI-Small_Patterns.txt` and `HI-Small_Trans.csv` to `data/raw`
### Python Environment
- Run `make create_environment`
- Run `source ./.venv/bin/activate` (or `.\\\\.venv\\\\Scripts\\\\activate` for Windows)
- Run `make requirements`
### Get Features for EDA
- Run `make data`
- Run `typer financial_crime/features.py run`
### Train Model
- Run `typer financial_crime/modeling/train.py run`
### Get Model Predictions
- Run `typer financial_crime/modeling/predict.py run`
### API
- Run the app with docker using the typical `docker compose up --build`
- Make a sample request @ localhost:8000/docs with the following payload:
```
{
  "raw_features": [
    {
      "Timestamp": "2022/09/01 00:20",
      "From Bank": "10",
      "Account": "8000EBD30",
      "To Bank": "10",
      "Account.1": "8000EBD30",
      "Amount Received": 3697.34,
      "Receiving Currency": "US Dollar",
      "Amount Paid": 3697.34,
      "Payment Currency": "US Dollar",
      "Payment Format": "Reinvestment"
    }
  ]
}
```
### Feature Store
#### Apply
- Run `feast apply` from the `financial_crime\feature_store\feature_repo` directory

## Modeling Results
TODO: boost positive class precision
              precision    recall  f1-score   support

           0       1.00      0.98      0.99   1014634
           1       0.03      0.66      0.05      1035

    accuracy                           0.98   1015669
   macro avg       0.51      0.82      0.52   1015669
weighted avg       1.00      0.98      0.99   1015669

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── papers             <- Archival academic work done on the subject completed in graduate school
|
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         financial_crime and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── financial_crime   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes financial_crime a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for exploratory modeling
    │
    |── api                     <- Live API for model inference
    ├── modeling    
    │   ├── __init__.py 
    |   ├── pipelines           <- Training/inference abstractions
    |   ├── transformers        <- Feature Engineering and Preprocessing abstractions
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

