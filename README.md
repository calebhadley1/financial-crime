# financial_crime

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Financial Crime detection using Machine Learning


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

Add:
- feature engineering
- class imbalance
- precision/recall tradeoffs
- threshold selection
- model calibration
- offline vs. online inference
- batch vs. streaming
- feature leakage prevention
- model versioning
- API serving
- Docker
- automated tests
- CI/CD
- monitoring
- drift detection
- latency measurements
- model performance over time

## Setup
### Download Source Data
1. Download the required input dataset from https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml
2. Copy the `HI-Small_accounts.csv`, `HI-Small_Patterns.txt` and `HI-Small_Trans.csv` to `data/raw`
### Python Environment
3. Run `make create_environment`
4. Run `source ./.venv/bin/activate` (or `.\\\\.venv\\\\Scripts\\\\activate` for Windows)
5. Run `make requirements`
### Setup Data for Modeling
6. Run `make data`
7. Run `typer financial_crime/features.py run`

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
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

