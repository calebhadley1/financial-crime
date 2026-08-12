# Financial Crime Detection

This repo contains work completed in my Master's of Data Science program on the subject of financial crime detection using machine learning

Todos:
- organize papers alongside the code used for research
- discuss techniques used (predictive modeling, eda, etc)
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