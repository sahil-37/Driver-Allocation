# 🚗 Driver Allocation Optimization

This repository presents a complete machine learning pipeline to identify the most suitable drivers for order allocation, as part of a take-home assignment.

---

## 🎯 Objective

Predict which driver is most likely to accept and complete a given order, using historical offer and booking logs.

---
## 🗂️ Project Structure

```text
├── data/             # Raw and processed data
├── src/              # Core logic: cleaning, features, training, prediction
├── submission/       # Output predictions and evaluation metrics
├── tests/            # Unit tests
├── config.toml       # Centralized pipeline config
├── Makefile          # Automates pipeline steps
└── README.md  
---
```

## ⚙️ Pipeline Execution

To run the entire ML pipeline from scratch:

```bash
make run


Or run each stage manually:

make data        # Clean and prepare logs
make features    # Engineer driver-order features
make train       # Train model on historical data
make predict     # Predict best driver for test orders

```
✅ Output Files
submission/metrics.json: Model evaluation scores (e.g., F1, accuracy).

submission/results.csv: Final prediction file with the format:

```json
| order_id  | driver_id |
|-----------|-----------|
| 100032007 | 987454392 |
| 100167816 | 790186080 |
```

🧪 Testing
To verify correctness of the pipeline:
```bash
make test
```
📌 Final Model Metrics
```json
{
  "f1_score": 0.8718,
  "accuracy": 0.89,
  "precision": 0.88,
  "recall": 0.86
}
```
