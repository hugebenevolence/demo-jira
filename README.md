# Demo Jira

Minimal house price prediction project for Jira and GitHub integration testing.

## Jira Issue

SCRUM-61 Train Linear Regression Baseline

## Project Structure

```text
demo-jira/
├── README.md
├── requirements.txt
├── src/
│   ├── load_data.py
│   ├── preprocess.py
│   ├── train_baseline.py
│   └── evaluate.py
├── app/
│   └── main.py
└── reports/
    └── figures/
```

## Run Baseline Training

```bash
pip install -r requirements.txt
python src/train_baseline.py
```

## Baseline Results

```text
R^2: 0.5758
MAE: 0.5332
RMSE: 0.7456
```
