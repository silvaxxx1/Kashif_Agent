# Kashif — Agent Directive (program.md)

This file is read by the Kashif agent before every run.
Edit this file to steer the agent without touching code.

---

## Goal
Maximize predictive performance on the target column.
Default metric: F1 (classification) / RMSE (regression)

## Constraints
- Do not use columns: []          ← add forbidden columns here
- Do not touch target column: ~   ← auto-filled at runtime
- Max rounds: 4
- Min improvement per round: 0.005

## Domain hints
← describe what you know about this dataset here
← example: "This is customer churn data. Tenure and monthly charges are likely strong signals."
← leave blank if unknown — the agent will profile the data itself

## Stopping criteria
Stop early if:
- CV score improvement is less than 0.5% over previous round
- All obvious feature engineering angles have been tried
- Max rounds reached

## Output preferences
- Save best model to: ./outputs/best_model.pkl
- Save report to: ./outputs/report.md
- Log all rounds to: ./outputs/experiment_log.json