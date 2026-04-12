# Kashif — Agent Directive (program.md)

This file is read by the Kashif agent before every run.
Edit this file to steer the agent without touching code.

---

## Goal
Predict Titanic passenger survival (binary classification: survived=1, died=0).

## Constraints
- Do not touch target column: survived
- Max rounds: 4
- Min improvement per round: 0.005

## Domain hints
This is the Titanic survival dataset. Key domain knowledge:
- pclass: ticket class (1=first, 2=second, 3=third) — strong signal, women/children in 1st class prioritised
- sex: female survival rate was much higher (women and children first)
- age: children were prioritised; young adults had lower survival
- sibsp: number of siblings/spouses aboard
- parch: number of parents/children aboard
- fare: correlated with pclass and cabin location
- embarked: port of embarkation (S=Southampton, C=Cherbourg, Q=Queenstown)

## Suggested features to engineer
- family_size = sibsp + parch + 1
- is_alone = 1 if family_size == 1 else 0
- fare_per_person = fare / family_size
- title = extracted from name (if available) — not available here
- age groups (child < 12, adult, senior)
- pclass * fare interaction
- sex encoded as binary (female=1)
- age * pclass interaction

## Stopping criteria
Stop early if:
- CV score improvement is less than 0.5% over previous round
- All obvious feature engineering angles have been tried
- Max rounds reached

## Output preferences
- Save best model to: ./outputs/best_model.pkl
- Save report to: ./outputs/report.md
- Log all rounds to: ./outputs/experiment_log.json