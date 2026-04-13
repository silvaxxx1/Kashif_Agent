# data/

> Local input datasets for Kashif runs and smoke tests.

Place your input CSV files here before running Kashif.

The `.csv` files in this directory are gitignored — do not commit raw data.

## Example datasets used in development

| File | Source | Rows | Task |
|---|---|---|---|
| `titanic.csv` | OpenML (id=40945) | 1309 | Binary classification |

## Fetching Titanic for local testing

```python
from sklearn.datasets import fetch_openml
import pandas as pd

titanic = fetch_openml("titanic", version=1, as_frame=True)
df = titanic.frame[["pclass","sex","age","sibsp","parch","fare","embarked","survived"]].copy()
df["survived"] = df["survived"].astype(int)
df.to_csv("data/titanic.csv", index=False)
```
