# Data

Download the course files from the
[AIcrowd challenge](https://www.aicrowd.com/challenges/epfl-machine-learning-project-1)
and put them in this folder, keeping their names:

| File | Contents |
|---|---|
| `x_train.csv` | 328,135 training rows: `Id` + 321 survey features |
| `y_train.csv` | `Id` and the label `_MICHD` (-1 / +1) |
| `x_test.csv` | 109,379 test rows with the same columns as `x_train.csv` |
| `sample_submission.csv` | The expected submission format; `run.py` checks its IDs against it |

The CSV files are not committed (see `.gitignore`). Every script also accepts
`--data-dir` to read them from another folder.
