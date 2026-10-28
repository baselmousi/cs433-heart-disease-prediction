# Data and preprocessing: what we did and why

This page explains how the raw survey becomes the matrix the model is trained on,
and where each decision came from. The decisions themselves are written down in
[`src/feature_spec.py`](../src/feature_spec.py), one entry per feature with a comment;
[`src/preprocessing.py`](../src/preprocessing.py) applies them. The notebooks in this
folder are the evidence.

| Notebook | Question it answers | Role |
|---|---|---|
| [`01_data_exploration.ipynb`](01_data_exploration.ipynb) | What is in the data, and how are the answers coded? | Background for everything below |
| [`02_feature_selection.ipynb`](02_feature_selection.ipynb) | What do we do with each of the 321 columns, and do the data support it? | Evidence for `feature_spec.py` and the pipeline |
| `02_data_exploration_step_by_step.ipynb` | How should five hand-picked features be encoded? | Only for the first baseline, `train_baseline.py` |
| `02_feature_selection_jd.ipynb` | Can model-based imputation (MICE) and FAMD help? | Explored; not used by the final pipeline |

Read `01`, then `02_feature_selection`. Section names below refer to the headings
inside those notebooks.

## 1. The data

`x_train.csv` holds 328,135 people and 321 columns; `x_test.csv` 109,379 people with
the same columns. The label `_MICHD` is +1 for 8.8 % of the training rows (told they
had a heart attack, angina or coronary heart disease) and -1 otherwise. The columns
are answers to the BRFSS 2015 telephone survey: 218 are raw answers to one question,
103 are variables CDC calculated from them (`01`, "Exploring the Dataset").

Five properties of the data drive the preprocessing:

1. **Numbers are codes, not quantities.** 7 / 77 mean "don't know", 9 / 99 "refused",
   88 "none" (often a real 0), and calculated columns have their own codes (for example
   `_DRNKWEK = 99900` for missing). The codes are not universal: in `EMPLOY1`, 7 means
   "retired". → steps 1-2 below.
2. **Skip patterns.** Most empty fields mean "not asked". Some questions are asked only
   after a given answer, so an empty field can itself be an answer. → step 4.
3. **Optional modules.** Some states asked extra questionnaires, so those columns are
   mostly empty. → section 3.
4. **The same information several times.** BRFSS ships many concepts as the raw answer
   plus one or more calculated versions (age four times, race seven, the
   physical-activity block thirty). → section 3.
5. **Some questions are asked because of a heart condition**, such as cardiac rehab
   after a heart attack. They reveal the label. → section 5.

The positive rate rises steeply with age (0.5 % at 18-24, 18.5 % at 75-79) and with
poor general health (2.1 % "excellent", 31.9 % "poor"), as the plots in `01` show.

## 2. How the decisions were made

1. **Read every code's meaning in the
   [codebook](https://www.cdc.gov/brfss/annual_data/annual_2015.html), and check it
   against value counts in `x_train`.** The counts caught what the codebook alone
   would not: `_DRNKWEK` is still ×100 in this release, some calculated columns hide
   missing codes, and some columns are exact copies (`02`, "Where the feature spec
   comes from").
2. **Keep a feature by default; drop it only for a stated reason.** The reasons are
   checked in this order: not about the respondent's health; redundant with a kept
   column; not usable as a feature (dates, or values computed only from age and sex);
   optional module with at least 90 % missing. Among redundant columns we keep the
   most informative one: the calculated version when it is already decoded (`_AGE80`,
   `_BMI5`, `_SMOKER3`), the raw answer when the calculated one throws information
   away (`BPHIGH4` keeps "borderline", which `_RFHYPE5` merges into "no").
3. **Test the claims on the data instead of trusting them**: every feature is covered
   exactly once, the redundancy and skip-pattern claims hold, and no dropped feature
   carries signal that a kept one lacks (`02`, sections "Decisions On Every Single
   Feature" to "Inspecting Continuous Features").
4. **Settle the choices that could go either way by cross-validation**, comparing
   each option with the final pipeline fold by fold (section 5).

## 3. Where the 321 columns went

| Group | Columns | Rule | Evidence in `02` | In `feature_spec.py` |
|---|---:|---|---|---|
| Kept | 77 | Kept by default: 35 binary, 6 ordinal, 14 continuous, 22 nominal | "Decisions On Every Single Feature" | `SPEC` |
| Dropped on purpose | 149 | 99 redundant with a kept column; 44 not about the respondent's health (phone screening, interview metadata, survey weights, a child in the household); 4 dates or places of a flu shot or HIV test; 2 computed only from age and sex | "Checking Redundancy and Skip Pattern Claims", "Did dropping lose any signal?" | `DROP`, one reason each |
| Optional modules | 89 | At least 90 % missing | "Did dropping lose any signal?" | `MODULE_NAN_THRESHOLD` |
| Leaky | 6 | Asked only because of a heart condition: cardiac rehab (`HAREHAB1`) and aspirin follow-ups | "Feature Leakage" | `LEAKY` |

Dropping lost no signal: every dropped feature that separates the classes well on its
own (univariate AUC above 0.6) is a recoding of a kept one, mostly of age, and the
module questions are weak (AUC at most 0.56).

## 4. From survey codes to the model matrix

`fit_preprocessor` learns every statistic from the rows it is given and
`transform_features` applies them, so in cross-validation each fold's pipeline is
fitted on its training rows only. The steps, in order:

| # | Step | Why | Evidence in `02` | Code |
|---|---|---|---|---|
| 1 | Missing codes → NaN | "Don't know" or 77 is not a quantity | "Cleaning Missing Values": `INCOME2` goes from 0.8 % to 17.9 % missing | `SPEC[name][1]`, `clean_columns` |
| 2 | Remap special answers, e.g. 88 → 0 days | "None" is a real answer | same | `SPEC[name][2]` |
| 3 | Fix units: `_DRNKWEK` × 0.01 | Its implied decimals are not applied in this release | "Where the feature spec comes from" | `SCALE` |
| 4 | Fill skip patterns: `POORHLTH` = 0 when both day counts are 0; `BPMEDS` = "no" without high blood pressure | "Not asked" implies the answer | "Checking Redundancy and Skip Pattern Claims" (exact in the data); missing share 48.7 % → 1.1 % and 59.8 % → 0.3 % | `FILLS` |
| 5 | Drop a kept feature with over 90 % missing after cleaning | Too empty to impute | "Cross-Validation Checks" (stricter thresholds are worse) | `max_nan=0.9`; removes none of the 77 |
| 6 | Binary: yes → 1, no → 0 | | | `fit_preprocessor` |
| 7 | Clip continuous features to their 0.5-99.5 % training quantiles | A few absurd values (`PA1MIN_` 54,000 minutes of activity a week) would squeeze everyone else after scaling | "Inspecting Continuous Features": at most 1 % of rows change | `clip=0.995` |
| 8 | Binary, ordinal and continuous: missing → training median | | | `fit_preprocessor` |
| 9 | Nominal: one 0/1 column per level, "missing" as a level of its own; the most frequent level is the reference and gets no column; levels with fewer than 30 training rows share the reference | Nominal codes have no order. With a column for every level, each feature's columns would add up to the intercept and least squares could not be solved | "Is missingness informative?" (non-answers carry signal) | `min_count=30` |
| 10 | Drop constant columns and exact combinations of earlier ones | The three arthritis-limit questions (`_LMT*`) share their skip pattern, so two "missing" levels are redundant | | `_independent`; `state["dependent"]` |
| 11 | Standardize every column with the training mean and std; add the intercept | Puts all features on one scale for gradient descent | | `fit_preprocessor` |

**Result: 200 columns** in every fold: the intercept, 55 numeric columns (35 binary +
6 ordinal + 14 continuous) and 144 one-hot levels (146 non-reference levels of the 22
nominal features, minus the 2 of step 10; `_STATE` alone gives 52). To print the count
from the repository root:

```bash
python -c "from src.preprocessing import *; names = read_feature_names('data/x_train.csv'); x = load_csv('data/x_train.csv')[:, 1:]; print(len(fit_preprocessor(x, names)['columns']))"
```

### The notebook's draft pipeline differs from the final one

`02` builds a first version of the pipeline ("Pipeline Draft") and reports **263
columns**. Three things changed before the final version in `src/preprocessing.py`:

| | Draft in `02` | Final | Columns |
|---|---|---|---:|
| Missing indicators | On | Off, because the notebook's own check ("Cross-Validation Checks") and E3 found no gain | -39 |
| Nominal features | A column for every level | Most frequent level as reference (step 9) | -22 |
| Exact combinations | Kept | Dropped (step 10) | -2 |

263 - 39 - 22 - 2 = 200. The extra columns carry no new information, and F1 agrees:
0.4271 for the draft, 0.4274 for the final pipeline.

## 5. Decisions we tested instead of assuming

Each row changes one thing in the final pipeline and keeps everything else, including
the model (regularized logistic regression, λ 1e-4, γ 1, 200 steps), the 5 folds
(stratified, seed 42) and the cutoff search. ΔF1 is the mean per-fold difference from
the final pipeline (F1 0.4274); "no clear difference" means the corrected paired
t-test does not reject equality at 95 % (|t| < 2.78). Run with `run_cv.py`, compared
with `compare_cv.py`.

| Decision | Alternative tested | Experiment | ΔF1 | Folds better | Verdict |
|---|---|---|---:|---:|---|
| Decode the survey codes (steps 1-4, 7-9) | Same 77 features, raw codes, mean-imputed | E2-raw-codes | -0.0219 | 0/5 | Decoding helps |
| Select and decode features | Every non-leaky column, raw codes, mean-imputed | E0-raw-logistic | -0.0140 | 0/5 | Selection helps |
| No missing indicators | Indicators for features with at least 1 % missing | E3-indicators | -0.0004 | 2/5 | No clear difference: keep the simpler matrix |
| Drop features over 90 % missing | Over 50 % | E3-max-nan-0.5 | -0.0016 | 2/5 | No clear difference: keep 0.9 |
| Ordinal features as numbers | One-hot | E4-ordinal-onehot | -0.0001 | 2/5 | No clear difference: keep numbers |
| Keep the borderline features | Drop the vaccine questions and `BPMEDS` | E5-no-borderline | -0.0020 | 1/5 | No clear difference, but every drop costs a little: keep |
| Linear features only | Add squares, age and BMI bins, age interactions | E8-all | +0.0011 | 4/5 | No clear difference: keep the simpler model |
| Leave out the leaky features | Add them back | E9-leaky | +0.0099 | 5/5 | Better, but left out (below) |

**Missingness is informative, yet indicators add nothing.** People who skip a common
question are about twice as often positive ("Is missingness informative?"), which is
why the draft used missing indicators. For nominal features, "missing" is already a
level of its own (step 9). For the numeric ones, the cross-validation shows no gain;
a likely reason, which we did not isolate, is that the conditions behind these skip
patterns are kept features themselves (`DIABETE3`, `HAVARTH3`, age, sex).

**Why the leaky features stay out.** `HAREHAB1` ("did you go to cardiac rehab after
your heart attack?") is answered by 773 training rows, all positive, and by 238 test
rows. It explains most of the +0.010: on all other rows, F1 moves only from 0.422 to
0.425 ("Feature Leakage"). A question that is only asked after the diagnosis restates
the label rather than predicting it, so the final model does not use it.

**Where the line is drawn.** A feature is left out when its answer is a consequence
of the diagnosis rather than a risk factor. `HAREHAB1` is the clear case. The five
aspirin questions ask about taking aspirin to prevent a heart attack or stroke; they
come from an optional module answered by at most 4.3 % of the rows, so the 90 %
module rule (section 3) would drop them anyway. The borderline features stay in
because who is asked does not depend on the label: `BPMEDS` is asked of everyone
with high blood pressure (the `BPHIGH4` = yes rows), and the flu and
pneumonia vaccine questions of everyone. Their answers still differ by label
(pneumonia vaccine: 15 % positive among "yes", 5 % among "no"), partly because
vaccination is recommended after a heart diagnosis, so they may carry some reverse
causation. Dropping all three changes F1 by -0.0020 (E5-no-borderline, no clear
difference), so no conclusion depends on this choice.

## 6. Limitations

- The feature decisions were made by hand, partly from checks on all labelled training
  rows (the univariate AUCs in `02`). Every fitted statistic is refit inside each fold,
  but the choice of the 77 features was not itself cross-validated.
- Redundancy was verified exactly in the data for a few cases (arthritis, smoking,
  age, the fitness formulas, the two skip-pattern fills). For the other drops, the
  evidence is the codebook and the AUC ranking of section 3.
