"""Per-feature cleaning and encoding decisions for the BRFSS 2015 heart-disease data.

Every one of the 321 features ends up in exactly one place:

- ``SPEC``: kept features, with how to clean and encode them.
- ``DROP``: features removed on purpose, with the reason.
- ``LEAKY``: follow-up questions asked only because of a heart condition. Dropped by
  default; the leakage experiment (E9) adds them back.
- Anything else is an optional-module question with >= 90 % NaN, dropped by
  ``MODULE_NAN_THRESHOLD`` (checked in notebooks/02_feature_selection.ipynb).

Each ``SPEC`` entry is ``(kind, missing_codes, remap)``:

- ``kind``: "binary" (1 = yes -> 1, 2 = no -> 0), "ordinal" or "continuous" (kept as a
  number), or "nominal" (one-hot encoded).
- ``missing_codes``: values meaning "don't know", "refused" or "not asked"; set to NaN.
- ``remap``: other value replacements, e.g. ``{88: 0}`` for "none" in day counts.

Codes follow the BRFSS 2015 codebook. Calculated variables in this release are already
divided by their implied decimals, except ``_DRNKWEK`` (still x100, see ``SCALE``).
"""

DK_REF = [7, 9]  # one-digit answers: 7 = don't know / not sure, 9 = refused
DK_REF2 = [77, 99]  # two-digit answers
NONE = {88: 0}  # 88 = "none" in day/count questions

MODULE_NAN_THRESHOLD = 0.9

SPEC = {
    # --- Health status and access to care
    "GENHLTH": ("ordinal", DK_REF, {}),  # 1 excellent ... 5 poor
    "PHYSHLTH": ("continuous", DK_REF2, NONE),  # days of poor physical health, last 30
    "MENTHLTH": ("continuous", DK_REF2, NONE),  # days of poor mental health, last 30
    "POORHLTH": (
        "continuous",
        DK_REF2,
        NONE,
    ),  # skipped if both above are 0 (see FILLS)
    "HLTHPLN1": ("binary", DK_REF, {}),  # any health coverage
    "PERSDOC2": ("nominal", DK_REF, {}),  # 1 one doctor, 2 more than one, 3 none
    "MEDCOST": ("binary", DK_REF, {}),  # could not see a doctor because of cost
    "CHECKUP1": ("ordinal", DK_REF, {8: 5}),  # 1 < 1 year ... 4 5+ years, 8 never -> 5
    # --- Blood pressure and cholesterol
    "BPHIGH4": (
        "nominal",
        DK_REF,
        {},
    ),  # 1 yes, 2 only in pregnancy, 3 no, 4 borderline
    "BPMEDS": ("binary", DK_REF, {}),  # asked only if BPHIGH4 = 1
    "_CHOLCHK": ("nominal", [9], {}),  # 1 checked <= 5 years, 2 > 5 years, 3 never
    "TOLDHI2": (
        "binary",
        DK_REF,
        {},
    ),  # told high cholesterol (skipped if never checked)
    # --- Chronic conditions
    "CVDSTRK3": ("binary", DK_REF, {}),  # stroke
    "CHCSCNCR": ("binary", DK_REF, {}),  # skin cancer
    "CHCOCNCR": ("binary", DK_REF, {}),  # other cancer
    "CHCCOPD1": ("binary", DK_REF, {}),  # COPD, emphysema, chronic bronchitis
    "HAVARTH3": ("binary", DK_REF, {}),  # arthritis
    "ADDEPEV2": ("binary", DK_REF, {}),  # depressive disorder
    "CHCKIDNY": ("binary", DK_REF, {}),  # kidney disease
    "_ASTHMS1": ("nominal", [9], {}),  # asthma: 1 current, 2 former, 3 never
    "DIABETE3": (
        "nominal",
        DK_REF,
        {},
    ),  # 1 yes, 2 only in pregnancy, 3 no, 4 pre-diabetes
    "DIABAGE2": (
        "continuous",
        [98, 99],
        {},
    ),  # age at diabetes diagnosis (diabetics only)
    "PDIABTST": ("binary", DK_REF, {}),  # tested for high blood sugar, last 3 years
    "PREDIAB1": ("nominal", DK_REF, {}),  # told pre-diabetes
    # --- Demographics
    "_STATE": ("nominal", [], {}),  # state FIPS code
    "MSCODE": ("nominal", [], {}),  # metropolitan status (landline only)
    "SEX": ("binary", [], {}),  # 1 male -> 1, 2 female -> 0
    "_AGE80": ("continuous", [], {}),  # age in years, top-coded at 80
    "_RACE": ("nominal", [9], {}),  # race/ethnicity, 8 groups
    "MARITAL": ("nominal", [9], {}),
    "EDUCA": ("ordinal", [9], {}),  # 1 never attended ... 6 college graduate
    "EMPLOY1": ("nominal", [9], {}),  # 7 = retired (a real answer, not "don't know")
    "INCOME2": ("ordinal", DK_REF2, {}),  # 1 < $10k ... 8 >= $75k
    "RENTHOM1": ("nominal", DK_REF, {}),  # own, rent, other
    "VETERAN3": ("binary", DK_REF, {}),
    "CHILDREN": ("continuous", [99], NONE),  # children in household
    "INTERNET": ("binary", DK_REF, {}),
    "PREGNANT": ("binary", DK_REF, {}),  # women 18-44 only
    "SXORIENT": ("nominal", DK_REF, {}),  # optional module
    "TRNSGNDR": ("nominal", DK_REF, {}),  # optional module
    # --- Body measures
    "_BMI5": ("continuous", [], {}),  # BMI (already / 100)
    "HTM4": ("continuous", [], {}),  # height in metres
    # --- Disability
    "QLACTLM2": ("binary", DK_REF, {}),  # activity limited by health problems
    "USEEQUIP": ("binary", DK_REF, {}),  # uses special equipment
    "BLIND": ("binary", DK_REF, {}),
    "DECIDE": ("binary", DK_REF, {}),  # difficulty concentrating or deciding
    "DIFFWALK": ("binary", DK_REF, {}),
    "DIFFDRES": ("binary", DK_REF, {}),
    "DIFFALON": ("binary", DK_REF, {}),  # difficulty doing errands alone
    # --- Tobacco and alcohol
    "_SMOKER3": ("nominal", [9], {}),  # 1 daily, 2 some days, 3 former, 4 never
    "STOPSMK2": ("binary", DK_REF, {}),  # tried to quit (current smokers)
    "LASTSMK2": ("nominal", DK_REF2, {}),  # time since quitting (former smokers)
    "USENOW3": ("ordinal", DK_REF, {}),  # smokeless tobacco: 1 daily, 2 some days, 3 no
    "DRNKANY5": ("binary", DK_REF, {}),  # any alcohol in the past 30 days
    "_DRNKWEK": ("continuous", [99900], {}),  # drinks per week (x100, see SCALE)
    "_RFBING5": ("binary", [9], {}),  # 1 no binge drinking -> 1, 2 yes -> 0
    # --- Diet and physical activity
    "_FRUTSUM": ("continuous", [], {}),  # fruit intake per day
    "_VEGESUM": ("continuous", [], {}),  # vegetable intake per day
    "EXERANY2": ("binary", DK_REF, {}),  # any exercise in the past 30 days
    "_PACAT1": ("ordinal", [9], {}),  # 1 highly active ... 4 inactive
    "PA1MIN_": ("continuous", [], {}),  # minutes of activity per week
    "STRFREQ_": ("continuous", [99], {}),  # strength training per week (99 = unknown)
    # --- Arthritis follow-ups (calculated versions, 3 or 4 = no arthritis)
    "_LMTACT1": ("nominal", [9], {}),
    "_LMTWRK1": ("nominal", [9], {}),
    "_LMTSCL1": ("nominal", [9], {}),
    "JOINPAIN": ("continuous", DK_REF2, {}),  # joint pain 0-10 (arthritis only)
    # --- Prevention and screening
    "_RFSEAT2": ("binary", [9], {}),  # always or nearly always wears a seat belt
    "FLUSHOT6": ("binary", DK_REF, {}),
    "PNEUVAC3": ("binary", DK_REF, {}),
    "HIVTST6": ("binary", DK_REF, {}),
    "BLDSTOOL": ("binary", DK_REF, {}),  # colorectal screening module
    "HADSIGM3": ("binary", DK_REF, {}),
    # --- Optional modules with < 90 % NaN
    "CAREGIV1": ("nominal", DK_REF, {}),  # caregiver (8 = care recipient died)
    "CRGVEXPT": ("binary", DK_REF, {}),  # expects to become a caregiver
    "CIMEMLOS": ("binary", DK_REF, {}),  # memory loss, age 45+
    "SCNTMNY1": ("nominal", DK_REF, {}),  # worried about rent money
    "SCNTMEL1": ("nominal", DK_REF, {}),  # worried about money for meals
}

# Values that can be filled from the skip pattern instead of imputed.
# (feature, value, condition): set feature to value where it is NaN and condition holds.
# ``get(name)`` returns the cleaned column of any SPEC feature.
FILLS = [
    # POORHLTH is asked only if PHYSHLTH or MENTHLTH > 0, so "not asked" means 0 days.
    ("POORHLTH", 0, lambda get: (get("PHYSHLTH") == 0) & (get("MENTHLTH") == 0)),
    # BPMEDS is asked only if BPHIGH4 = 1 (high blood pressure), so otherwise "no" (2).
    ("BPMEDS", 2, lambda get: (get("BPHIGH4") >= 2) & (get("BPHIGH4") <= 4)),
]

SCALE = {"_DRNKWEK": 0.01}  # implied decimals not yet applied in this release

LEAKY = {
    "HAREHAB1": "cardiac rehab after a heart attack: present => label is always +1",
    "CVDASPRN": "takes aspirin to prevent heart disease (asked about heart history)",
    "ASPUNSAF": "aspirin follow-up",
    "RLIVPAIN": "aspirin follow-up",
    "RDUCHART": "aspirin reduces the chance of a heart attack (follow-up)",
    "RDUCSTRK": "aspirin follow-up",
}

_METADATA = "interview metadata, not health information"
_PHONE = "phone screening / household listing"
_WEIGHT = "survey design or weighting variable"
_CHILD = "about a random child in the household, not the respondent"


def _redundant(keep):
    return f"redundant with {keep}"


DROP = {
    # Interview metadata and survey design
    **{
        n: _METADATA
        for n in [
            "FMONTH",
            "IDATE",
            "IMONTH",
            "IDAY",
            "IYEAR",
            "DISPCODE",
            "SEQNO",
            "_PSU",
            "QSTVER",
            "QSTLANG",
        ]
    },
    **{
        n: _PHONE
        for n in [
            "CTELENUM",
            "PVTRESD1",
            "COLGHOUS",
            "STATERES",
            "CELLFON3",
            "LADULT",
            "NUMADULT",
            "NUMMEN",
            "NUMWOMEN",
            "CTELNUM1",
            "CELLFON2",
            "CADULT",
            "PVTRESD2",
            "CCLGHOUS",
            "CSTATE",
            "LANDLINE",
            "HHADULT",
            "NUMHHOL2",
            "NUMPHON2",
            "CPDEMO1",
            "_DUALUSE",
        ]
    },
    **{
        n: _WEIGHT
        for n in [
            "_STSTR",
            "_STRWT",
            "_RAWRAKE",
            "_WT2RAKE",
            "_CLLCPWT",
            "_DUALCOR",
            "_LLCPWT",
        ]
    },
    **{
        n: _CHILD
        for n in ["_CHISPNC", "_CRACE1", "_CPRACE", "RCSGENDR", "RCSRLTN2", "CASTHDX2"]
    },
    # Health status and care
    "_RFHLTH": _redundant("GENHLTH"),
    "_HCVU651": _redundant("HLTHPLN1 (coverage, ages 18-64 only)"),
    "_RFHYPE5": _redundant("BPHIGH4"),
    "BLOODCHO": _redundant("_CHOLCHK"),
    "CHOLCHK": _redundant("_CHOLCHK"),
    "_RFCHOL": _redundant("TOLDHI2"),
    "ASTHMA3": _redundant("_ASTHMS1"),
    "ASTHNOW": _redundant("_ASTHMS1"),
    "_LTASTH1": _redundant("_ASTHMS1"),
    "_CASTHM1": _redundant("_ASTHMS1"),
    "_DRDXAR1": _redundant("HAVARTH3 (identical values)"),
    "LMTJOIN3": _redundant("_LMTACT1"),
    "ARTHDIS2": _redundant("_LMTWRK1"),
    "ARTHSOCL": _redundant("_LMTSCL1"),
    # Demographics and body measures
    "_AGEG5YR": _redundant("_AGE80"),
    "_AGE65YR": _redundant("_AGE80"),
    "_AGE_G": _redundant("_AGE80"),
    "_EDUCAG": _redundant("EDUCA"),
    "_INCOMG": _redundant("INCOME2"),
    "_CHLDCNT": _redundant("CHILDREN"),
    "_PRACE1": _redundant("_RACE"),
    "_MRACE1": _redundant("_RACE"),
    "_HISPANC": _redundant("_RACE"),
    "_RACEG21": _redundant("_RACE"),
    "_RACEGR3": _redundant("_RACE"),
    "_RACE_G1": _redundant("_RACE"),
    "WEIGHT2": _redundant("_BMI5 and HTM4 (raw, mixed lb/kg codes)"),
    "HEIGHT3": _redundant("HTM4 (raw, mixed ft/cm codes)"),
    "WTKG3": _redundant("_BMI5 and HTM4"),
    "HTIN4": _redundant("HTM4"),
    "_BMI5CAT": _redundant("_BMI5"),
    "_RFBMI5": _redundant("_BMI5"),
    # Tobacco and alcohol
    "SMOKE100": _redundant("_SMOKER3"),
    "SMOKDAY2": _redundant("_SMOKER3"),
    "_RFSMOK3": _redundant("_SMOKER3"),
    "ALCDAY5": _redundant("_DRNKWEK (raw 1xx/2xx frequency code)"),
    "AVEDRNK2": _redundant("_DRNKWEK"),
    "DRNK3GE5": _redundant("_RFBING5"),
    "MAXDRNKS": _redundant("_RFBING5"),
    "DROCDY3_": _redundant("_DRNKWEK"),
    "_RFDRHV5": _redundant("_DRNKWEK"),
    # Diet: raw answers and per-item daily intakes, summarised by _FRUTSUM / _VEGESUM
    **{
        n: _redundant("_FRUTSUM / _VEGESUM")
        for n in [
            "FRUITJU1",
            "FRUIT1",
            "FVBEANS",
            "FVGREEN",
            "FVORANG",
            "VEGETAB1",
            "FTJUDA1_",
            "FRUTDA1_",
            "BEANDAY_",
            "GRENDAY_",
            "ORNGDAY_",
            "VEGEDA1_",
            "_MISFRTN",
            "_MISVEGN",
            "_FRTRESP",
            "_VEGRESP",
            "_FRTLT1",
            "_VEGLT1",
            "_FRT16",
            "_VEG23",
            "_FRUITEX",
            "_VEGETEX",
        ]
    },
    # Physical activity: raw answers and intermediate calculations
    **{
        n: _redundant("_PACAT1 / PA1MIN_ / STRFREQ_")
        for n in [
            "EXRACT11",
            "EXEROFT1",
            "EXERHMM1",
            "EXRACT21",
            "EXEROFT2",
            "EXERHMM2",
            "STRENGTH",
            "METVL11_",
            "METVL21_",
            "ACTIN11_",
            "ACTIN21_",
            "PADUR1_",
            "PADUR2_",
            "PAFREQ1_",
            "PAFREQ2_",
            "_MINAC11",
            "_MINAC21",
            "PAMISS1_",
            "PAMIN11_",
            "PAMIN21_",
            "PAVIG11_",
            "PAVIG21_",
            "PA1VIGM_",
            "_PAINDX1",
            "_PA150R2",
            "_PA300R2",
            "_PA30021",
            "_PASTRNG",
            "_PAREC1",
            "_PASTAE1",
            "_TOTINDA",
        ]
    },
    "MAXVO2_": "computed from age and sex only (999 = missing)",
    "FC60_": "computed from age and sex only (999 = missing)",
    # Prevention
    "SEATBELT": _redundant("_RFSEAT2"),
    "_RFSEAT3": _redundant("_RFSEAT2"),
    "FLSHTMY2": "date of last flu shot (month-year code)",
    "IMFVPLAC": "place of last flu shot",
    "_FLSHOT6": _redundant("FLUSHOT6 (age 65+ only)"),
    "_PNEUMO2": _redundant("PNEUVAC3 (age 65+ only)"),
    "HIVTSTD3": "date of last HIV test (month-year code)",
    "WHRTST10": "place of last HIV test",
    "_AIDTST3": _redundant("HIVTST6"),
}
