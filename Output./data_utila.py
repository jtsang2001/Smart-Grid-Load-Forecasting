"""
data_utils.py
"""

import warnings
from typing import Tuple, List

import numpy as np
import pandas as pd
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer

# Canadian statutory holidays (Federal + Ontario Provincial) 2015-2026
def _build_holiday_set(years: range) -> set:
    """Return a set of datetime.date objects that are statutory holidays."""
    from dateutil.relativedelta import relativedelta, MO

    holidays = set()
    for y in years:
        # New Year's Day
        holidays.add(pd.Timestamp(y, 1, 1).date())
        # Family Day (3rd Monday in February – Ontario)
        d = pd.Timestamp(y, 2, 1) + pd.offsets.Week(weekday=0) * 2
        # find the 3rd Monday
        d = pd.Timestamp(y, 2, 1)
        mondays = 0
        while mondays < 3:
            if d.weekday() == 0:
                mondays += 1
            if mondays < 3:
                d += pd.Timedelta(days=1)
        holidays.add(d.date())
        # Good Friday – third Friday before Easter
        easter = _easter(y)
        holidays.add((easter - pd.Timedelta(days=2)).date())
        # Victoria Day (Monday before May 25)
        d = pd.Timestamp(y, 5, 25)
        while d.weekday() != 0:
            d -= pd.Timedelta(days=1)
        holidays.add(d.date())
        # Canada Day
        holidays.add(pd.Timestamp(y, 7, 1).date())
        # Civic Holiday (1st Monday in August)
        d = pd.Timestamp(y, 8, 1)
        while d.weekday() != 0:
            d += pd.Timedelta(days=1)
        holidays.add(d.date())
        # Labour Day (1st Monday in September)
        d = pd.Timestamp(y, 9, 1)
        while d.weekday() != 0:
            d += pd.Timedelta(days=1)
        holidays.add(d.date())
        # Thanksgiving (2nd Monday in October)
        d = pd.Timestamp(y, 10, 1)
        mondays = 0
        while mondays < 2:
            if d.weekday() == 0:
                mondays += 1
            if mondays < 2:
                d += pd.Timedelta(days=1)
        holidays.add(d.date())
        # Remembrance Day
        holidays.add(pd.Timestamp(y, 11, 11).date())
        # Christmas
        holidays.add(pd.Timestamp(y, 12, 25).date())
        # Boxing Day
        holidays.add(pd.Timestamp(y, 12, 26).date())
    return holidays

def _easter(year: int) -> pd.Timestamp:
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return pd.Timestamp(year, month, day)


#
def prepare_data(
    path: str,
    target_col: str,
) -> Tuple[pd.DataFrame, List[str], List[str]]:

    target_col = target_col.lower()

    # Load
    df = pd.read_csv(path)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Timestamp
    df["timestamp"] = pd.to_datetime(df["date"], format="mixed")
    df = df.drop(columns=["date"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Impute Solar NaNs (9 missing) with rolling-median interpolation
    if "solar" in df.columns:
        df["solar"] = (
            df["solar"]
            .interpolate(method="linear", limit_direction="both")
            .fillna(0.0)
        )

    #Cast to int
    int_cols = df.select_dtypes(
        include=["int8", "int16", "int32", "int64"]
    ).columns.tolist()
    df[int_cols] = df[int_cols].astype("float32")

    # Cyclic calendar features 
    hour   = df["timestamp"].dt.hour
    dow    = df["timestamp"].dt.dayofweek        # 0=Monday
    doy    = df["timestamp"].dt.dayofyear
    month  = df["timestamp"].dt.month

    df["hour_sin"]   = np.sin(2 * np.pi * hour   / 24)
    df["hour_cos"]   = np.cos(2 * np.pi * hour   / 24)
    df["dow_sin"]    = np.sin(2 * np.pi * dow    /  7)
    df["dow_cos"]    = np.cos(2 * np.pi * dow    /  7)
    df["doy_sin"]    = np.sin(2 * np.pi * doy    / 365)
    df["doy_cos"]    = np.cos(2 * np.pi * doy    / 365)
    df["month_sin"]  = np.sin(2 * np.pi * (month - 1) / 12)
    df["month_cos"]  = np.cos(2 * np.pi * (month - 1) / 12)
    df["is_weekend"] = (dow >= 5).astype(float)

    # Statutory holiday flag
    years = range(df["timestamp"].dt.year.min(), df["timestamp"].dt.year.max() + 2)
    holiday_set = _build_holiday_set(years)
    df["is_holiday"] = df["timestamp"].dt.date.isin(holiday_set).astype(float)

    # Combined flag used by electricity load models
    df["is_off_peak_day"] = ((df["is_weekend"] + df["is_holiday"]) > 0).astype(float)


    df["hour_month_sin"] = np.sin(2 * np.pi * hour / 24) * np.sin(2 * np.pi * (month - 1) / 12)
    df["hour_month_cos"] = np.cos(2 * np.pi * hour / 24) * np.cos(2 * np.pi * (month - 1) / 12)

    #Captures year trend, as usage increases every year
    df["year_trend"] = ((df["timestamp"].dt.year - 2015) / 10.0).astype("float32")

    #Lag features
    for col in [target_col] + [
        c for c in ["nuclear", "gas", "hydro", "wind", "solar",
                    "biofuel", "total_output", "ontario_demand", "market_demand"]
        if c in df.columns and c != target_col
    ]:
        pass  # iterate below over target only to keep feature count manageable

    global_mean = df[target_col].mean()
    for lag in [1, 24, 48, 168]:
        col_name = f"{target_col}_lag{lag}"
        df[col_name] = (
            df[target_col]
            .shift(lag)
            .fillna(method="bfill")
            .fillna(global_mean)
            .astype("float32")
        )

    #Rolling statistics
    for window, label in [(24, "24h"), (168, "168h")]:
        rolled = df[target_col].rolling(window, min_periods=1)
        df[f"{target_col}_rmean_{label}"] = rolled.mean().astype("float32")
        df[f"{target_col}_rstd_{label}"]  = rolled.std().fillna(0).astype("float32")


    df["group_id"] = "ontario"


    df["time_idx"] = np.arange(len(df), dtype=int)

    #L2: Known future reals
    known_reals: List[str] = [
        "hour_sin", "hour_cos",
        "dow_sin",  "dow_cos",
        "doy_sin",  "doy_cos",
        "month_sin","month_cos",
        "is_weekend",
        "is_holiday",
        "is_off_peak_day",
        "hour_month_sin",
        "hour_month_cos",
        "year_trend",        
    ]

    lag_cols      = [f"{target_col}_lag{l}"        for l in [1, 24, 48, 168]]
    rolling_cols  = [f"{target_col}_rmean_24h",  f"{target_col}_rstd_24h",
                     f"{target_col}_rmean_168h", f"{target_col}_rstd_168h"]

    gen_cols = ["nuclear", "gas", "hydro", "wind", "solar", "biofuel", "total_output"]
    other_target = (
        "market_demand" if target_col == "ontario_demand" else "ontario_demand"
    )
    #L1: Past IP includes generation, targets, lag, and rolling statistics
    unknown_reals: List[str] = (
        [c for c in gen_cols if c in df.columns]
        + ([other_target] if other_target in df.columns else [])
        + lag_cols
        + rolling_cols
    )
    
    known_reals   = [c for c in known_reals   if c in df.columns]
    unknown_reals = [c for c in unknown_reals if c in df.columns]

    return df, unknown_reals, known_reals

#Build dataset
def create_forecasting_dataset(
    df: pd.DataFrame,
    target_col: str,
    unknown_reals: List[str],
    known_reals: List[str],
    max_encoder_length: int,
    max_prediction_length: int,
    training_dataset: TimeSeriesDataSet = None,
) -> TimeSeriesDataSet:

    target_col = target_col.lower()

    if training_dataset is not None:
        return TimeSeriesDataSet.from_dataset(
            training_dataset,
            df,
            predict=False,
            stop_randomization=True,
        )

    return TimeSeriesDataSet(
        df,
        time_idx="time_idx",
        target=target_col,
        group_ids=["group_id"],
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        #L1: Known Future
        time_varying_known_reals=known_reals,
        time_varying_unknown_reals=unknown_reals + [target_col],
        target_normalizer=GroupNormalizer(
            groups=["group_id"],
            transformation="softplus",   # better than log1p for MW-scale targets
        ),
        #Append to L1 and L2
        add_relative_time_idx=True,
        #L0: Static Metadata
        add_target_scales=True,
        add_encoder_length=True,
        
        allow_missing_timesteps=False,
    )
