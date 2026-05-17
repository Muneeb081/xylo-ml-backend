"""
Data Ingestion Layer
====================
Reads all 42 raw House*.csv files from PRECON/ and the Metadata.csv.
- Handles heterogeneous column schemas per house
- Resamples 1-min data to 15-min intervals
- Maps raw column names to canonical room/appliance groups
- Returns a unified dict {house_id: DataFrame} plus the metadata table
"""
from __future__ import annotations

import re
import sys
import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Resolve paths without circular import
_SRC_DIR = Path(__file__).resolve().parent
_BASE_DIR = _SRC_DIR.parent

RAW_DATA_DIR   = _BASE_DIR / "PRECON"
METADATA_PATH  = _BASE_DIR / "Metadata.csv"
RESAMPLE_FREQ  = "15min"
USAGE_COL      = "Usage_kW"

from src.config import CANONICAL_ROOM_PATTERNS, META_FEATURES


# ── Metadata ───────────────────────────────────────────────────────────────────

def load_metadata() -> pd.DataFrame:
    """Load and clean Metadata.csv. Index = house_id (1-based int)."""
    meta = pd.read_csv(METADATA_PATH)
    # First column is house name like "House 1"
    meta.columns = meta.columns.str.strip()
    name_col = meta.columns[0]
    meta["house_id"] = meta[name_col].str.extract(r"(\d+)").astype(int)
    meta = meta.set_index("house_id")

    # Numeric cleanup — strip spaces, coerce
    for col in meta.columns:
        if col != name_col:
            meta[col] = pd.to_numeric(meta[col].astype(str).str.strip(), errors="coerce")

    meta = meta.fillna(0)
    return meta


# ── Column canonicalization ────────────────────────────────────────────────────

def _map_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map raw appliance columns to canonical room groups by summing all matches.
    Adds a column for each canonical group; drops original raw appliance columns.
    Keeps 'Usage_kW' (total) and metadata columns unchanged.
    """
    protected = {USAGE_COL, "house_id", "n_rooms", "n_acs", "n_people", "area_sqft", "building_year"}
    raw_appliance_cols = [c for c in df.columns if c not in protected]

    canonical_vals: Dict[str, pd.Series] = {}
    assigned: set = set()

    for group, pattern in CANONICAL_ROOM_PATTERNS.items():
        matched = [c for c in raw_appliance_cols if re.search(pattern, c)]
        if matched:
            canonical_vals[group] = df[matched].sum(axis=1)
            assigned.update(matched)

    # Un-matched appliance columns → summed into "other_kw"
    unmatched = [c for c in raw_appliance_cols if c not in assigned]
    if unmatched:
        canonical_vals["other_kw"] = df[unmatched].sum(axis=1)

    # Drop raw appliance cols
    df = df.drop(columns=raw_appliance_cols, errors="ignore")

    # Add canonical groups
    for name, series in canonical_vals.items():
        df[name] = series.values

    return df


# ── Single-house loader ────────────────────────────────────────────────────────

def load_single_house(
    csv_path: Path,
    house_id: int,
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    """Load one house CSV → clean, resample, canonicalize."""

    df = pd.read_csv(
        csv_path,
        parse_dates=["Date_Time"],
        infer_datetime_format=True,
        low_memory=False,
    )
    df.columns = df.columns.str.strip()

    # Normalise timestamp column name
    ts_col = next((c for c in df.columns if "date" in c.lower() or "time" in c.lower()), None)
    if ts_col is None:
        raise ValueError(f"No timestamp column found in {csv_path.name}")
    df = df.rename(columns={ts_col: "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()

    # Convert all values to numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Rename Usage column if needed
    usage_candidates = [c for c in df.columns if re.search(r"(?i)usage|total|main", c) and "kw" in c.lower()]
    if USAGE_COL not in df.columns and usage_candidates:
        df = df.rename(columns={usage_candidates[0]: USAGE_COL})
    if USAGE_COL not in df.columns:
        # Derive from sum of all appliance kW
        kw_cols = [c for c in df.columns if "kw" in c.lower()]
        df[USAGE_COL] = df[kw_cols].sum(axis=1) if kw_cols else 0.0

    # Resample to 15-min
    df = df.resample(RESAMPLE_FREQ).mean()

    # Remove negative values (metering errors)
    for col in df.columns:
        df[col] = df[col].clip(lower=0)

    # Attach house_id
    df["house_id"] = house_id

    # Attach metadata features
    if house_id in metadata.index:
        row = metadata.loc[house_id]
        df["n_rooms"]       = float(row.get("Total_No_of_Rooms", 0))
        df["n_acs"]         = float(row.get("No_of_ACs", 0))
        df["n_people"]      = float(row.get("No_of_People", 0))
        df["area_sqft"]     = float(row.get("Property_Area_sqft", 0))
        df["building_year"] = float(row.get("Building_Year", 2000))
    else:
        df["n_rooms"] = df["n_acs"] = df["n_people"] = df["area_sqft"] = df["building_year"] = 0.0

    # Canonicalize appliance columns
    df = _map_columns(df)

    return df


# ── All houses ─────────────────────────────────────────────────────────────────

def load_all_houses(
    limit: Optional[int] = None,
    verbose: bool = True,
) -> Tuple[Dict[int, pd.DataFrame], pd.DataFrame]:
    """
    Load all House*.csv files from PRECON/.

    Parameters
    ----------
    limit   : if set, load only the first `limit` houses (for quick testing)
    verbose : print progress

    Returns
    -------
    houses  : dict {house_id: DataFrame}
    metadata: DataFrame indexed by house_id
    """
    metadata = load_metadata()
    csv_files = sorted(RAW_DATA_DIR.glob("House*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No House*.csv files found in {RAW_DATA_DIR}")

    if limit:
        csv_files = csv_files[:limit]

    houses: Dict[int, pd.DataFrame] = {}

    for csv_path in csv_files:
        m = re.search(r"House(\d+)", csv_path.name)
        if not m:
            continue
        house_id = int(m.group(1))
        try:
            df = load_single_house(csv_path, house_id, metadata)
            houses[house_id] = df
            if verbose:
                print(f"  [OK] House {house_id:2d}: {len(df):>7,} rows | "
                      f"cols={len(df.columns)} | "
                      f"{df.index.min().date()} → {df.index.max().date()}")
        except Exception as exc:
            if verbose:
                print(f"  [WARN] House {house_id}: {exc}")

    if verbose:
        total_rows = sum(len(d) for d in houses.values())
        print(f"\n  Loaded {len(houses)} houses, {total_rows:,} total 15-min rows")

    return houses, metadata


# ── Firebase / real-time JSON adapter ─────────────────────────────────────────

def firebase_json_to_df(payload: dict) -> pd.DataFrame:
    """
    Convert a Firebase Realtime Database JSON payload to a single-row DataFrame
    compatible with the inference pipeline.

    Expected Firebase payload format:
    {
        "house_id": 1,
        "timestamp": "2024-01-15T14:30:00",
        "Usage_kW": 2.5,
        "AC_DR_kW": 1.2,
        "Kitchen_kW": 0.4,
        ...
    }
    Any appliance key accepted — they are canonicalized automatically.
    """
    ts = pd.to_datetime(payload.get("timestamp", pd.Timestamp.now()))
    row = {k: float(v) for k, v in payload.items()
           if k not in ("house_id", "timestamp") and isinstance(v, (int, float))}
    row["house_id"] = int(payload.get("house_id", 0))

    df = pd.DataFrame([row], index=[ts])
    df.index.name = "timestamp"

    # Ensure Usage_kW exists
    if USAGE_COL not in df.columns:
        kw_cols = [c for c in df.columns if "kw" in c.lower()]
        df[USAGE_COL] = df[kw_cols].sum(axis=1) if kw_cols else 0.0

    df = _map_columns(df)
    return df
