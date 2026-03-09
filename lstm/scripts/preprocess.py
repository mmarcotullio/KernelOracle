from __future__ import annotations

import argparse
import json
import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd

def encode_df(df: pd.DataFrame, vocab: Dict) -> pd.DataFrame:
    df = df.copy()

    df["pid_str"] = df["pid"].astype(str)
    df["pid_idx"] = df["pid_str"].map(vocab["pid_to_idx"]).astype(np.int64)

    df["prev_state_str"] = df["prev_state"].astype(str).fillna("UNK")

    if "UNK" not in vocab["state_to_idx"]:

        vocab["state_to_idx"]["UNK"] = len(vocab["state_to_idx"])
        vocab["idx_to_state"][vocab["state_to_idx"]["UNK"]] = "UNK"

    df["state_idx"] = df["prev_state_str"].map(lambda s: vocab["state_to_idx"].get(s, vocab["state_to_idx"]["UNK"])).astype(np.int64)


    df["cpu"] = pd.to_numeric(df.get("cpu", 0), errors="coerce").fillna(0.0).astype(np.float32)
    df["time_diff"] = pd.to_numeric(df.get("time_diff", 0), errors="coerce").fillna(0.0).astype(np.float32)


    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    return df