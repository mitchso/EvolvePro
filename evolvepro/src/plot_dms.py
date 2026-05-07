"""
plot.py — Plotting and reporting utilities for EvolvePro.

This module provides helpers for two purposes:

1. **DMS simulation results** — reading and visualising the CSV output of
   ``grid_search`` / ``directed_evolution_simulation`` across datasets and
   hyper-parameter sweeps.
"""
import os
from typing import Dict, List, Optional

import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from evolvepro.src.data import load_experimental_data

pd.options.mode.chained_assignment = None


# ---------------------------------------------------------------------------
# DMS simulation data loading
# ---------------------------------------------------------------------------

def read_dms_data(
    directory: str,
    datasets: List[str],
    model: str,
    experiment: str,
    group_columns: List[str],
    aggregate_columns: List[str],
    file_pattern: str = '{dataset}_{model}_{experiment}.csv',
) -> pd.DataFrame:
    """Read and aggregate simulation result CSVs across multiple datasets.

    Expects one CSV per dataset, following the naming convention produced by
    ``grid_search``.  Each CSV is aggregated (mean + std) over ``group_columns``
    and concatenated into a single DataFrame.

    Args:
        directory: Directory containing the simulation result CSV files.
        datasets: List of dataset identifiers (e.g. ``["GFP", "Cas9"]``).
        model: Embedding model name used in file naming.
        experiment: Experiment label used in file naming.
        group_columns: Columns to group by before aggregating, e.g.
            ``["round_num", "regression_type"]``.
        aggregate_columns: Numeric columns to summarise, e.g.
            ``["top_activity_scaled", "activity_binary_percentage"]``.
        file_pattern: Python format string for constructing file names.  The
            default ``"{dataset}_{model}_{experiment}.csv"`` matches the output
            of ``grid_search``.

    Returns:
        Concatenated DataFrame with mean and std columns for each aggregate column,
        plus ``dataset``, ``model``, and ``experiment`` identifier columns.
        Returns an empty DataFrame if no files were found.
    """
    all_dfs = []
    for dataset in datasets:
        fname = file_pattern.format(dataset=dataset, model=model, experiment=experiment)
        fpath = os.path.join(directory, fname)
        try:
            df = pd.read_csv(fpath)
            df = _aggregate_dataframe(df, group_columns, aggregate_columns)
            df['dataset'] = dataset
            df['model'] = model
            df['experiment'] = experiment
            all_dfs.append(df)
        except FileNotFoundError:
            print(f'File not found, skipping: {fname}')
        except Exception as exc:
            print(f'Error reading {fname}: {exc}')

    return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()


def _aggregate_dataframe(
    df: pd.DataFrame,
    group_columns: List[str],
    aggregate_columns: List[str],
) -> pd.DataFrame:
    """Group a simulation DataFrame and compute per-group mean and std.

    ``"None"`` strings (written by the simulation for round 0) are treated as
    missing values and excluded from aggregation.

    Args:
        df: Raw simulation results DataFrame.
        group_columns: Columns to group by.
        aggregate_columns: Numeric columns to aggregate.

    Returns:
        DataFrame with flattened ``<column>_mean`` / ``<column>_std`` columns.
    """
    df = df.replace('None', np.nan).dropna(subset=aggregate_columns, how='all')
    df[aggregate_columns] = df[aggregate_columns].apply(pd.to_numeric, errors='coerce')
    stats = df.groupby(group_columns)[aggregate_columns].agg(['mean', 'std'])
    stats.columns = [f'{col}_{stat}' for col, stat in stats.columns]
    return stats.reset_index()

