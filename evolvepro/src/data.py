"""
data.py — File loading and data alignment for EvolvePro.

This module handles two distinct input scenarios:

  1. DMS (deep mutational scanning) datasets — pre-processed CSV/PT files used for
     benchmarking active-learning strategies in simulation.

  2. Experimental datasets — Excel round files produced by wet-lab assays, used to
     guide real directed-evolution campaigns.

See README.md for full file format specifications.
"""

import os
import warnings
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from Bio import SeqIO
import re as _re


def _is_valid_variant(value: str) -> bool:
    """Return True if *value* is 'WT' or valid mutation notation (single or multi).

    Valid forms:
      - ``"WT"``
      - A single mutation token: one uppercase letter, one or more digits, one
        uppercase letter (e.g. ``"A107M"``).
      - Multiple such tokens joined by underscores (e.g. ``"A107M_F73C"``).
    """
    if value == 'WT':
        return True
    tokens = value.split('_')
    _MUTATION_TOKEN_RE = _re.compile(r'^[A-Z]\d+[A-Z]$')
    return all(_MUTATION_TOKEN_RE.match(t) for t in tokens)


def validate_csv(file_path: str, file_type: str) -> pd.DataFrame:
    """Load and validate a CSV file used by EvolvePro.

    Two ``file_type`` values are accepted:

    ``"experimental"``
        The CSV must have exactly one column named ``"variant"`` (case-insensitive)
        and one column named ``"activity"`` (case-insensitive).  The ``variant``
        column must contain only ``"WT"`` or standard mutation notation
        (``<WT residue><position><mut residue>``, optionally chained with ``_``).
        The ``activity`` column must be fully numeric.  Additional columns are
        allowed without restriction.

    ``"embeddings"``
        The first column is treated as the variant index.  It must contain only
        valid variant strings (same rules as above).  All remaining columns must
        be numeric.

    Args:
        file_path: Absolute or relative path to the CSV file.
        file_type: Either ``"experimental"`` or ``"embeddings"``.

    Returns:
        The loaded DataFrame, with column names normalised to lowercase for
        ``"experimental"`` files (``variant`` and ``activity`` are always
        lowercase on return).

    Raises:
        FileNotFoundError: If *file_path* does not exist.
        ValueError: If *file_type* is not one of the accepted values, or if
            the file fails any structural or content check.
    """
    valid_types = ('experimental', 'embeddings')
    if file_type not in valid_types:
        raise ValueError(
            f"file_type must be one of {valid_types}, got '{file_type}'."
        )

    if not os.path.isfile(file_path):
        raise FileNotFoundError(
            f"CSV file not found: '{file_path}'."
        )

    df = pd.read_csv(file_path) if file_type == 'embeddings' else pd.read_excel(file_path)

    if df.empty:
        raise ValueError(
            f"CSV file '{file_path}' is empty."
        )

    if file_type == 'experimental':
        # --- Normalise column names for case-insensitive matching ---
        lower_col_map = {c: c.lower() for c in df.columns}
        df = df.rename(columns=lower_col_map)

        if 'variant' not in df.columns:
            raise ValueError(
                f"Experimental CSV '{file_path}' must contain a 'variant' column "
                f"(case-insensitive). Found columns: {df.columns.tolist()}."
            )
        if 'activity' not in df.columns:
            raise ValueError(
                f"Experimental CSV '{file_path}' must contain an 'activity' column "
                f"(case-insensitive). Found columns: {df.columns.tolist()}."
            )

        # --- Validate variant column ---
        invalid = [v for v in df['variant'].astype(str) if not _is_valid_variant(v)]
        if invalid:
            shown = invalid[:10]
            suffix = ' (and more)' if len(invalid) > 10 else ''
            raise ValueError(
                f"Experimental CSV '{file_path}' contains invalid value(s) in the "
                f"'variant' column: {shown}{suffix}.\n"
                f"Each entry must be 'WT', a single mutation token "
                f"(e.g. 'A107M'), or multiple tokens joined by underscores "
                f"(e.g. 'A107M_F73C'). No other values are accepted."
            )

        # --- Validate activity column ---
        activity_numeric = pd.to_numeric(df['activity'], errors='coerce')
        bad_activity = df.loc[activity_numeric.isna() & df['activity'].notna(), 'activity'].tolist()
        if bad_activity:
            shown = bad_activity[:10]
            suffix = ' (and more)' if len(bad_activity) > 10 else ''
            raise ValueError(
                f"Experimental CSV '{file_path}': 'activity' column contains "
                f"non-numeric value(s): {shown}{suffix}."
            )

    else:  # embeddings
        # --- First column is the variant index ---
        variant_col = df.columns[0]
        invalid = [v for v in df[variant_col].astype(str) if not _is_valid_variant(v)]
        if invalid:
            shown = invalid[:10]
            suffix = ' (and more)' if len(invalid) > 10 else ''
            raise ValueError(
                f"Embeddings CSV '{file_path}' contains invalid value(s) in the "
                f"first (variant index) column '{variant_col}': {shown}{suffix}.\n"
                f"Each entry must be 'WT', a single mutation token "
                f"(e.g. 'A107M'), or multiple tokens joined by underscores "
                f"(e.g. 'A107M_F73C'). No other values are accepted."
            )

        # --- All remaining columns must be numeric ---
        data_cols = df.columns[1:]
        non_numeric = [
            c for c in data_cols
            if not pd.api.types.is_numeric_dtype(df[c])
        ]
        if non_numeric:
            raise ValueError(
                f"Embeddings CSV '{file_path}' contains non-numeric data in "
                f"column(s): {non_numeric}. All embedding dimensions must be numeric."
            )
        missing_vals = df[data_cols].isnull().sum().sum()
        if missing_vals > 0:
            raise ValueError(
                f"Embeddings CSV '{file_path}' contains {missing_vals} missing "
                f"value(s) in the embedding columns. All values must be present."
            )

        # Set the first column as the index to match expected caller conventions
        df = df.set_index(variant_col)

    return df


def load_experimental_embeddings(file: str) -> pd.DataFrame:

    embeddings = validate_csv(file, 'embeddings')
    #
    # if 'WT' not in embeddings.index:
    #     warnings.warn(
    #         f"No 'WT' row found in embeddings file '{embeddings_file_name}'. "
    #         f"A WT embedding is required. Ensure your FASTA included a WT record "
    #         f"before running PLM extraction.",
    #         UserWarning,
    #         stacklevel=2,
    #     )

    return embeddings

def load_experimental_data(file: str, wt_fasta_path: str) -> pd.DataFrame:
    df = validate_csv(file, 'experimental')
    return df


def create_iteration_dataframes(
    df_list: List[pd.DataFrame],
    expected_variants: List[str],
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Build the iteration-tracking and labels DataFrames from multiple round files.

    Each element of ``df_list`` represents one assay round (in chronological order).
    WT is assigned iteration 0 and must appear only in the first round file.
    All other variants are assigned an iteration equal to the 1-based round index.

    The resulting ``labels`` DataFrame covers *all* variants in the embeddings index
    (``expected_variants``).  Variants not yet tested have ``NaN`` for activity and
    iteration, and will be treated as the unlabelled pool for model predictions.

    Args:
        df_list: List of DataFrames produced by :func:`load_experimental_data`,
            one per assay round, in chronological order.
        expected_variants: Ordered list of every variant name in the embeddings
            index.  Determines the final row order of the returned ``labels``.

    Returns:
        ``(iteration, labels)`` on success, or raises on invalid input.

        - ``iteration`` — columns: ``variant``, ``iteration`` (float).
          Contains only the tested variants.
        - ``labels`` — columns: ``variant``, ``activity``, ``activity_binary``,
          ``activity_scaled``, ``iteration``.  Contains every variant in
          ``expected_variants``; untested rows have ``NaN`` activity/iteration.
    """
    if not df_list:
        raise ValueError(
            "df_list is empty. At least one round DataFrame must be provided."
        )

    processed_rounds = []

    for round_num, df in enumerate(df_list, start=1):
        df_copy = df.copy()

        wt_rows = df_copy[df_copy['variant'] == 'WT']

        if round_num == 1:
            if wt_rows.empty:
                warnings.warn(
                    f"Round 1 does not contain a 'WT' entry. WT should appear in "
                    f"the first round file with Variant = 'WT' (see README §4). "
                    f"Proceeding without a WT reference point.",
                    UserWarning,
                    stacklevel=2,
                )
            df_copy.loc[df_copy['variant'] == 'WT', 'iteration'] = 0
        else:
            if not wt_rows.empty:
                # WT in subsequent rounds is silently dropped per spec — warn instead.
                warnings.warn(
                    f"Round {round_num} contains a 'WT' entry. WT should only "
                    f"appear in round 1; the WT row will be dropped from round "
                    f"{round_num} (see README §4).",
                    UserWarning,
                    stacklevel=2,
                )
            df_copy = df_copy[df_copy['variant'] != 'WT']

        df_copy.loc[df_copy['variant'] != 'WT', 'iteration'] = float(round_num)
        df_copy['iteration'] = df_copy['iteration'].astype(float)
        processed_rounds.append(df_copy)

    combined = pd.concat(processed_rounds, ignore_index=True)

    # Raises ValueError on duplicates (no longer returns True/False).
    _check_duplicates(combined)

    # --- Warn about tested variants missing from the embeddings index ---
    tested_variants = set(combined['variant'])
    expected_set = set(expected_variants)
    missing_from_embeddings = tested_variants - expected_set - {'WT'}
    if missing_from_embeddings:
        shown = sorted(missing_from_embeddings)[:10]
        suffix = ' (and more)' if len(missing_from_embeddings) > 10 else ''
        warnings.warn(
            f"{len(missing_from_embeddings)} tested variant(s) are not present in "
            f"the embeddings index and will be excluded from model training: "
            f"{shown}{suffix}. "
            f"Ensure every tested variant has a corresponding row in the embeddings "
            f"CSV (see README §3, Step 4).",
            UserWarning,
            stacklevel=2,
        )

    iteration = combined[['variant', 'iteration']]

    labels = combined[['variant', 'activity', 'iteration']].copy()
    act_min = labels['activity'].min()
    act_max = labels['activity'].max()

    if act_max == act_min:
        warnings.warn(
            f"All measured activity values are identical ({act_min}). "
            f"'activity_scaled' will be 0.0 for every variant because min-max "
            f"scaling is undefined when max == min. Check that the 'activity' "
            f"column contains meaningful variation across your round files.",
            UserWarning,
            stacklevel=2,
        )

    labels['activity_binary'] = (labels['activity'] >= 1).astype(float)
    labels['activity_scaled'] = (labels['activity'] - act_min) / (act_max - act_min)

    # Inform the user how many variants are in the unlabelled prediction pool.
    n_untested = len(expected_set - tested_variants)
    if n_untested > 0:
        print(f"{n_untested} variant(s) in the embeddings index have not yet been "
            f"tested and will be treated as the unlabelled prediction pool.")

    labels = _add_missing_variants(labels, expected_variants)
    labels = (
        labels.set_index('variant')
        .reindex(expected_variants, fill_value=np.nan)
        .reset_index()
    )
    labels = labels.rename(columns={'index': 'variant'})

    return iteration, labels


def _check_duplicates(df: pd.DataFrame) -> None:
    """Raise ValueError if any variant appears more than once across all rounds.

    Args:
        df: Combined DataFrame with ``variant`` and ``iteration`` columns.
    """
    duplicates = df[df.duplicated(subset=['variant'], keep=False)]
    if not duplicates.empty:
        raise ValueError(
            f"Duplicate variant(s) found across round files — each variant may "
            f"only be measured once (see README §4):\n"
            + duplicates[['variant', 'iteration']].to_string(index=False)
            + "\nRemove the duplicate entry from the appropriate round file and retry."
        )


def _add_missing_variants(
    df: pd.DataFrame, expected_variants: List[str]
) -> pd.DataFrame:
    """Append rows for variants that appear in the embeddings but not yet in labels.

    Args:
        df: Existing labels DataFrame with columns ``variant``, ``activity``,
            ``activity_binary``, ``activity_scaled``, ``iteration``.
        expected_variants: Full list of variant names from the embeddings index.

    Returns:
        Extended DataFrame including placeholder rows (all ``NaN``) for untested
        variants.
    """
    missing = set(expected_variants) - set(df['variant'])
    if not missing:
        return df

    placeholder = pd.DataFrame({
        'variant': list(missing),
        'activity': np.nan,
        'activity_binary': np.nan,
        'activity_scaled': np.nan,
        'iteration': np.nan,
    })
    return pd.concat([df, placeholder], ignore_index=True)