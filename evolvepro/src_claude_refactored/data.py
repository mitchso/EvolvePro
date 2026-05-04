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
import torch
from Bio import SeqIO


# ---------------------------------------------------------------------------
# DMS data loading
# ---------------------------------------------------------------------------

def load_dms_data(
    dataset_name: str,
    model_name: str,
    embeddings_path: str,
    labels_path: str,
    embeddings_file_type: str,
    embeddings_type_pt: str = 'both',
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Load and align embeddings and labels for a DMS benchmark dataset.

    Embeddings and labels are sorted by variant name and cross-filtered so that
    only variants present in both files are retained.

    Args:
        dataset_name: Short identifier for the protein / dataset (e.g. ``"GFP"``).
            Used to construct file names as ``<dataset_name>_labels.csv`` and
            ``<dataset_name>_<model_name>.<embeddings_file_type>``.
        model_name: Name of the embedding model (e.g. ``"esm2_t33_650M_UR50D"``).
        embeddings_path: Directory containing the embeddings file.
        labels_path: Directory containing the labels CSV.
        embeddings_file_type: ``"csv"`` or ``"pt"``.

            - ``"csv"`` — a numeric matrix with variants as the row index and
              embedding dimensions as columns (see README §3).
            - ``"pt"`` — a PyTorch file containing a dict mapping variant names
              to tensors with keys ``"average"`` and/or ``"mutated"``.

        embeddings_type_pt: Which tensor(s) to use from a ``.pt`` file:
            ``"average"``, ``"mutated"``, or ``"both"`` (concatenated).
            Ignored when ``embeddings_file_type="csv"``.

    Returns:
        ``(embeddings, labels)`` — both DataFrames aligned on the ``variant``
        column/index, or ``(None, None)`` on failure.
    """
    # --- Validate embeddings_file_type early ---
    valid_file_types = ('csv', 'pt')
    if embeddings_file_type not in valid_file_types:
        raise ValueError(
            f"Invalid embeddings_file_type '{embeddings_file_type}'. "
            f"Must be one of: {valid_file_types}."
        )

    labels_file = os.path.join(labels_path, f'{dataset_name}_labels.csv')
    embeddings_file = os.path.join(
        embeddings_path, f'{dataset_name}_{model_name}.{embeddings_file_type}'
    )

    # --- Validate file existence ---
    if not os.path.isfile(labels_file):
        raise FileNotFoundError(
            f"Labels file not found: '{labels_file}'. "
            f"Expected a CSV named '<dataset_name>_labels.csv' in labels_path."
        )
    if not os.path.isfile(embeddings_file):
        raise FileNotFoundError(
            f"Embeddings file not found: '{embeddings_file}'. "
            f"Expected '<dataset_name>_<model_name>.{embeddings_file_type}' "
            f"in embeddings_path."
        )

    # --- Load and validate labels ---
    labels = pd.read_csv(labels_file)

    required_label_cols = {'variant', 'activity'}
    missing_cols = required_label_cols - set(labels.columns)
    if missing_cols:
        raise ValueError(
            f"Labels file '{labels_file}' is missing required column(s): "
            f"{sorted(missing_cols)}. "
            f"Expected columns: 'variant', 'activity' (and optionally "
            f"'activity_scaled', 'activity_binary') — see README §6."
        )

    # --- Load embeddings ---
    if embeddings_file_type == 'csv':
        embeddings = pd.read_csv(embeddings_file, index_col=0)

        non_numeric_cols = [
            col for col in embeddings.columns
            if not pd.api.types.is_numeric_dtype(embeddings[col])
        ]
        if non_numeric_cols:
            raise ValueError(
                f"Embeddings file '{embeddings_file}' contains non-numeric values "
                f"in column(s): {non_numeric_cols}. "
                f"All embedding dimensions must be numeric (see README §3)."
            )

        missing_vals = embeddings.isnull().sum().sum()
        if missing_vals > 0:
            raise ValueError(
                f"Embeddings file '{embeddings_file}' contains {missing_vals} "
                f"missing value(s). All embedding dimensions must be numeric "
                f"with no missing entries (see README §3)."
            )

    elif embeddings_file_type == 'pt':
        raw = torch.load(embeddings_file)
        processed = _process_pt_embeddings(raw, embeddings_type_pt)
        if processed is None:
            return None, None
        embeddings = pd.DataFrame.from_dict(processed, orient='index')

    # --- Filter labels to those with measured activity, with a warning ---
    n_before = len(labels)
    labels = labels[labels['activity'].notna()]
    n_dropped = n_before - len(labels)
    if n_dropped > 0:
        warnings.warn(
            f"Dropped {n_dropped} row(s) from '{labels_file}' with missing "
            f"'activity' values. {len(labels)} variants remain.",
            UserWarning,
            stacklevel=2,
        )

    # --- Cross-filter: warn about variants present in only one file ---
    label_variants = set(labels['variant'])
    embedding_variants = set(embeddings.index)

    only_in_labels = label_variants - embedding_variants
    only_in_embeddings = embedding_variants - label_variants

    if only_in_labels:
        warnings.warn(
            f"{len(only_in_labels)} variant(s) are in the labels file but have no "
            f"embedding and will be excluded: {sorted(only_in_labels)}. "
            f"Ensure every labelled variant has a corresponding embedding row "
            f"(see README §3).",
            UserWarning,
            stacklevel=2,
        )

    if only_in_embeddings:
        warnings.warn(
            f"{len(only_in_embeddings)} variant(s) are in the embeddings file but "
            f"not in the labels file and will be excluded: "
            f"{sorted(only_in_embeddings)}. "
            f"This is expected if the embeddings cover candidates absent from the "
            f"DMS labels.",
            UserWarning,
            stacklevel=2,
        )

    embeddings = embeddings[embeddings.index.isin(labels['variant'])]

    # --- Sort both by variant so rows correspond to the same variant ---
    labels = labels.sort_values(by='variant').reset_index(drop=True)
    embeddings = embeddings.loc[labels['variant']]

    if labels['variant'].tolist() == embeddings.index.tolist():
        print('Embeddings and labels are aligned.')
        return embeddings, labels

    raise RuntimeError(
        'Embeddings and labels could not be aligned after sorting. '
        'Check that variant names are identical strings in both files '
        '(e.g. "A107M" not "107M") — see README §5.'
    )


def _process_pt_embeddings(
    embeddings: Dict, embeddings_type_pt: str
) -> Optional[Dict]:
    """Convert a raw PyTorch embeddings dict to a plain numpy dict.

    Args:
        embeddings: Dict mapping variant name → dict with ``"average"`` and/or
            ``"mutated"`` tensors.
        embeddings_type_pt: Which representation to use: ``"average"``,
            ``"mutated"``, or ``"both"`` (average and mutated concatenated).

    Returns:
        Dict mapping variant name → 1-D numpy array, or ``None`` on bad input.
    """
    valid_types = ('average', 'mutated', 'both')
    if embeddings_type_pt not in valid_types:
        raise ValueError(
            f"Invalid embeddings_type_pt '{embeddings_type_pt}'. "
            f"Must be one of: {valid_types}."
        )

    required_keys = {'average', 'mutated'} if embeddings_type_pt == 'both' else {embeddings_type_pt}
    malformed = [
        k for k, v in embeddings.items()
        if not isinstance(v, dict) or not required_keys.issubset(v.keys())
    ]
    if malformed:
        shown = malformed[:10]
        suffix = ' (and more)' if len(malformed) > 10 else ''
        raise ValueError(
            f"The following variant(s) in the .pt embeddings file are missing "
            f"the required key(s) {sorted(required_keys)}: {shown}{suffix}. "
            f"Each entry must be a dict with keys 'average' and/or 'mutated'."
        )

    if embeddings_type_pt == 'average':
        return {key: val['average'].numpy() for key, val in embeddings.items()}
    if embeddings_type_pt == 'mutated':
        return {key: val['mutated'].numpy() for key, val in embeddings.items()}
    # 'both'
    return {
        key: np.concatenate((val['average'].numpy(), val['mutated'].numpy()))
        for key, val in embeddings.items()
    }


# ---------------------------------------------------------------------------
# Experimental data loading
# ---------------------------------------------------------------------------

def load_experimental_embeddings(
    base_path: str,
    embeddings_file_name: str,
    rename_WT: bool = False,
) -> pd.DataFrame:
    """Load a pre-computed embeddings CSV for an experimental protein.

    The CSV must have variant names as the row index (first column) and numeric
    embedding dimensions as the remaining columns (see README §3).

    Args:
        base_path: Directory containing the embeddings file.
        embeddings_file_name: File name of the embeddings CSV, e.g.
            ``"mutants_esm2_t33_650M_UR50D.csv"``.
        rename_WT: If ``True``, rename the row ``"WT Wild-type sequence"`` to
            ``"WT"``. Needed when the embedding model used the long description
            as the sequence ID.

    Returns:
        DataFrame indexed by variant name.
    """
    file_path = os.path.join(base_path, embeddings_file_name)

    if not os.path.isfile(file_path):
        raise FileNotFoundError(
            f"Embeddings file not found: '{file_path}'. "
            f"Check that embeddings_base_path and embeddings_file_name are correct."
        )

    embeddings = pd.read_csv(file_path, index_col=0)

    if embeddings.empty:
        raise ValueError(
            f"Embeddings file '{file_path}' is empty. "
            f"Expected a CSV with variant names as the row index and numeric "
            f"embedding dimensions as columns (see README §3)."
        )

    non_numeric_cols = [
        col for col in embeddings.columns
        if not pd.api.types.is_numeric_dtype(embeddings[col])
    ]
    if non_numeric_cols:
        raise ValueError(
            f"Embeddings file '{file_path}' contains non-numeric values in "
            f"column(s): {non_numeric_cols}. "
            f"All embedding dimensions must be numeric with no missing entries "
            f"(see README §3)."
        )

    missing_vals = embeddings.isnull().sum().sum()
    if missing_vals > 0:
        raise ValueError(
            f"Embeddings file '{file_path}' contains {missing_vals} missing "
            f"value(s). All embedding dimensions must be numeric with no missing "
            f"entries (see README §3)."
        )

    # --- WT row handling ---
    has_wt = 'WT' in embeddings.index
    has_wt_long = 'WT Wild-type sequence' in embeddings.index

    if rename_WT:
        if has_wt_long:
            embeddings = embeddings.rename(index={'WT Wild-type sequence': 'WT'})
        elif not has_wt:
            warnings.warn(
                f"rename_WT=True was set for '{embeddings_file_name}', but no row "
                f"named 'WT Wild-type sequence' was found. The flag had no effect. "
                f"If your WT row already has the correct name 'WT', set rename_WT=False.",
                UserWarning,
                stacklevel=2,
            )
    elif has_wt_long and not has_wt:
        warnings.warn(
            f"Embeddings file '{embeddings_file_name}' contains a row named "
            f"'WT Wild-type sequence' but not 'WT'. The model expects the WT row "
            f"to be named 'WT'. Pass rename_WT=True to rename it automatically "
            f"(see README §3, Step 3).",
            UserWarning,
            stacklevel=2,
        )

    if 'WT' not in embeddings.index:
        warnings.warn(
            f"No 'WT' row found in embeddings file '{embeddings_file_name}' after "
            f"applying rename_WT={rename_WT}. A WT embedding is required. "
            f"Ensure your FASTA included a WT record before running PLM extraction.",
            UserWarning,
            stacklevel=2,
        )

    return embeddings


def load_experimental_data(
    base_path: str,
    round_file_name: str,
    wt_fasta_path: str,
    single_mutant: bool = True,
) -> pd.DataFrame:
    """Load one round of wet-lab assay results from an Excel file.

    The Excel file must contain at minimum:

    - A ``Variant`` column with variant identifiers.
    - An ``activity`` column with the measured assay value.

    **Single-mutant mode** (``single_mutant=True``, the default):
        The ``Variant`` column uses a *position-only* format, e.g. ``"107M"`` for a
        mutation to methionine at position 107.  The WT amino acid is looked up from
        the FASTA and prepended automatically, producing ``"A107M"``.  The special
        value ``"WT"`` is passed through unchanged.

    **Multi-mutant mode** (``single_mutant=False``):
        The ``Variant`` column must already contain full variant names, e.g.
        ``"A107M_F73C"``.  No conversion is applied.

    Args:
        base_path: Directory containing the round Excel file.
        round_file_name: File name, e.g. ``"D4_round_1.xlsx"``.
        wt_fasta_path: Path to the wild-type FASTA file.  Used to look up the WT
            amino acid at each mutated position (single-mutant mode only).
        single_mutant: Whether to apply single-mutant variant name conversion.

    Returns:
        DataFrame with an ``updated_variant`` column (converted variant names) plus
        all original columns from the Excel file.
    """
    file_path = os.path.join(base_path, round_file_name)

    if not os.path.isfile(file_path):
        raise FileNotFoundError(
            f"Round file not found: '{file_path}'. "
            f"Check that round_base_path and the file name are correct."
        )

    if not round_file_name.lower().endswith('.xlsx'):
        raise ValueError(
            f"Round file '{round_file_name}' does not have a .xlsx extension. "
            f"Only Excel (.xlsx) files are supported as round files (see README §4)."
        )

    df = pd.read_excel(file_path)

    # --- Validate required columns ---
    required_cols = {'Variant', 'activity'}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"Round file '{round_file_name}' is missing required column(s): "
            f"{sorted(missing_cols)}. "
            f"Each round file must contain at least 'Variant' and 'activity' "
            f"columns (see README §4)."
        )

    # --- Warn about and drop rows with missing activity ---
    n_before = len(df)
    df = df[df['activity'].notna()]
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        warnings.warn(
            f"Round file '{round_file_name}': dropped {n_dropped} row(s) with "
            f"missing 'activity' values. {len(df)} row(s) remain.",
            UserWarning,
            stacklevel=2,
        )

    if df.empty:
        raise ValueError(
            f"Round file '{round_file_name}' contains no rows with a valid "
            f"'activity' value after dropping missing entries. "
            f"Ensure the file is correctly populated."
        )

    # --- Load and validate the WT FASTA ---
    if not os.path.isfile(wt_fasta_path):
        raise FileNotFoundError(
            f"Wild-type FASTA file not found: '{wt_fasta_path}'. "
            f"Provide the path to the WT FASTA used to generate the candidate "
            f"library (see README §1)."
        )

    records = list(SeqIO.parse(wt_fasta_path, 'fasta'))
    if len(records) == 0:
        raise ValueError(
            f"Wild-type FASTA file '{wt_fasta_path}' contains no records. "
            f"The file must contain exactly one FASTA record for the WT protein "
            f"sequence (see README §1)."
        )
    if len(records) > 1:
        warnings.warn(
            f"Wild-type FASTA file '{wt_fasta_path}' contains {len(records)} "
            f"records; only the first record will be used as the WT sequence. "
            f"Remove extra records to avoid ambiguity.",
            UserWarning,
            stacklevel=2,
        )

    wt_sequence = str(records[0].seq)

    if single_mutant:
        # --- Validate all variant strings before conversion ---
        invalid_variants = []
        for v in df['Variant']:
            if v == 'WT':
                continue
            try:
                _validate_single_mutant_variant(str(v), wt_sequence)
            except (ValueError, IndexError) as exc:
                invalid_variants.append((v, str(exc)))

        if invalid_variants:
            shown = '\n  '.join(f"'{v}': {msg}" for v, msg in invalid_variants[:10])
            suffix = '\n  (and more)' if len(invalid_variants) > 10 else ''
            raise ValueError(
                f"Round file '{round_file_name}' contains variant(s) that could "
                f"not be parsed as position-only single-mutant format "
                f"(e.g. '107M' means Met at position 107):\n"
                f"  {shown}{suffix}\n"
                f"If these are multi-mutant variants, pass them via "
                f"round_file_names_multi with single_mutant=False (see README §4)."
            )

        df['updated_variant'] = df['Variant'].apply(
            lambda x: _convert_single_mutant_variant(x, wt_sequence)
        )
    else:
        # Multi-mutant names are already fully specified; just rename the column.
        # Warn if entries appear to use the position-only format by mistake.
        position_only = [
            v for v in df['Variant']
            if v != 'WT' and _looks_like_position_only(str(v))
        ]
        if position_only:
            shown = position_only[:10]
            suffix = ' (and more)' if len(position_only) > 10 else ''
            warnings.warn(
                f"Round file '{round_file_name}' is loaded in multi-mutant mode "
                f"(single_mutant=False), but the following variant(s) look like "
                f"position-only single-mutant format (e.g. '107M'): {shown}{suffix}. "
                f"If these are single-mutant rounds, pass them via "
                f"round_file_names_single (see README §4).",
                UserWarning,
                stacklevel=2,
            )
        df = df.rename(columns={'Variant': 'updated_variant'})

    return df


def _validate_single_mutant_variant(variant: str, wt_sequence: str) -> None:
    """Raise ValueError if a variant string cannot be parsed as position-only format.

    Args:
        variant: Raw variant string, expected to be e.g. ``"107M"``.
        wt_sequence: Full wild-type protein sequence.
    """
    if len(variant) < 2:
        raise ValueError(
            f"Too short to be a valid position-only variant (expected e.g. '107M')."
        )
    position_str = variant[:-1]
    if not position_str.isdigit():
        raise ValueError(
            f"Position '{position_str}' is not a valid integer. "
            f"Expected format: '<position><amino_acid>', e.g. '107M'. "
            f"Do not include the WT amino acid in the Variant column for "
            f"single-mutant mode (see README §4)."
        )
    position = int(position_str)
    if position < 1 or position > len(wt_sequence):
        raise ValueError(
            f"Position {position} is out of range for the WT sequence "
            f"(length {len(wt_sequence)}). Positions are 1-based."
        )


def _looks_like_position_only(variant: str) -> bool:
    """Return True if the variant string looks like position-only format (e.g. '107M')."""
    return len(variant) >= 2 and variant[:-1].isdigit()


def _convert_single_mutant_variant(variant: str, wt_sequence: str) -> str:
    """Convert a position-only variant label to standard notation.

    The Excel ``Variant`` column for single mutants uses only the position and
    mutant amino acid (e.g. ``"107M"``).  This function prepends the WT amino acid
    found at that position in the FASTA, yielding ``"A107M"``.

    Args:
        variant: Raw variant string from the Excel file, e.g. ``"107M"`` or ``"WT"``.
        wt_sequence: Full wild-type protein sequence (1-letter amino acids).

    Returns:
        Standard variant name, e.g. ``"A107M"``, or ``"WT"`` unchanged.
    """
    if variant == 'WT':
        return variant
    position = int(variant[:-1])          # strip the mutant amino acid suffix
    wt_aa = wt_sequence[position - 1]    # 1-based → 0-based index
    return wt_aa + variant               # e.g. "A" + "107M" → "A107M"


# ---------------------------------------------------------------------------
# Iteration / labels dataframe construction
# ---------------------------------------------------------------------------

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

        if 'updated_variant' not in df_copy.columns:
            raise ValueError(
                f"Round {round_num} DataFrame is missing the 'updated_variant' "
                f"column. Ensure it was produced by load_experimental_data()."
            )
        if 'activity' not in df_copy.columns:
            raise ValueError(
                f"Round {round_num} DataFrame is missing the 'activity' column."
            )

        wt_rows = df_copy[df_copy['updated_variant'] == 'WT']

        if round_num == 1:
            if wt_rows.empty:
                warnings.warn(
                    f"Round 1 does not contain a 'WT' entry. WT should appear in "
                    f"the first round file with Variant = 'WT' (see README §4). "
                    f"Proceeding without a WT reference point.",
                    UserWarning,
                    stacklevel=2,
                )
            df_copy.loc[df_copy['updated_variant'] == 'WT', 'iteration'] = 0
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
            df_copy = df_copy[df_copy['updated_variant'] != 'WT']

        df_copy.loc[df_copy['updated_variant'] != 'WT', 'iteration'] = float(round_num)
        df_copy['iteration'] = df_copy['iteration'].astype(float)
        df_copy = df_copy.rename(columns={'updated_variant': 'variant'})
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
        print(
            f"{n_untested} variant(s) in the embeddings index have not yet been "
            f"tested and will be treated as the unlabelled prediction pool."
        )

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
