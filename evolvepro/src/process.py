"""
process.py — Raw data pre-processing and FASTA generation for EvolvePro.

This module handles two concerns:

1. **DMS pre-processing** — reading raw deep-mutational-scanning Excel / CSV files,
   applying activity cutoffs to create binary labels, and producing the labels CSV
   and mutant FASTA files consumed by the rest of the package.

2. **Experimental setup** — generating wild-type and mutant FASTA files from a
   protein sequence, and suggesting an initial panel of variants to test.

Typical usage order for a new experimental project:

    generate_wt(wt_sequence, "WT.fasta")
    generate_single_aa_mutants("WT.fasta", "mutants.fasta")
    # → run ESM / language model on mutants.fasta →
    suggest_initial_mutants("mutants.fasta", num_mutants=36, random_seed=42)
"""

import os
from itertools import combinations
from typing import Callable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord


# ---------------------------------------------------------------------------
# DMS data pre-processing
# ---------------------------------------------------------------------------

def process_dataset(
    file_path: str,
    wt_fasta_path: str,
    dataset_name: str,
    activity_column: str,
    cutoff_value: float,
    output_dir: str,
    sheet_name: Optional[str] = None,
    cutoff_rule: str = 'greater_than',
    cutoff_percentiles: Optional[List[float]] = None,
    cutoff_function: Optional[Callable] = None,
    AA_shift: Optional[int] = None,
    drop_columns: bool = False,
) -> Tuple[pd.DataFrame, List[float]]:
    """Pre-process a raw DMS file into the labels CSV and mutant FASTA used by EvolvePro.

    Reads an activity measurement table, generates a FASTA file of mutant sequences,
    adds normalised and binarised activity columns, and saves a labels CSV.

    **Input file format** (Excel or CSV):

    The file must contain at minimum:

    - A ``variant`` column with variant names in standard notation, e.g. ``"A107M"``
      (WT amino acid + 1-based position + mutant amino acid).  WT is represented
      as ``"WT"``.
    - The column named by ``activity_column`` with numeric activity values.

    Additional columns are preserved unless ``drop_columns=True``.

    **Output files** (written to ``output_dir``):

    - ``<dataset_name>_labels.csv`` — processed labels with ``variant``,
      ``activity``, ``activity_scaled``, and ``activity_binary`` columns.
    - ``<dataset_name>.fasta`` — one sequence per variant, including WT, for use
      with a protein language model to generate embeddings.

    Args:
        file_path: Path to the raw activity table (Excel ``.xlsx`` or ``.csv``).
        wt_fasta_path: Path to the wild-type FASTA file, used to verify and
            reconstruct mutant sequences.
        dataset_name: Short identifier used for output file names.
        activity_column: Column name in the input file containing the numeric
            activity readout.
        cutoff_value: Threshold value for the primary binary label
            (``activity_binary``).
        output_dir: Directory where output files will be written.
        sheet_name: Excel sheet name to read (pass ``None`` for the first sheet).
        cutoff_rule: How to apply ``cutoff_value``:

            - ``"greater_than"`` — variant is active if ``activity > cutoff_value``.
            - ``"less_than"`` — variant is active if ``activity < cutoff_value``.
            - ``"custom"`` — use ``cutoff_function`` (see below).

        cutoff_percentiles: Optional list of percentile values (0–100).  For each,
            an extra ``activity_binary_<p>p`` column is added using the same
            ``cutoff_rule``.
        cutoff_function: Required when ``cutoff_rule="custom"``.  Callable with
            signature ``fn(df, activity_column, cutoff) → boolean Series``.
        AA_shift: Positional offset applied when converting variant names to
            0-based sequence indices.  Use when the variant numbering in the
            input file is offset from the actual sequence position (e.g. if
            position 1 in the file corresponds to residue 5 in the sequence,
            pass ``AA_shift=5``).  ``None`` applies the standard ``position - 1``
            conversion.
        drop_columns: If ``True``, keep only the core activity columns in the
            saved CSV (removes any extra columns from the raw file).

    Returns:
        ``(filtered_df, fractions_above_cutoff)``

        - ``filtered_df`` — processed DataFrame (same content as the saved CSV).
        - ``fractions_above_cutoff`` — list of fractions of active variants at
          each cutoff (primary cutoff first, then any percentile cutoffs).
    """
    # --- Read input ---
    if file_path.endswith('.xlsx'):
        dataframe = (
            pd.read_excel(file_path, sheet_name=sheet_name)
            if isinstance(sheet_name, str)
            else pd.read_excel(file_path)
        )
    elif file_path.endswith('.csv'):
        dataframe = pd.read_csv(file_path)
    else:
        raise ValueError(
            f"Unsupported file format '{os.path.splitext(file_path)[1]}'. "
            "Provide an Excel (.xlsx) or CSV (.csv) file."
        )

    # Drop rows with missing activity measurements.
    filtered_df = dataframe.dropna(subset=[activity_column]).copy()

    # Generate the mutant FASTA.
    wt_sequence = next(SeqIO.parse(wt_fasta_path, 'fasta')).seq
    generate_mutation_fasta(filtered_df, wt_sequence, dataset_name, output_dir, AA_shift)

    # Normalise activity.
    filtered_df['activity'] = filtered_df[activity_column]
    filtered_df['activity_scaled'] = (
        (filtered_df[activity_column] - filtered_df[activity_column].min())
        / (filtered_df[activity_column].max() - filtered_df[activity_column].min())
    )

    # Apply cutoffs and record statistics.
    cutoff_percentiles = cutoff_percentiles or []
    all_cutoffs = [cutoff_value] + [
        np.percentile(filtered_df[activity_column], p) for p in cutoff_percentiles
    ]
    cutoff_labels = [''] + [f'_{p}p' for p in cutoff_percentiles]

    total = len(filtered_df)
    fractions_above = []
    numbers_above = []

    for cutoff, label in zip(all_cutoffs, cutoff_labels):
        col = f'activity_binary{label}'
        if cutoff_rule == 'greater_than':
            filtered_df[col] = (filtered_df[activity_column] > cutoff).astype(int)
        elif cutoff_rule == 'less_than':
            filtered_df[col] = (filtered_df[activity_column] < cutoff).astype(int)
        elif cutoff_rule == 'custom':
            if cutoff_function is None:
                raise ValueError(
                    "cutoff_function must be provided when cutoff_rule='custom'."
                )
            filtered_df[col] = cutoff_function(filtered_df, activity_column, cutoff).astype(int)
        else:
            raise ValueError(
                f"Invalid cutoff_rule '{cutoff_rule}'. "
                "Choose 'greater_than', 'less_than', or 'custom'."
            )
        n = filtered_df[col].sum()
        numbers_above.append(n)
        fractions_above.append(n / total if total > 0 else 0)

    print(f'Cutoff values:      {all_cutoffs}')
    print(f'Number above cutoff: {numbers_above}')
    print(f'Fraction above cutoff: {fractions_above}')

    if drop_columns:
        core_cols = (
            ['variant', activity_column, 'activity', 'activity_scaled']
            + [f'activity_binary{label}' for label in cutoff_labels]
        )
        filtered_df = filtered_df[core_cols]

    os.makedirs(output_dir, exist_ok=True)
    filtered_df.to_csv(
        os.path.join(output_dir, f'{dataset_name}_labels.csv'), index=False
    )

    return filtered_df, fractions_above


def generate_mutation_fasta(
    df: pd.DataFrame,
    wt_sequence,
    dataset_name: str,
    output_dir: str,
    AA_shift: Optional[int],
) -> None:
    """Write a FASTA file with the mutant sequence for every variant in ``df``.

    Variant names must be in standard notation: ``<WT AA><position><mutant AA>``,
    e.g. ``"A107M"``.  The position is 1-based.  If the WT amino acid at that
    position does not match, the variant is skipped and an error is printed.

    Args:
        df: DataFrame with a ``variant`` column.
        wt_sequence: Wild-type protein sequence (BioPython ``Seq`` or plain ``str``).
        dataset_name: Used to name the output file (``<dataset_name>.fasta``).
        output_dir: Directory where the FASTA will be written.
        AA_shift: Positional offset; see :func:`process_dataset` for details.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'{dataset_name}.fasta')

    with open(output_file, 'w') as f:
        for variant in df['variant']:
            if 'WT' in str(variant):
                f.write(f'>{variant}\n{wt_sequence}\n')
            else:
                shift = 1 if AA_shift is None else AA_shift
                position = int(variant[1:-1]) - shift   # 1-based → 0-based
                wt_aa = variant[0]
                mutant_aa = variant[-1]
                if wt_sequence[position] == wt_aa:
                    seq = wt_sequence[:position] + mutant_aa + wt_sequence[position + 1:]
                    f.write(f'>{variant}\n{seq}\n')
                else:
                    print(
                        f'Warning: WT residue at position {position + 1} '
                        f'is {wt_sequence[position]!r}, not {wt_aa!r}. '
                        f'Skipping {variant}.'
                    )


# ---------------------------------------------------------------------------
# DMS dataset-specific pre-processors
# ---------------------------------------------------------------------------

def markin_custom_cutoff(
    markin_df: pd.DataFrame, activity_column: str, cutoff: float
) -> pd.Series:
    """Custom binary-activity rule for the Markin et al. DMS dataset.

    A variant is considered active if its measured activity exceeds ``cutoff``
    AND its associated p-value is below 0.05.

    Args:
        markin_df: DataFrame with ``activity_column`` and a ``pvalue`` column.
        activity_column: Name of the activity column.
        cutoff: Activity threshold.

    Returns:
        Boolean Series (True = active).
    """
    return (markin_df[activity_column] > cutoff) & (markin_df['pvalue'] < 0.05)


def preprocess_cas12f(input_cas12f: str, preprocessed_output_file: str) -> None:
    """Pre-process the Cas12f DMS dataset into the standard format.

    Reads the raw Cas12f Excel file, renames and normalises columns, and saves
    a cleaned Excel file ready for :func:`process_dataset`.

    Args:
        input_cas12f: Path to the raw Cas12f Excel file.
        preprocessed_output_file: Path to write the cleaned Excel file.
    """
    df = pd.read_excel(input_cas12f)

    # Standardise the variant column name.
    df = df.rename(columns={'Mutation': 'variant'})

    # Normalise the activity column to [0, 1].
    activity_col = df.columns[1]
    df['activity'] = (
        (df[activity_col] - df[activity_col].min())
        / (df[activity_col].max() - df[activity_col].min())
    )

    df.to_excel(preprocessed_output_file, index=False)


def preprocess_cov2_S(input_cov2_S: str, preprocessed_output_file: str) -> None:
    """Pre-process the SARS-CoV-2 spike protein DMS dataset.

    Reads the raw escape-fraction CSV, builds standard variant names, averages
    across conditions, and saves a cleaned CSV.

    Args:
        input_cov2_S: Path to the raw DMS CSV (must contain ``wildtype``,
            ``site``, ``mutation``, and ``mut_escape`` columns).
        preprocessed_output_file: Path to write the cleaned CSV.
    """
    df = pd.read_csv(input_cov2_S)

    # Build standard variant names: e.g. "E484K".
    df['variant'] = df['wildtype'] + df['site'].astype(str) + df['mutation']

    df = df.drop(columns=['site_total_escape', 'site_max_escape', 'condition'])

    # Average escape fraction across antibody conditions.
    df_avg = (
        df.groupby(['variant', 'site', 'wildtype', 'mutation'])['mut_escape']
        .mean()
        .reset_index()
        .sort_values('site')
    )

    df_avg.to_csv(preprocessed_output_file, index=False)


# ---------------------------------------------------------------------------
# Plotting helpers (for DMS data exploration)
# ---------------------------------------------------------------------------

def plot_mutations_per_position(df: pd.DataFrame) -> None:
    """Bar chart: how many distinct mutations are present at each sequence position.

    Args:
        df: DataFrame with a ``variant`` column in standard notation.
    """
    df_filtered = df.dropna(subset=['variant']).query('variant != "WT"')
    counts: dict = {}
    for variant in df_filtered['variant']:
        pos = int(variant[1:-1])
        counts[pos] = counts.get(pos, 0) + 1

    plt.bar(counts.keys(), counts.values())
    plt.xlabel('Position')
    plt.ylabel('Number of mutations')
    plt.title('Mutations per position')
    plt.show()


def plot_histogram_of_readout(
    df: pd.DataFrame,
    column_name: str,
    cutoff: Optional[float] = None,
) -> None:
    """Histogram of activity values across all variants.

    Draws a vertical dashed line at the WT value (if present) and optionally at
    a specified cutoff.

    Args:
        df: DataFrame with ``variant`` and ``column_name`` columns.
        column_name: Activity column to plot.
        cutoff: Optional threshold to highlight on the histogram.
    """
    fig, ax = plt.subplots()
    ax.hist(df[column_name].values, bins=100)
    ax.set_xlabel(column_name)
    ax.set_ylabel('Number of mutants')
    ax.set_title(f'{column_name} distribution across mutants')

    if 'WT' in df['variant'].values:
        wt_val = df.loc[df['variant'] == 'WT', column_name].values[0]
        ax.axvline(wt_val, color='red', linestyle='--', label='WT')
    if cutoff is not None:
        ax.axvline(cutoff, color='black', linestyle='--', label='cutoff')

    ax.legend()
    plt.show()


# ---------------------------------------------------------------------------
# Experimental setup helpers
# ---------------------------------------------------------------------------

def generate_wt(wt_sequence: str, output_file: str) -> None:
    """Write a single-record FASTA file containing the wild-type sequence.

    The resulting file is used throughout the pipeline wherever ``wt_fasta_path``
    is required.

    Args:
        wt_sequence: Full wild-type protein sequence (one-letter amino acid codes).
        output_file: Destination path for the FASTA file, e.g. ``"WT.fasta"``.
    """
    record = SeqRecord(Seq(wt_sequence), id='WT', description='')
    with open(output_file, 'w') as fh:
        SeqIO.write(record, fh, 'fasta')


def generate_single_aa_mutants(
    wt_fasta: str,
    output_file: str,
    positions: Optional[List[int]] = None,
) -> None:
    """Generate all possible single amino-acid substitutions and save as FASTA.

    Iterates over every position (or a specified subset) and every amino acid
    in the standard 20-letter alphabet, creating one FASTA record per unique
    substitution.  The WT sequence is included as the first record.

    Variant names follow the standard convention: ``<WT AA><position><mutant AA>``
    (1-based), e.g. ``"A107M"``.

    This FASTA is the starting point for generating protein embeddings.

    Args:
        wt_fasta: Path to the wild-type FASTA file.
        output_file: Destination path for the mutant FASTA, e.g. ``"mutants.fasta"``.
        positions: 1-based list of positions to mutate.  ``None`` mutates all
            positions in the sequence.
    """
    aa_alphabet = 'ACDEFGHIKLMNPQRSTVWY'
    wt_sequence = SeqIO.read(wt_fasta, 'fasta').seq
    records = [SeqRecord(wt_sequence, id='WT', description='')]

    positions_to_mutate = (
        [p - 1 for p in positions]          # convert to 0-based
        if positions is not None
        else range(len(wt_sequence))
    )

    for i in positions_to_mutate:
        wt_aa = wt_sequence[i]
        for mut_aa in aa_alphabet:
            if mut_aa == wt_aa:
                continue
            mutant_seq = wt_sequence[:i] + mut_aa + wt_sequence[i + 1:]
            variant = f'{wt_aa}{i + 1}{mut_aa}'
            records.append(SeqRecord(Seq(mutant_seq), id=variant, description=''))

    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
    with open(output_file, 'w') as fh:
        SeqIO.write(records, fh, 'fasta')

    print(f'Mutant FASTA written: {len(records)} sequences → {output_file}')


def generate_n_mutant_combinations(
    wt_fasta: str,
    mutant_file: str,
    n: int,
    output_file: str,
    threshold: float = 1.0,
) -> None:
    """Generate all n-fold combinations of beneficial single mutants.

    Reads a list of single mutants from an Excel file, filters to those above a
    minimum activity threshold, and writes a FASTA containing every valid
    n-mutation combination (combinations that mutate the same position more than
    once are skipped).

    Multi-mutant variant names join single-mutant tokens with underscores:
    ``"A107M_F73C"``.

    Args:
        wt_fasta: Path to the wild-type FASTA file.
        mutant_file: Path to an Excel file with ``Variant`` and ``activity``
            columns.  ``Variant`` values should be in position-only format
            (e.g. ``"107M"``).  Only variants with
            ``activity > threshold`` are included.
        n: Number of mutations to combine (e.g. ``2`` for double mutants).
        output_file: Destination path for the output FASTA.
        threshold: Minimum single-mutant activity score for inclusion.
    """
    wt_sequence = str(SeqIO.read(wt_fasta, 'fasta').seq)

    mutants = pd.read_excel(mutant_file)
    mutants = mutants[mutants['activity'] > threshold].copy()

    # Convert position-only notation to standard notation.
    mutants[['position', 'mutant_aa']] = (
        mutants['Variant'].str.extract(r'(\d+)([A-Z]+)', expand=True)
    )
    mutants['wt_aa'] = mutants['position'].apply(
        lambda p: wt_sequence[int(p) - 1]
    )
    mutants['variant'] = mutants['wt_aa'] + mutants['position'] + mutants['mutant_aa']

    records = [SeqRecord(Seq(wt_sequence), id='WT', description='Wild-type sequence')]
    combos = list(combinations(mutants['variant'], n))
    valid_count = 0

    for combo in combos:
        positions_used: set = set()
        valid = True
        mutant_seq = wt_sequence
        name_parts = []

        for mutant in combo:
            wt_aa, pos_str, mut_aa = mutant[0], mutant[1:-1], mutant[-1]
            pos_idx = int(pos_str) - 1          # 0-based index

            if pos_idx in positions_used:       # conflicting mutations → skip
                valid = False
                break

            positions_used.add(pos_idx)
            mutant_seq = mutant_seq[:pos_idx] + mut_aa + mutant_seq[pos_idx + 1:]
            name_parts.append(mutant)

        if valid:
            records.append(
                SeqRecord(Seq(mutant_seq), id='_'.join(name_parts), description='')
            )
            valid_count += 1

    print(f'Combinations generated:  {len(combos)}')
    print(f'Valid combinations written: {valid_count}')

    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
    with open(output_file, 'w') as fh:
        SeqIO.write(records, fh, 'fasta')


def suggest_initial_mutants(
    fasta_file: str,
    num_mutants: int = 10,
    random_seed: Optional[int] = None,
) -> List[str]:
    """Randomly sample an initial panel of mutants to test in the first wet-lab round.

    Before any activity measurements exist, random sampling is a simple and
    unbiased way to select a diverse starting set.  For embedding-space-aware
    selection, use ``first_round(..., first_round_strategy="diverse_medoids")``
    instead.

    Args:
        fasta_file: Path to the mutant FASTA file (e.g. ``"mutants.fasta"``).
            Each record ID must be a variant name.  WT may or may not be included;
            it is sampled like any other record.
        num_mutants: Number of variants to suggest.  Capped at the number of
            records in the FASTA if the file is smaller.
        random_seed: Integer seed for reproducibility.

    Returns:
        List of suggested variant names (FASTA record IDs).
    """
    records = list(SeqIO.parse(fasta_file, 'fasta'))
    num_mutants = min(num_mutants, len(records))

    np.random.seed(random_seed)
    selected_indices = np.random.choice(len(records), num_mutants, replace=False)
    selected = [records[i] for i in selected_indices]

    print(f'\nSuggested {num_mutants} mutants for initial testing:')
    for i, rec in enumerate(selected, 1):
        print(f'  {i:>3}. {rec.id}')

    return [rec.id for rec in selected]
