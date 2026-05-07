"""
plot.py — Plotting and reporting utilities for EvolvePro.

**Experimental campaign results** — reading wet-lab round files and plotting
   per-variant activity across rounds.
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
# Experimental data loading
# ---------------------------------------------------------------------------

def read_exp_data(
    round_base_path: str,
    round_file_names_single: List[str],
    wt_fasta_path: str,
    round_file_names_multi: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Load and combine experimental round files into a single DataFrame.

    Each file is assigned a sequential ``iteration`` number (1, 2, …).  WT is
    assigned ``iteration = 0`` and appears only in the first file.

    Args:
        round_base_path: Directory containing the Excel round files.
        round_file_names_single: Ordered list of single-mutant Excel file names.
        wt_fasta_path: Path to the wild-type FASTA file.
        round_file_names_multi: Optional list of multi-mutant Excel file names,
            appended after the single-mutant rounds.

    Returns:
        Combined DataFrame with ``variant``, ``activity``, and ``iteration``
        columns (plus any other columns from the Excel files).
    """
    all_data = []
    for fname in round_file_names_single:
        all_data.append(
            load_experimental_data(round_base_path, fname, wt_fasta_path, single_mutant=True)
        )
    if round_file_names_multi:
        for fname in round_file_names_multi:
            all_data.append(
                load_experimental_data(round_base_path, fname, wt_fasta_path, single_mutant=False)
            )

    processed = []
    for round_num, df in enumerate(all_data, start=1):
        df_copy = df.copy()
        if round_num == 1:
            df_copy.loc[df_copy['updated_variant'] == 'WT', 'iteration'] = 0
        else:
            df_copy = df_copy[df_copy['updated_variant'] != 'WT']
        df_copy.loc[df_copy['updated_variant'] != 'WT', 'iteration'] = float(round_num)
        df_copy['iteration'] = df_copy['iteration'].astype(float)
        df_copy = df_copy.rename(columns={'updated_variant': 'variant'})
        processed.append(df_copy)

    return pd.concat(processed, ignore_index=True)


# ---------------------------------------------------------------------------
# Generic DataFrame utilities
# ---------------------------------------------------------------------------

def filter_dataframe(
    df: pd.DataFrame,
    conditions: Dict,
    output_dir: Optional[str] = None,
    output_file: Optional[str] = None,
) -> pd.DataFrame:
    """Filter a DataFrame by a dictionary of column → value conditions.

    Args:
        df: Input DataFrame.
        conditions: Mapping of column name to a scalar value (equality filter)
            or a list of values (membership filter).  Conditions are applied
            sequentially.
        output_dir: If provided (together with ``output_file``), save the result.
        output_file: CSV file name for the saved result.

    Returns:
        Filtered DataFrame.
    """
    filtered_df = df.copy()
    for column, value in conditions.items():
        if isinstance(value, list):
            filtered_df = filtered_df[filtered_df[column].isin(value)]
        else:
            filtered_df = filtered_df[filtered_df[column] == value]

    if output_dir and output_file:
        os.makedirs(output_dir, exist_ok=True)
        filtered_df.to_csv(os.path.join(output_dir, output_file), index=False)

    return filtered_df


def save_dataframe(
    df: pd.DataFrame,
    output_dir: Optional[str] = None,
    output_file: Optional[str] = None,
) -> None:
    """Save a DataFrame to a CSV file.

    Args:
        df: DataFrame to save.
        output_dir: Directory to save the file in.
        output_file: CSV file name.
    """
    if output_dir and output_file:
        os.makedirs(output_dir, exist_ok=True)
        df.to_csv(os.path.join(output_dir, output_file), index=False)


def apply_labels(
    df: pd.DataFrame,
    column: str,
    prefix: str = '',
    suffix: str = '',
    value_column: Optional[str] = None,
    format_string: str = '{}',
) -> pd.DataFrame:
    """Add a formatted string column derived from the index or another column.

    Args:
        df: Input DataFrame.
        column: Name of the new column to create.
        prefix: String prepended to each label.
        suffix: String appended to each label.
        value_column: Column whose values are formatted.  If ``None``, the row
            index is used.
        format_string: Python format string applied to each value.

    Returns:
        DataFrame with the new ``column`` added.
    """
    source = df[value_column] if value_column is not None else df.index.to_series()
    df[column] = prefix + source.map(lambda x: format_string.format(x)) + suffix
    return df


def load_external_data(
    file_path: str,
    label: Optional[str] = None,
    rename_columns: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """Load a CSV file and optionally add a label column or rename columns.

    Args:
        file_path: Path to the CSV file.
        label: If provided, a ``label`` column is added with this value in every row.
        rename_columns: Dict mapping old column names to new names.

    Returns:
        Loaded and processed DataFrame.
    """
    df = pd.read_csv(file_path)
    if label:
        df['label'] = label
    if rename_columns:
        df = df.rename(columns=rename_columns)
    return df


def concatenate_dataframes(
    dataframes: List[pd.DataFrame],
    output_dir: Optional[str] = None,
    output_file: Optional[str] = None,
) -> pd.DataFrame:
    """Concatenate a list of DataFrames and optionally save the result.

    Args:
        dataframes: DataFrames to concatenate (row-wise).
        output_dir: Directory to save the result.
        output_file: CSV file name for the saved result.

    Returns:
        Concatenated DataFrame.
    """
    result = pd.concat(dataframes, ignore_index=True)
    if output_dir and output_file:
        os.makedirs(output_dir, exist_ok=True)
        result.to_csv(os.path.join(output_dir, output_file), index=False)
    return result


# ---------------------------------------------------------------------------
# Plotting — simulation comparisons
# ---------------------------------------------------------------------------

def plot_comparison(
    concatenated_df: pd.DataFrame,
    palette=None,
    variable: str = 'activity_binary_percentage_mean',
    title: Optional[str] = None,
    output_dir: Optional[str] = None,
    output_file: Optional[str] = None,
) -> None:
    """Bar plots comparing a metric across datasets and model labels.

    Produces two plots:

    1. Grouped bar chart with ``dataset`` on the x-axis, coloured by ``label``.
    2. Bar + swarm chart with ``label`` on the x-axis (one bar per model).

    Args:
        concatenated_df: DataFrame with ``dataset``, ``label``, and ``variable``
            columns, typically produced by concatenating several
            :func:`load_external_data` results.
        palette: Seaborn colour palette.  ``None`` uses the default ``"tab10"``.
        variable: Column to plot on the y-axis.
        title: Plot title.
        output_dir: Directory to save plots.
        output_file: Base file name; plots are saved as
            ``<output_file>_by_dataset.png`` and ``<output_file>_by_model.png``.
    """
    colors = palette if palette is not None else sns.color_palette('tab10')
    y_label = variable.replace('_', ' ').title()

    # Plot 1: grouped by dataset.
    plt.figure(figsize=(10, 6))
    sns.barplot(data=concatenated_df, x='dataset', y=variable,
                hue='label', palette=colors, alpha=0.75)
    plt.xlabel('Dataset')
    plt.ylabel(y_label)
    plt.title(title)
    plt.xticks(rotation=45)
    plt.legend(title='Label', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    if output_dir and output_file:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(
            os.path.join(output_dir, f'{output_file}_by_dataset.png'),
            dpi=300, bbox_inches='tight',
        )
    plt.show()

    # Plot 2: grouped by label with individual data points.
    plt.figure(figsize=(7, 6))
    sns.barplot(data=concatenated_df, x='label', y=variable,
                palette=colors, alpha=0.75)
    sns.swarmplot(data=concatenated_df, x='label', y=variable,
                  size=4, color='black')
    plt.xlabel('Model')
    plt.ylabel(y_label)
    plt.title(title)
    plt.xticks(rotation=90)
    plt.tight_layout()
    if output_dir and output_file:
        plt.savefig(
            os.path.join(output_dir, f'{output_file}_by_model.png'),
            dpi=300, bbox_inches='tight',
        )
    plt.show()


def plot_grid_search_bar(
    df: pd.DataFrame,
    variable: str = 'activity_binary_percentage_mean',
    strategy_column: Optional[str] = None,
    title: Optional[str] = None,
    output_dir: Optional[str] = None,
    output_file: Optional[str] = None,
) -> None:
    """Grouped bar chart comparing strategies across datasets.

    For each dataset, bars are grouped by the unique values of ``strategy_column``.
    Also prints a summary of how often each strategy is the best-performing.

    Args:
        df: Aggregated simulation results DataFrame with ``dataset``,
            ``round_num``, ``strategy_column``, and ``variable`` columns.
        variable: Metric to plot on the y-axis.
        strategy_column: Column whose values define the bar groups (e.g.
            ``"regression_type"`` or ``"learning_strategy"``).
        title: Plot title.  Auto-generated if ``None``.
        output_dir: Directory to save the plot.
        output_file: Base file name (saved as ``<output_file>_grid_bar.png``).
    """
    if strategy_column is None:
        raise ValueError('strategy_column must be specified.')

    round_num = df['round_num'].iloc[0]
    grouped = df.groupby(['dataset', strategy_column])[variable].mean().unstack()

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=df, x='dataset', y=variable, hue=strategy_column, alpha=0.75)

    if title is None:
        title = (
            f'{variable.replace("_", " ").title()} by '
            f'{strategy_column.replace("_", " ").title()} ({round_num} rounds)'
        )
    ax.set_title(title)
    ax.set_xlabel('Dataset')
    ax.set_ylabel(variable.replace('_', ' ').title())
    ax.legend(
        title=strategy_column.replace('_', ' ').title(),
        bbox_to_anchor=(1.05, 1), loc='upper left',
    )
    plt.xticks(rotation=45)
    plt.tight_layout()

    if output_dir and output_file:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(
            os.path.join(output_dir, f'{output_file}_grid_bar.png'),
            dpi=300, bbox_inches='tight',
        )
    plt.show()

    winning_counts = grouped.apply(lambda x: x.idxmax(), axis=1).value_counts()
    print('\nBest strategy per dataset:')
    print(winning_counts)


def plot_grid_search_heatmap(
    df: pd.DataFrame,
    variable: str = 'activity_binary_percentage_mean',
    strategy_columns: Optional[List[str]] = None,
    title: Optional[str] = None,
    output_dir: Optional[str] = None,
    output_file: Optional[str] = None,
) -> None:
    """Heatmap comparing the interaction of two hyper-parameter strategies.

    Args:
        df: Aggregated simulation results DataFrame.
        variable: Metric to display in heatmap cells.
        strategy_columns: Exactly two column names defining the heatmap axes.
        title: Plot title.  Auto-generated if ``None``.
        output_dir: Directory to save the plot.
        output_file: Base file name (saved as ``<output_file>_grid_heatmap.png``).
    """
    if not strategy_columns or len(strategy_columns) != 2:
        raise ValueError('strategy_columns must be a list of exactly two column names.')

    round_num = df['round_num'].iloc[0]
    grouped = df.groupby(strategy_columns)[variable].mean().unstack()

    plt.figure(figsize=(12, 8))
    ax = sns.heatmap(grouped, cmap='viridis', annot=True, fmt='.2f', linewidths=0.5)

    if title is None:
        s0 = strategy_columns[0].replace('_', ' ').title()
        s1 = strategy_columns[1].replace('_', ' ').title()
        title = (
            f'Average {variable.replace("_", " ").title()} '
            f'by {s0} and {s1} ({round_num} rounds)'
        )

    ax.set_title(title)
    ax.set_xlabel(strategy_columns[1].replace('_', ' ').title())
    ax.set_ylabel(strategy_columns[0].replace('_', ' ').title())
    plt.tight_layout()

    if output_dir and output_file:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(
            os.path.join(output_dir, f'{output_file}_grid_heatmap.png'),
            dpi=300, bbox_inches='tight',
        )
    plt.show()


def plot_by_round(
    df: pd.DataFrame,
    variable: str = 'activity_binary_percentage_mean',
    output_dir: Optional[str] = None,
    output_file: Optional[str] = None,
) -> None:
    """Line chart of a metric over rounds, one line per dataset.

    Args:
        df: Aggregated results DataFrame with ``dataset``, ``round_num``,
            and ``variable`` columns.
        variable: Metric to plot on the y-axis.
        output_dir: Directory to save the plot.
        output_file: Base file name (saved as ``<output_file>_by_round.png``).
    """
    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    datasets = df['dataset'].unique()
    palette = dict(zip(datasets, sns.color_palette('tab10', n_colors=len(datasets))))

    for dataset in datasets:
        sub = df[df['dataset'] == dataset]
        sns.lineplot(
            x=sub['round_num'], y=sub[variable],
            ax=ax, marker='o', color=palette[dataset], label=dataset,
        )

    ax.set_xlabel('Round')
    ax.set_ylabel(variable.replace('_', ' ').title())
    ax.set_title(f'{variable.replace("_", " ").title()} by Round')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    if output_dir and output_file:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(
            os.path.join(output_dir, f'{output_file}_by_round.png'),
            dpi=300, bbox_inches='tight',
        )
    plt.show()


def plot_by_round_split(
    df: pd.DataFrame,
    variable: str = 'activity_binary_percentage_mean',
    split_variable: str = 'num_mutants_per_round',
    output_dir: Optional[str] = None,
    output_file: Optional[str] = None,
) -> None:
    """Grid of line charts — one subplot per dataset, lines split by a parameter.

    Useful for comparing the effect of a single hyper-parameter (e.g. panel
    size) across all datasets simultaneously.

    Args:
        df: Aggregated results DataFrame.
        variable: Metric to plot on the y-axis.
        split_variable: Column whose unique values define separate lines within
            each subplot.
        output_dir: Directory to save the plot.
        output_file: Base file name (saved as
            ``<output_file>_by_round_split_<split_variable>.png``).
    """
    datasets = df['dataset'].unique()
    split_values = sorted(df[split_variable].unique())
    palette = dict(zip(split_values, sns.color_palette('tab10', n_colors=len(split_values))))

    n_cols = 3
    n_rows = (len(datasets) + n_cols - 1) // n_cols
    fig = plt.figure(figsize=(20, 4 * n_rows))
    gs = fig.add_gridspec(n_rows, n_cols)

    for idx, dataset in enumerate(datasets):
        ax = fig.add_subplot(gs[idx // n_cols, idx % n_cols])
        sub = df[df['dataset'] == dataset]
        for val in split_values:
            sns.lineplot(
                data=sub[sub[split_variable] == val],
                x='round_num', y=variable,
                marker='o', ax=ax, color=palette[val],
                label=f'{split_variable}: {val}',
            )
        ax.set_xlabel('Round')
        ax.set_ylabel(variable.replace('_', ' ').title())
        ax.set_title(dataset)
        ax.get_legend().remove()

    # Remove unused subplot axes.
    for idx in range(len(datasets), n_rows * n_cols):
        fig.delaxes(fig.add_subplot(gs[idx // n_cols, idx % n_cols]))

    # Shared legend.
    legend_handles = [
        plt.Line2D([0], [0], color=palette[v], marker='o', linestyle='-')
        for v in split_values
    ]
    fig.legend(
        legend_handles,
        [f'{split_variable}: {v}' for v in split_values],
        loc='center left', bbox_to_anchor=(1.0, 0.5),
    )
    fig.suptitle(f'{variable.replace("_", " ").title()} by Round', y=1.02)
    plt.tight_layout(rect=[0, 0, 0.98, 1])

    if output_dir and output_file:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(
            os.path.join(output_dir, f'{output_file}_by_round_split_{split_variable}.png'),
            dpi=300, bbox_inches='tight',
        )
    plt.show()


# ---------------------------------------------------------------------------
# Plotting — experimental results
# ---------------------------------------------------------------------------

def plot_variants_by_iteration(
    df: pd.DataFrame,
    activity_column: str = 'activity',
    output_dir: Optional[str] = None,
    output_file: Optional[str] = None,
) -> None:
    """Bar chart showing per-variant activity grouped and coloured by assay round.

    Variants are sorted within each round by ascending activity so the chart
    reads from least to most active left-to-right within each group.

    Args:
        df: DataFrame with ``variant``, ``iteration``, and ``activity_column``
            columns, typically produced by :func:`read_exp_data`.
        activity_column: Column containing numeric activity values.
        output_dir: Directory to save the plot.
        output_file: Base file name (saved as ``<output_file>_by_iteration.png``).
    """
    df = df.copy()
    df['iteration'] = df['iteration'].astype(int)
    df = df.sort_values(['iteration', activity_column]).reset_index(drop=True)

    plt.figure(figsize=(12, 6))
    for iteration, group in df.groupby('iteration'):
        plt.bar(group.index, group[activity_column], label=f'Round {iteration}')

    plt.xticks(df.index, df['variant'], rotation=90)
    plt.ylabel(activity_column.capitalize())
    plt.legend()
    plt.tight_layout()

    if output_dir and output_file:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(
            os.path.join(output_dir, f'{output_file}_by_iteration.png'),
            dpi=300, bbox_inches='tight',
        )
    plt.show()
