"""
evolve.py — Top-level orchestration for EvolvePro.

Three public entry points:

- ``directed_evolution_simulation`` — run a full simulated directed-evolution
  campaign on a DMS dataset, tracking per-round metrics across many simulations.

- ``grid_search`` — sweep over combinations of hyper-parameters (regression
  model, embedding type, learning strategy, …) and save a results CSV.

- ``evolve_experimental`` — process one or more completed assay rounds and
  output ranked predictions for the *next* round (the primary function used in
  day-to-day experimental campaigns).

- ``evolve_experimental_multi`` — like ``evolve_experimental`` but supports
  embedding files that mix single-mutant and multi-mutant variants across rounds.
"""

import os
import random
import time
import warnings
from typing import List, Optional, Tuple

import pandas as pd

from evolvepro.src.data import (
    create_iteration_dataframes,
    load_dms_data,
    load_experimental_data,
    load_experimental_embeddings,
)
from evolvepro.src.model import first_round, top_layer, _VALID_REGRESSION_TYPES
from evolvepro.src.utils import pca_embeddings

_VALID_LEARNING_STRATEGIES = frozenset({'topn', 'topn2bottomn2', 'dist', 'random'})
_VALID_FIRST_ROUND_STRATEGIES = frozenset({'random', 'diverse_medoids', 'explicit_variants'})


# ---------------------------------------------------------------------------
# Simulation (DMS benchmarking)
# ---------------------------------------------------------------------------

def directed_evolution_simulation(
    labels: pd.DataFrame,
    embeddings: pd.DataFrame,
    num_simulations: int,
    num_iterations: int,
    num_mutants_per_round: int = 10,
    measured_var: str = 'activity',
    regression_type: str = 'ridge',
    learning_strategy: str = 'topn',
    top_n: Optional[int] = None,
    final_round: int = 10,
    first_round_strategy: str = 'random',
    embedding_type: Optional[str] = None,
    explicit_variants: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Simulate a directed-evolution campaign on a fully-labelled DMS dataset.

    Each simulation independently selects a first-round panel, then iteratively
    trains a regression model and selects new variants for the next round.
    Because all activity values are known upfront, the simulation can measure
    the quality of each round's predictions against held-out ground truth.

    Args:
        labels: Full DMS labels DataFrame.  Must contain ``variant``,
            ``activity``, ``activity_scaled``, and ``activity_binary`` columns
            (produced by ``process_dataset`` / ``load_dms_data``).
        embeddings: DataFrame of variant embeddings indexed by variant name.
            Row order must match ``labels``.
        num_simulations: Number of independent simulation runs.  Each run uses
            a different random seed so results can be averaged.
        num_iterations: Number of active-learning rounds per simulation
            (not counting round 0, which is the initial panel selection).
        num_mutants_per_round: Number of variants to select at each round.
        measured_var: Column in ``labels`` used as the regression target,
            typically ``"activity"``.
        regression_type: Regression model passed to ``top_layer``.
            One of ``"ridge"``, ``"lasso"``, ``"elasticnet"``, ``"linear"``,
            ``"neuralnet"``, ``"randomforest"``, ``"gradientboosting"``,
            ``"knn"``, ``"gp"``.
        learning_strategy: How to select the next round's variants from
            model predictions:

            - ``"topn"`` — choose the top ``num_mutants_per_round`` by predicted
              activity (greedy exploitation).
            - ``"topn2bottomn2"`` — split equally between the top-predicted and
              bottom-predicted variants (exploration / exploitation balance).
            - ``"dist"`` — choose variants furthest in embedding space from the
              training set (pure exploration).
            - ``"random"`` — choose randomly from untested variants.

        top_n: Reserved for future use.
        final_round: How many top predictions to include when computing summary
            statistics (``median_activity_scaled``, ``activity_binary_percentage``).
        first_round_strategy: Initial panel selection strategy.
            One of ``"random"``, ``"diverse_medoids"``, ``"explicit_variants"``.
        embedding_type: Passed through to ``first_round``; set to
            ``"embeddings_pca"`` if the embeddings are already PCA-reduced.
        explicit_variants: Used only when ``first_round_strategy="explicit_variants"``.

    Returns:
        DataFrame with one row per (simulation, round) pair, containing model
        metrics and the selected variants for each round.  Columns include:

        ``simulation_num``, ``round_num``, ``num_mutants_per_round``,
        ``first_round_strategy``, ``measured_var``, ``learning_strategy``,
        ``regression_type``, ``embedding_type``, ``test_error``, ``train_error``,
        ``train_r_squared``, ``test_r_squared``, ``alpha``, ``spearman_corr``,
        ``median_activity_scaled``, ``top_activity_scaled``,
        ``activity_binary_percentage``, ``top_variant``,
        ``top_final_round_variants``, ``this_round_variants``,
        ``next_round_variants``.
    """
    # --- Validate parameters up front ---
    if regression_type not in _VALID_REGRESSION_TYPES:
        raise ValueError(
            f"Unknown regression_type '{regression_type}'. "
            f"Must be one of: {sorted(_VALID_REGRESSION_TYPES)}."
        )
    if learning_strategy not in _VALID_LEARNING_STRATEGIES:
        raise ValueError(
            f"Unknown learning_strategy '{learning_strategy}'. "
            f"Must be one of: {sorted(_VALID_LEARNING_STRATEGIES)}."
        )
    if first_round_strategy not in _VALID_FIRST_ROUND_STRATEGIES:
        raise ValueError(
            f"Unknown first_round_strategy '{first_round_strategy}'. "
            f"Must be one of: {sorted(_VALID_FIRST_ROUND_STRATEGIES)}."
        )
    if measured_var not in labels.columns:
        raise ValueError(
            f"measured_var '{measured_var}' is not a column in labels. "
            f"Available columns: {list(labels.columns)}."
        )
    if num_simulations < 1:
        raise ValueError(
            f"num_simulations must be at least 1, got {num_simulations}."
        )
    if num_iterations < 1:
        raise ValueError(
            f"num_iterations must be at least 1, got {num_iterations}."
        )
    if num_mutants_per_round < 1:
        raise ValueError(
            f"num_mutants_per_round must be at least 1, got {num_mutants_per_round}."
        )

    output_list = []

    for sim_idx in range(1, num_simulations + 1):
        records = {
            'simulation_num': [], 'round_num': [], 'num_mutants_per_round': [],
            'first_round_strategy': [], 'measured_var': [], 'learning_strategy': [],
            'regression_type': [], 'embedding_type': [], 'test_error': [],
            'train_error': [], 'train_r_squared': [], 'test_r_squared': [],
            'alpha': [], 'spearman_corr': [], 'median_activity_scaled': [],
            'top_activity_scaled': [], 'activity_binary_percentage': [],
            'top_variant': [], 'top_final_round_variants': [],
            'this_round_variants': [], 'next_round_variants': [],
        }

        iteration_new = None
        labels_new = None

        for round_num in range(0, num_iterations + 1):

            # ---- Round 0: initial panel selection ----
            if round_num == 0:
                labels_new, iteration_new, this_round_variants = first_round(
                    labels,
                    embeddings,
                    explicit_variants=explicit_variants,
                    num_mutants_per_round=num_mutants_per_round,
                    first_round_strategy=first_round_strategy,
                    embedding_type=embedding_type,
                    random_seed=sim_idx,
                )

                _append_record(
                    records, sim_idx, round_num, num_mutants_per_round,
                    first_round_strategy, measured_var, learning_strategy,
                    regression_type, embedding_type,
                    test_error='None', train_error='None',
                    train_r2='None', test_r2='None', alpha='None',
                    spearman='None', med_scaled='None', top_scaled='None',
                    bin_pct='None', top_var='None', top_vars='None',
                    this_vars='None',
                    next_vars=','.join(this_round_variants),
                )

            # ---- Rounds 1‥N: train model, select next panel ----
            else:
                print(f'Sim {sim_idx} | Round {round_num} | '
                      f'Training on iterations: {iteration_new["iteration"].unique().tolist()}')

                (
                    train_error, test_error, train_r2, test_r2, alpha,
                    median_scaled, top_scaled, top_var, top_vars,
                    bin_pct, spearman, df_test_new, this_round_variants,
                ) = top_layer(
                    iter_train=iteration_new['iteration'].unique().tolist(),
                    iter_test=None,
                    embeddings_pd=embeddings,
                    labels_pd=labels_new,
                    measured_var=measured_var,
                    regression_type=regression_type,
                    top_n=top_n,
                    final_round=final_round,
                )

                iteration_new_ids = _select_next_variants(
                    df_test_new, learning_strategy, num_mutants_per_round
                )

                iteration_new = pd.concat(
                    [
                        pd.DataFrame({'variant': iteration_new_ids, 'iteration': round_num}),
                        iteration_new,
                    ],
                    ignore_index=True,
                )
                labels_new = pd.merge(labels, iteration_new, on='variant', how='left')

                _append_record(
                    records, sim_idx, round_num, num_mutants_per_round,
                    first_round_strategy, measured_var, learning_strategy,
                    regression_type, embedding_type,
                    test_error=test_error, train_error=train_error,
                    train_r2=train_r2, test_r2=test_r2, alpha=alpha,
                    spearman=spearman, med_scaled=median_scaled,
                    top_scaled=top_scaled, bin_pct=bin_pct,
                    top_var=top_var, top_vars=top_vars,
                    this_vars=','.join(iteration_new['variant']),
                    next_vars=','.join(iteration_new_ids),
                )

        output_list.append(pd.DataFrame(records))

    return pd.concat(output_list, ignore_index=True)


def _select_next_variants(
    df_test: pd.DataFrame,
    learning_strategy: str,
    num_mutants_per_round: int,
) -> pd.Series:
    """Return variant names to test in the next round.

    Args:
        df_test: Predictions DataFrame returned by ``top_layer``.
        learning_strategy: Selection strategy (see ``directed_evolution_simulation``).
        num_mutants_per_round: How many variants to select.

    Returns:
        Series of selected variant names.
    """
    available = len(df_test)
    if available == 0:
        raise ValueError(
            "df_test is empty — there are no untested variants to select from. "
            "All candidates have been measured."
        )
    if num_mutants_per_round > available:
        warnings.warn(
            f"num_mutants_per_round ({num_mutants_per_round}) exceeds the number "
            f"of remaining untested variants ({available}). "
            f"Selecting all {available} available variant(s) instead.",
            UserWarning,
            stacklevel=2,
        )
        num_mutants_per_round = available

    if learning_strategy == 'topn':
        return df_test.sort_values('y_pred', ascending=False).head(num_mutants_per_round)['variant']
    if learning_strategy == 'topn2bottomn2':
        half = num_mutants_per_round // 2
        top = df_test.sort_values('y_pred', ascending=False).head(half)['variant']
        bottom = df_test.sort_values('y_pred', ascending=False).tail(half)['variant']
        return pd.concat([top, bottom])
    if learning_strategy == 'dist':
        if df_test['dist_metric'].isna().all():
            raise ValueError(
                "learning_strategy='dist' requires 'dist_metric' values in df_test, "
                "but all values are NaN. This strategy is only supported in "
                "simulation mode (experimental=False)."
            )
        return df_test.sort_values('dist_metric', ascending=False).head(num_mutants_per_round)['variant']
    if learning_strategy == 'random':
        return pd.Series(random.sample(list(df_test['variant']), num_mutants_per_round))

    # Should never be reached — validated in directed_evolution_simulation.
    raise ValueError(
        f"Unknown learning_strategy '{learning_strategy}'. "
        f"Must be one of: {sorted(_VALID_LEARNING_STRATEGIES)}."
    )


def _append_record(records, sim, rnd, n_mut, fr_strat, mvar, lstrat,
                   reg, emb, test_error, train_error, train_r2, test_r2,
                   alpha, spearman, med_scaled, top_scaled, bin_pct,
                   top_var, top_vars, this_vars, next_vars):
    """Append one round's metrics to the per-simulation tracking lists."""
    records['simulation_num'].append(sim)
    records['round_num'].append(rnd)
    records['num_mutants_per_round'].append(n_mut)
    records['first_round_strategy'].append(fr_strat)
    records['measured_var'].append(mvar)
    records['learning_strategy'].append(lstrat)
    records['regression_type'].append(reg)
    records['embedding_type'].append(emb)
    records['test_error'].append(test_error)
    records['train_error'].append(train_error)
    records['train_r_squared'].append(train_r2)
    records['test_r_squared'].append(test_r2)
    records['alpha'].append(alpha)
    records['spearman_corr'].append(spearman)
    records['median_activity_scaled'].append(med_scaled)
    records['top_activity_scaled'].append(top_scaled)
    records['activity_binary_percentage'].append(bin_pct)
    records['top_variant'].append(top_var)
    records['top_final_round_variants'].append(top_vars)
    records['this_round_variants'].append(this_vars)
    records['next_round_variants'].append(next_vars)


# ---------------------------------------------------------------------------
# Grid search over hyper-parameters (DMS benchmarking)
# ---------------------------------------------------------------------------

def grid_search(
    dataset_name: str,
    experiment_name: str,
    model_name: str,
    embeddings_path: str,
    labels_path: str,
    num_simulations: int,
    num_iterations: List[int],
    measured_var: List[str],
    learning_strategies: List[str],
    num_mutants_per_round: List[int],
    num_final_round_mutants: int,
    first_round_strategies: List[str],
    embedding_types: List[str],
    pca_components: Optional[List[int]],
    regression_types: List[str],
    embeddings_file_type: str,
    output_dir: str,
    embeddings_type_pt: Optional[str] = None,
) -> None:
    """Run ``directed_evolution_simulation`` over a grid of hyper-parameters.

    All results are concatenated and saved to a single CSV file in ``output_dir``.
    This is useful for benchmarking which combination of embedding, regression
    model, and learning strategy performs best on a given DMS dataset.

    The output file is named:
        ``<dataset_name>_<model_name>_<experiment_name>[_<embeddings_type_pt>].csv``

    Args:
        dataset_name: Short identifier for the protein (used in file names).
        experiment_name: Short label for this grid search run.
        model_name: Name of the embedding model (part of the embeddings file name).
        embeddings_path: Directory containing the embeddings file.
        labels_path: Directory containing the labels CSV.
        num_simulations: Number of independent simulation repeats per combination.
        num_iterations: List of iteration counts to try, e.g. ``[3, 5, 10]``.
        measured_var: List of target columns, e.g. ``["activity"]``.
        learning_strategies: List of selection strategies to try.
        num_mutants_per_round: List of panel sizes to try, e.g. ``[8, 16]``.
        num_final_round_mutants: Top-N cutoff for summary-metric computation.
        first_round_strategies: List of initial-round strategies to try.
        embedding_types: List of embedding representations to try.  Each entry
            must be either ``"embeddings"`` or ``"embeddings_pca_<N>"``
            (matching the ``pca_components`` argument).
        pca_components: List of PCA dimensionalities to pre-compute, e.g.
            ``[10, 50]``.  Pass ``None`` to skip PCA variants.
        regression_types: List of regression models to try.
        embeddings_file_type: ``"csv"`` or ``"pt"``.
        output_dir: Directory where the results CSV will be saved.
        embeddings_type_pt: Tensor type for ``.pt`` embeddings files
            (``"average"``, ``"mutated"``, or ``"both"``).  ``None`` for CSV files.
    """
    # --- Validate list parameters before starting the potentially long search ---
    invalid_strategies = [s for s in learning_strategies if s not in _VALID_LEARNING_STRATEGIES]
    if invalid_strategies:
        raise ValueError(
            f"Invalid learning_strategy value(s): {invalid_strategies}. "
            f"Must be one of: {sorted(_VALID_LEARNING_STRATEGIES)}."
        )

    invalid_fr_strategies = [s for s in first_round_strategies if s not in _VALID_FIRST_ROUND_STRATEGIES]
    if invalid_fr_strategies:
        raise ValueError(
            f"Invalid first_round_strategy value(s): {invalid_fr_strategies}. "
            f"Must be one of: {sorted(_VALID_FIRST_ROUND_STRATEGIES)}."
        )

    invalid_reg_types = [r for r in regression_types if r not in _VALID_REGRESSION_TYPES]
    if invalid_reg_types:
        raise ValueError(
            f"Invalid regression_type value(s): {invalid_reg_types}. "
            f"Must be one of: {sorted(_VALID_REGRESSION_TYPES)}."
        )

    if not os.path.isdir(embeddings_path):
        raise FileNotFoundError(
            f"embeddings_path directory not found: '{embeddings_path}'."
        )
    if not os.path.isdir(labels_path):
        raise FileNotFoundError(
            f"labels_path directory not found: '{labels_path}'."
        )

    embeddings, labels = load_dms_data(
        dataset_name, model_name, embeddings_path, labels_path,
        embeddings_file_type, embeddings_type_pt,
    )
    if embeddings is None or labels is None:
        raise RuntimeError(
            'Failed to load DMS data. Check the error messages above for details.'
        )

    # --- Pre-compute PCA-reduced embedding variants ---
    embeddings_library = {'embeddings': embeddings}
    if pca_components is not None:
        for n in pca_components:
            key = f'embeddings_pca_{n}'
            if key not in embedding_types:
                warnings.warn(
                    f"PCA variant '{key}' was computed (pca_components includes {n}) "
                    f"but '{key}' does not appear in embedding_types and will never "
                    f"be used. Add '{key}' to embedding_types or remove {n} from "
                    f"pca_components.",
                    UserWarning,
                    stacklevel=2,
                )
            embeddings_library[key] = pca_embeddings(embeddings, n_components=n)

    # --- Warn about embedding_types that have no precomputed entry ---
    missing_emb_types = [e for e in embedding_types if e not in embeddings_library]
    if missing_emb_types:
        raise ValueError(
            f"embedding_types contains entry/entries with no precomputed embeddings: "
            f"{missing_emb_types}. "
            f"Entries other than 'embeddings' must follow the pattern "
            f"'embeddings_pca_<N>' and the corresponding N must appear in "
            f"pca_components."
        )

    total = (
        len(learning_strategies) * len(measured_var) * len(num_iterations)
        * len(num_mutants_per_round) * len(embedding_types) * len(regression_types)
        * len(first_round_strategies)
    )
    print(f'Total hyper-parameter combinations: {total}')

    output_list = []
    combo_count = 0
    start_time = time.time()

    for strategy in learning_strategies:
        for var in measured_var:
            for iterations in num_iterations:
                for mutants_per_round in num_mutants_per_round:
                    for emb_type in embedding_types:
                        for reg_type in regression_types:
                            for fr_strategy in first_round_strategies:
                                combo_count += 1
                                result = directed_evolution_simulation(
                                    labels=labels,
                                    embeddings=embeddings_library[emb_type],
                                    num_simulations=num_simulations,
                                    num_iterations=iterations,
                                    num_mutants_per_round=mutants_per_round,
                                    measured_var=var,
                                    regression_type=reg_type,
                                    learning_strategy=strategy,
                                    final_round=num_final_round_mutants,
                                    first_round_strategy=fr_strategy,
                                    embedding_type=emb_type,
                                )
                                output_list.append(result)
                                print(
                                    f'  Progress: {combo_count}/{total} '
                                    f'({100 * combo_count / total:.1f}%)'
                                )

    elapsed = time.time() - start_time
    print(f'Total execution time: {elapsed:.1f} s')

    df_results = pd.concat(output_list, ignore_index=True)
    os.makedirs(output_dir, exist_ok=True)

    suffix = f'_{embeddings_type_pt}' if embeddings_type_pt else ''
    out_path = os.path.join(
        output_dir,
        f'{dataset_name}_{model_name}_{experiment_name}{suffix}.csv',
    )
    df_results.to_csv(out_path, index=False)
    print(f'Results saved to {out_path}')


# ---------------------------------------------------------------------------
# Experimental directed evolution (single-mutant)
# ---------------------------------------------------------------------------

def evolve_experimental(
    protein_name: str,
    round_name: str,
    embeddings_base_path: str,
    embeddings_file_name: str,
    round_base_path: str,
    round_file_names: List[str],
    wt_fasta_path: str,
    rename_WT: bool = False,
    number_of_variants: int = 12,
    output_dir: Optional[str] = None,
) -> None:
    """Process completed assay rounds and rank untested variants for the next round.

    This is the primary function for real experimental campaigns.  It loads
    embeddings and all assay data collected so far, trains a Random Forest
    model on the measured variants, and outputs a ranked list of predictions
    for every variant that has not yet been tested.

    Call this once after each wet-lab round.  Pass **all** round files collected
    so far (not just the most recent one) so the model has access to the full
    measurement history.

    **Required file formats** — see README.md for full specifications:

    - ``embeddings_file_name``: CSV with variant names as row index.
    - ``round_file_names``: Excel (``.xlsx``) files, each with ``Variant`` and
      ``activity`` columns.  Variants use position-only notation (e.g. ``"107M"``);
      the WT amino acid is inferred automatically from the FASTA.

    Args:
        protein_name: Human-readable protein identifier, used only for log output.
        round_name: Short label for the current round (e.g. ``"round_2"``).
            Determines the name of the output subdirectory.
        embeddings_base_path: Directory containing the embeddings CSV.
        embeddings_file_name: File name of the embeddings CSV, e.g.
            ``"mutants_esm2_t33_650M_UR50D.csv"``.
        round_base_path: Directory containing the assay Excel files.
        round_file_names: Ordered list of Excel file names, one per assay round,
            in chronological order (oldest first).  All files in the list must
            contain single-mutant variants.
        wt_fasta_path: Path to the wild-type FASTA file.
        rename_WT: If ``True``, rename the embedding row
            ``"WT Wild-type sequence"`` → ``"WT"``.
        number_of_variants: How many top candidates to print to the console.
            Does not affect saved output (all predictions are saved).
        output_dir: If provided, results are saved under
            ``<output_dir>/<round_name>/``.  Pass ``None`` to skip saving.

    Output files (when ``output_dir`` is provided):

    - ``iteration.csv`` — which variants were tested in which round.
    - ``this_round_variants.csv`` — variants used to train the model.
    - ``df_test.csv`` — all untested variants ranked by predicted activity
      (**use this to plan the next round**).
    - ``df_sorted_all.csv`` — all variants (train + test) ranked by prediction.
    """
    if not round_file_names:
        raise ValueError(
            "round_file_names is empty. At least one round Excel file must be "
            "provided."
        )

    print(f'Processing {protein_name} — {round_name}')

    embeddings = load_experimental_embeddings(
        embeddings_base_path, embeddings_file_name, rename_WT
    )

    all_experimental_data = []
    for fname in round_file_names:
        exp_data = load_experimental_data(round_base_path, fname, wt_fasta_path)
        all_experimental_data.append(exp_data)
        print(f'  Loaded {fname}: {exp_data.shape}')

    iteration, labels = create_iteration_dataframes(
        all_experimental_data, embeddings.index.tolist()
    )

    this_round_variants, df_test, df_sorted_all = top_layer(
        iter_train=iteration['iteration'].unique().tolist(),
        iter_test=None,
        embeddings_pd=embeddings,
        labels_pd=labels,
        measured_var='activity',
        regression_type='randomforest',
        experimental=True,
    )

    print(f'\nVariants tested so far: {len(this_round_variants)}')
    print(
        f'\nTop {number_of_variants} predicted variants:\n'
        + df_test.sort_values('y_pred', ascending=False)
        .head(number_of_variants)[['variant', 'y_pred']]
        .to_string(index=False)
    )

    if output_dir is not None:
        save_dir = os.path.join(output_dir, round_name)
        os.makedirs(save_dir, exist_ok=True)
        iteration.to_csv(os.path.join(save_dir, 'iteration.csv'))
        this_round_variants.to_csv(os.path.join(save_dir, 'this_round_variants.csv'))
        df_test.sort_values('y_pred', ascending=False).to_csv(
            os.path.join(save_dir, 'df_test.csv')
        )
        df_sorted_all.to_csv(os.path.join(save_dir, 'df_sorted_all.csv'))
        print(f'Results saved to {save_dir}')


# ---------------------------------------------------------------------------
# Experimental directed evolution (multi-mutant)
# ---------------------------------------------------------------------------

def evolve_experimental_multi(
    protein_name: str,
    round_name: str,
    embeddings_base_path: str,
    embeddings_file_names: List[str],
    round_base_path: str,
    round_file_names_single: List[str],
    round_file_names_multi: List[str],
    wt_fasta_path: str,
    rename_WT: bool = False,
    number_of_variants: int = 12,
    output_dir: Optional[str] = None,
) -> Tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """Like ``evolve_experimental`` but supports multi-mutant variants.

    Use this function when later rounds include combinatorial (multi-site) mutants
    in addition to single-mutant variants.  It accepts separate embedding files for
    the single-mutant and multi-mutant libraries, concatenating them internally.

    **Multi-mutant variant name format**: variants are already fully specified in
    the Excel file (e.g. ``"A107M_F73C"``); the code applies no conversion.

    Args:
        protein_name: Human-readable protein identifier (log output only).
        round_name: Label for the current round; determines output subdirectory.
        embeddings_base_path: Directory containing all embeddings CSV files.
        embeddings_file_names: Ordered list of embeddings CSV file names.
            The first file is expected to contain WT; WT rows are stripped from
            all subsequent files before concatenation to avoid duplicates.
        round_base_path: Directory containing all assay Excel files.
        round_file_names_single: Excel files containing single-mutant results,
            in chronological order.
        round_file_names_multi: Excel files containing multi-mutant results,
            in chronological order (appended after single-mutant rounds).
        wt_fasta_path: Path to the wild-type FASTA file.
        rename_WT: If ``True``, rename ``"WT Wild-type sequence"`` → ``"WT"``
            in the first embeddings file.
        number_of_variants: How many top candidates to print to the console.
        output_dir: Directory under which results are saved.  Pass ``None`` to
            skip saving.

    Returns:
        ``(this_round_variants, df_test, df_sorted_all)`` — same structure as
        ``evolve_experimental`` (see that function's docstring).
    """
    # --- Validate inputs ---
    if not embeddings_file_names:
        raise ValueError(
            "embeddings_file_names is empty. At least one embeddings CSV file "
            "must be provided."
        )
    if not round_file_names_single and not round_file_names_multi:
        raise ValueError(
            "Both round_file_names_single and round_file_names_multi are empty. "
            "At least one round file must be provided."
        )

    print(f'Processing {protein_name} — {round_name}')

    # --- Load and concatenate embedding files, keeping only one WT row ---
    embeddings_parts = []
    for i, fname in enumerate(embeddings_file_names):
        emb = load_experimental_embeddings(embeddings_base_path, fname,
                                           rename_WT=(rename_WT and i == 0))
        if i > 0:
            wt_rows_to_drop = [v for v in ('WT', 'WT Wild-type sequence') if v in emb.index]
            if wt_rows_to_drop:
                warnings.warn(
                    f"Embeddings file '{fname}' (file {i + 1} of "
                    f"{len(embeddings_file_names)}) contains WT row(s) "
                    f"{wt_rows_to_drop}. WT is retained from the first file only; "
                    f"these rows will be dropped. To avoid this, omit WT from all "
                    f"files except the first (see README, multi-mutant embeddings).",
                    UserWarning,
                    stacklevel=2,
                )
            emb = emb[~emb.index.isin(['WT', 'WT Wild-type sequence'])]
        embeddings_parts.append(emb)

    # --- Warn if embedding files have inconsistent dimensionality ---
    dims = [part.shape[1] for part in embeddings_parts]
    if len(set(dims)) > 1:
        dim_info = {embeddings_file_names[i]: dims[i] for i in range(len(dims))}
        raise ValueError(
            f"Embeddings files have inconsistent numbers of dimensions: {dim_info}. "
            f"All files must be produced by the same PLM and extraction layer "
            f"(see README, multi-mutant embeddings, Step 3)."
        )

    embeddings = pd.concat(embeddings_parts)

    # --- Warn about duplicate index entries after concatenation ---
    duplicated_indices = embeddings.index[embeddings.index.duplicated()].tolist()
    if duplicated_indices:
        raise ValueError(
            f"Duplicate variant names found in the concatenated embeddings index: "
            f"{duplicated_indices[:10]}"
            + (' (and more)' if len(duplicated_indices) > 10 else '')
            + ". Ensure each variant appears in exactly one embeddings file."
        )

    print(f'Embeddings loaded: {embeddings.shape}')

    # --- Load single-mutant rounds ---
    all_experimental_data = []
    for fname in round_file_names_single:
        exp_data = load_experimental_data(
            round_base_path, fname, wt_fasta_path, single_mutant=True
        )
        all_experimental_data.append(exp_data)
        print(f'  Loaded single-mutant file {fname}: {exp_data.shape}')

    # --- Load multi-mutant rounds ---
    for fname in round_file_names_multi:
        exp_data = load_experimental_data(
            round_base_path, fname, wt_fasta_path, single_mutant=False
        )
        all_experimental_data.append(exp_data)
        print(f'  Loaded multi-mutant file {fname}: {exp_data.shape}')

    iteration, labels = create_iteration_dataframes(
        all_experimental_data, embeddings.index.tolist()
    )

    this_round_variants, df_test, df_sorted_all = top_layer(
        iter_train=iteration['iteration'].unique().tolist(),
        iter_test=None,
        embeddings_pd=embeddings,
        labels_pd=labels,
        measured_var='activity',
        regression_type='randomforest',
        experimental=True,
    )

    print(f'\nVariants tested so far: {len(this_round_variants)}')
    print(
        f'\nTop {number_of_variants} predicted variants:\n'
        + df_test.sort_values('y_pred', ascending=False)
        .head(number_of_variants)[['variant', 'y_pred']]
        .to_string(index=False)
    )

    if output_dir is not None:
        save_dir = os.path.join(output_dir, protein_name, round_name)
        os.makedirs(save_dir, exist_ok=True)
        iteration.to_csv(os.path.join(save_dir, 'iteration.csv'))
        this_round_variants.to_csv(os.path.join(save_dir, 'this_round_variants.csv'))
        df_test.sort_values('y_pred', ascending=False).to_csv(
            os.path.join(save_dir, 'df_test.csv')
        )
        df_sorted_all.to_csv(os.path.join(save_dir, 'df_sorted_all.csv'))
        print(f'Results saved to {save_dir}')

    return this_round_variants, df_test, df_sorted_all
