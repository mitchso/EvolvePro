"""
model.py — Active-learning model for EvolvePro.

Two public functions:

- ``first_round`` — selects the initial panel of variants to test before any
  measurements exist, using either random sampling or diversity-maximising
  K-medoids clustering.

- ``top_layer`` — trains a regression model on the measured variants and
  predicts activity for all untested variants, returning ranked candidates for
  the next round.
"""

import warnings
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn import linear_model
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
import xgboost

warnings.simplefilter(action='ignore', category=FutureWarning)

_VALID_REGRESSION_TYPES = frozenset({
    'ridge', 'lasso', 'elasticnet', 'linear',
    'neuralnet', 'randomforest', 'gradientboosting', 'knn', 'gp',
})

_VALID_FIRST_ROUND_STRATEGIES = frozenset({
    'random', 'diverse_medoids', 'explicit_variants',
})


# ---------------------------------------------------------------------------
# First-round variant selection
# ---------------------------------------------------------------------------

def first_round(
    labels: pd.DataFrame,
    embeddings: pd.DataFrame,
    explicit_variants: Optional[List[str]] = None,
    num_mutants_per_round: int = 16,
    first_round_strategy: str = 'random',
    embedding_type: Optional[str] = None,
    random_seed: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Select the initial set of variants to test before any measurements exist.

    Three strategies are supported:

    - ``"random"`` — draw variants uniformly at random (excluding WT).
    - ``"diverse_medoids"`` — cluster all variants in embedding space with
      K-medoids and return the medoid of each cluster.  This maximises coverage
      of sequence space in the first round.  Requires the optional dependency
      ``sklearn_extra``.
    - ``"explicit_variants"`` — use a caller-supplied list of variant names.

    Args:
        labels: Full variant labels DataFrame with at minimum a ``variant`` column.
        embeddings: DataFrame of per-variant embeddings indexed by variant name.
        explicit_variants: Variant names to use when ``first_round_strategy`` is
            ``"explicit_variants"``.  Ignored otherwise.
        num_mutants_per_round: Number of variants to select (excluding WT).
        first_round_strategy: One of ``"random"``, ``"diverse_medoids"``, or
            ``"explicit_variants"``.
        embedding_type: Set to ``"embeddings_pca"`` to skip the internal PCA step
            in the ``"diverse_medoids"`` strategy (i.e. embeddings are already
            reduced).  ``None`` triggers PCA to 10 components before clustering.
        random_seed: Integer seed for reproducibility.

    Returns:
        ``(labels_zero, iteration_zero, this_round_variants)``

        - ``labels_zero`` — ``labels`` left-joined with iteration assignments.
        - ``iteration_zero`` — DataFrame with ``variant`` and ``iteration``
          columns; iteration is 0 for all selected variants and for WT.
        - ``this_round_variants`` — Series of selected variant names.
    """
    # --- Validate inputs ---
    if 'variant' not in labels.columns:
        raise ValueError(
            "labels DataFrame is missing the required 'variant' column."
        )

    if first_round_strategy not in _VALID_FIRST_ROUND_STRATEGIES:
        raise ValueError(
            f"Unknown first_round_strategy '{first_round_strategy}'. "
            f"Must be one of: {sorted(_VALID_FIRST_ROUND_STRATEGIES)}."
        )

    if first_round_strategy == 'explicit_variants':
        if not explicit_variants:
            raise ValueError(
                "first_round_strategy='explicit_variants' requires a non-empty "
                "explicit_variants list."
            )
        # Warn about any requested variants not present in the labels.
        label_variant_set = set(labels['variant'])
        missing = [v for v in explicit_variants if v not in label_variant_set]
        if missing:
            warnings.warn(
                f"The following explicit_variants are not present in the labels "
                f"DataFrame and will be ignored: {missing}. "
                f"Ensure variant names exactly match those in the embeddings index.",
                UserWarning,
                stacklevel=2,
            )

    print(f'Total variants: {len(labels)}')
    variants_without_wt = labels.loc[labels['variant'] != 'WT', 'variant']
    print(f'Non-WT variants: {len(variants_without_wt)}')

    if num_mutants_per_round > len(variants_without_wt):
        raise ValueError(
            f"num_mutants_per_round ({num_mutants_per_round}) exceeds the number "
            f"of available non-WT variants ({len(variants_without_wt)}). "
            f"Reduce num_mutants_per_round or increase the candidate library."
        )

    if first_round_strategy == 'random':
        if random_seed is not None:
            np.random.seed(random_seed)
        selected = np.random.choice(
            variants_without_wt, size=num_mutants_per_round, replace=False
        )

    elif first_round_strategy == 'diverse_medoids':
        try:
            from sklearn_extra.cluster import KMedoids
        except ImportError:
            raise ImportError(
                "first_round_strategy='diverse_medoids' requires the 'sklearn_extra' "
                "package. Install it with: pip install scikit-learn-extra"
            )

        if random_seed is not None:
            np.random.seed(random_seed)

        print(f'Embeddings available: {len(embeddings)}')
        embeddings_no_wt = (
            embeddings.drop('WT') if 'WT' in embeddings.index else embeddings.copy()
        )
        if 'WT' not in embeddings.index:
            warnings.warn(
                "No 'WT' row found in embeddings for diverse_medoids clustering. "
                "Proceeding without removing WT from the candidate pool.",
                UserWarning,
                stacklevel=2,
            )
        print(f'Embeddings (no WT): {len(embeddings_no_wt)}')

        if embedding_type != 'embeddings_pca':
            print('Applying PCA (10 components) before clustering.')
            pca = PCA(n_components=min(10, embeddings_no_wt.shape[1]))
            reduced = pca.fit_transform(embeddings_no_wt)[:, :10]
        else:
            reduced = embeddings_no_wt.values

        km = KMedoids(
            n_clusters=num_mutants_per_round,
            metric='euclidean',
            random_state=random_seed,
        ).fit(reduced)
        selected = embeddings_no_wt.index[km.medoid_indices_].tolist()

    elif first_round_strategy == 'explicit_variants':
        selected = explicit_variants

    iteration_zero = pd.concat(
        [
            pd.DataFrame({'variant': selected, 'iteration': 0}),
            pd.DataFrame({'variant': ['WT'], 'iteration': [0]}),
        ],
        ignore_index=True,
    )
    labels_zero = pd.merge(labels, iteration_zero, on='variant', how='left')
    return labels_zero, iteration_zero, iteration_zero['variant']


# ---------------------------------------------------------------------------
# Active-learning model (subsequent rounds)
# ---------------------------------------------------------------------------

def top_layer(
    iter_train: List[int],
    iter_test: Optional[int],
    embeddings_pd: pd.DataFrame,
    labels_pd: pd.DataFrame,
    measured_var: str,
    regression_type: str = 'randomforest',
    top_n: Optional[int] = None,
    final_round: int = 10,
    experimental: bool = False,
) -> Union[Tuple, None]:
    """Train a regression model and rank untested variants by predicted activity.

    Trains on all variants whose ``iteration`` value appears in ``iter_train``,
    and predicts on the held-out set (either ``iter_test`` or, when
    ``experimental=True``, all rows with ``NaN`` iteration).

    Supported regression models (``regression_type``):
        ``"ridge"``, ``"lasso"``, ``"elasticnet"``, ``"linear"``,
        ``"neuralnet"``, ``"randomforest"`` (default), ``"gradientboosting"``,
        ``"knn"``, ``"gp"``

    Args:
        iter_train: List of iteration numbers whose variants form the training set.
        iter_test: Iteration number for the test set.  Pass ``None`` in
            experimental mode to use all unassigned (``NaN``) variants as the
            test set.
        embeddings_pd: DataFrame of variant embeddings (one row per variant).
            Row order must match ``labels_pd``.
        labels_pd: DataFrame with at minimum ``variant``, ``iteration``,
            ``activity``, ``activity_scaled``, and ``activity_binary`` columns.
            Row order must match ``embeddings_pd``.
        measured_var: Name of the column in ``labels_pd`` to use as the
            regression target (typically ``"activity"``).
        regression_type: Which scikit-learn / XGBoost model to use.
        top_n: Reserved for future use (currently unused).
        final_round: Number of top predictions to include when computing
            summary metrics (``median_activity_scaled``, etc.).
        experimental: If ``True``, operate in experimental mode:
            - Test set = variants with ``NaN`` iteration (not yet measured).
            - MSE / R² are not computed (no ground truth).
            - Returns a simplified 3-tuple.

    Returns:
        **Experimental mode** (``experimental=True``):
            ``(this_round_variants, df_test, df_sorted_all)``

            - ``this_round_variants`` — Series of variant names used for training.
            - ``df_test`` — predictions for all untested variants, unsorted.
            - ``df_sorted_all`` — all variants ranked by predicted activity.

        **Simulation mode** (``experimental=False``):
            ``(train_error, test_error, train_r_squared, test_r_squared, alpha,
            median_activity_scaled, top_activity_scaled, top_variant,
            top_final_round_variants, activity_binary_percentage,
            spearman_corr, df_test, this_round_variants)``
    """
    # --- Validate regression_type early ---
    if regression_type not in _VALID_REGRESSION_TYPES:
        raise ValueError(
            f"Unknown regression_type '{regression_type}'. "
            f"Must be one of: {sorted(_VALID_REGRESSION_TYPES)}."
        )

    # --- Validate measured_var ---
    if measured_var not in labels_pd.columns:
        raise ValueError(
            f"measured_var '{measured_var}' is not a column in labels_pd. "
            f"Available columns: {list(labels_pd.columns)}."
        )

    required_label_cols = {'variant', 'iteration', 'activity_scaled', 'activity_binary'}
    missing_cols = required_label_cols - set(labels_pd.columns)
    if missing_cols:
        raise ValueError(
            f"labels_pd is missing required column(s): {sorted(missing_cols)}. "
            f"Ensure labels_pd was produced by create_iteration_dataframes()."
        )

    # --- Validate alignment in experimental mode ---
    if experimental:
        if labels_pd['variant'].tolist() != embeddings_pd.index.tolist():
            raise ValueError(
                'Embeddings and labels are not aligned: the variant order differs. '
                'Ensure both are sorted by variant name before calling top_layer() '
                '(see README §3, Step 4).'
            )

    # --- Validate that embeddings and labels have the same number of rows ---
    if len(embeddings_pd) != len(labels_pd):
        raise ValueError(
            f"embeddings_pd has {len(embeddings_pd)} rows but labels_pd has "
            f"{len(labels_pd)} rows. They must have the same number of rows in "
            f"the same order."
        )

    # Reset indices so positional selection with .loc is safe.
    embeddings_pd = embeddings_pd.reset_index(drop=True)
    labels_pd = labels_pd.reset_index(drop=True)
    iteration = labels_pd['iteration']

    # --- Build train / test split ---
    idx_train = iteration[iteration.isin(iter_train)].index.to_numpy()
    if len(idx_train) == 0:
        raise ValueError(
            f"No training variants found for iter_train={iter_train}. "
            f"Check that the iteration values in labels_pd match iter_train."
        )

    if iter_test is not None:
        idx_test = iteration[iteration == iter_test].index.to_numpy()
        if len(idx_test) == 0:
            raise ValueError(
                f"No test variants found for iter_test={iter_test}. "
                f"Check that the iteration values in labels_pd match iter_test."
            )
    else:
        idx_test = iteration[iteration.isna()].index.to_numpy()
        if len(idx_test) == 0:
            warnings.warn(
                "No untested variants (NaN iteration) found in labels_pd. "
                "The prediction pool is empty — all variants have been measured. "
                "df_test will be empty.",
                UserWarning,
                stacklevel=2,
            )

    X_train = embeddings_pd.loc[idx_train]
    X_test = embeddings_pd.loc[idx_test]

    y_train = labels_pd.loc[idx_train, measured_var]
    y_train_scaled = labels_pd.loc[idx_train, 'activity_scaled']
    y_train_binary = labels_pd.loc[idx_train, 'activity_binary']

    if iter_test is not None:
        idx_test_mask = iteration.isin([iter_test])
    else:
        idx_test_mask = iteration.isna()
    y_test = labels_pd.loc[idx_test_mask, measured_var]
    y_test_scaled = labels_pd.loc[idx_test_mask, 'activity_scaled']
    y_test_binary = labels_pd.loc[idx_test_mask, 'activity_binary']

    # --- Check embedding dimensionality consistency ---
    if not experimental and X_train.shape[1] != X_test.shape[1]:
        raise ValueError(
            f"Training embeddings have {X_train.shape[1]} dimensions but test "
            f"embeddings have {X_test.shape[1]} dimensions. All variants must use "
            f"the same PLM and extraction layer (see README §3)."
        )

    # --- Build and fit model ---
    model = _build_model(regression_type)
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test) if len(X_test) > 0 else np.array([])

    y_std_train = np.zeros(len(y_pred_train))
    y_std_test = np.zeros(len(y_pred_test))

    # --- Metrics ---
    train_error = mean_squared_error(y_train, y_pred_train)
    train_r_squared = r2_score(y_train, y_pred_train)

    if not experimental:
        test_error = mean_squared_error(y_test, y_pred_test)
        test_r_squared = r2_score(y_test, y_pred_test)
    else:
        test_error = None
        test_r_squared = None

    if regression_type in ('ridge', 'lasso', 'elasticnet'):
        alpha = model.alpha_
    else:
        alpha = 0

    dist_train = cdist(X_train, X_test, metric='euclidean').min(axis=1) if len(X_test) > 0 else np.full(len(X_train), np.nan)
    dist_test = (
        None
        if experimental
        else cdist(X_test, X_train, metric='euclidean').min(axis=1)
    )

    # --- Assemble result DataFrames ---
    df_train = pd.DataFrame({
        'variant': labels_pd.loc[idx_train, 'variant'].values,
        'y_pred': y_pred_train,
        'y_actual': y_train.values,
        'y_actual_scaled': y_train_scaled.values,
        'y_actual_binary': y_train_binary.values,
        'dist_metric': dist_train,
        'std_predictions': y_std_train,
    })
    df_test = pd.DataFrame({
        'variant': labels_pd.loc[idx_test, 'variant'].values,
        'y_pred': y_pred_test,
        'y_actual': y_test.values,
        'y_actual_scaled': y_test_scaled.values,
        'y_actual_binary': y_test_binary.values,
        'dist_metric': dist_test,
        'std_predictions': y_std_test,
    })

    df_sorted_all = (
        pd.concat([df_train, df_test])
        .sort_values('y_pred', ascending=False)
        .reset_index(drop=True)
    )

    this_round_variants = df_train['variant']

    if experimental:
        return this_round_variants, df_test, df_sorted_all

    # --- Summary metrics (simulation mode only) ---
    top_slice = df_sorted_all.loc[:final_round]
    median_activity_scaled = top_slice['y_actual_scaled'].median()
    top_activity_scaled = top_slice['y_actual_scaled'].max()
    top_variant = df_sorted_all.loc[
        df_sorted_all['y_actual_scaled'] == top_activity_scaled, 'variant'
    ].values[0]
    top_final_round_variants = ','.join(top_slice['variant'].tolist())
    spearman_corr = (
        df_sorted_all[['y_pred', 'y_actual']].corr(method='spearman').iloc[0, 1]
    )
    activity_binary_percentage = top_slice['y_actual_binary'].mean()

    return (
        train_error,
        test_error,
        train_r_squared,
        test_r_squared,
        alpha,
        median_activity_scaled,
        top_activity_scaled,
        top_variant,
        top_final_round_variants,
        activity_binary_percentage,
        spearman_corr,
        df_test,
        this_round_variants,
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _build_model(regression_type: str):
    """Instantiate the chosen scikit-learn / XGBoost regression model.

    Args:
        regression_type: One of ``"ridge"``, ``"lasso"``, ``"elasticnet"``,
            ``"linear"``, ``"neuralnet"``, ``"randomforest"``,
            ``"gradientboosting"``, ``"knn"``, ``"gp"``.

    Returns:
        An unfitted regression estimator.

    Raises:
        ValueError: If ``regression_type`` is not recognised.
    """
    models = {
        'ridge': linear_model.RidgeCV(),
        'lasso': linear_model.LassoCV(max_iter=100_000, tol=1e-3),
        'elasticnet': linear_model.ElasticNetCV(max_iter=100_000, tol=1e-3),
        'linear': linear_model.LinearRegression(),
        'neuralnet': MLPRegressor(
            hidden_layer_sizes=(5,),
            max_iter=1000,
            activation='relu',
            solver='adam',
            alpha=0.001,
            random_state=1,
        ),
        'randomforest': RandomForestRegressor(
            n_estimators=100,
            criterion='friedman_mse',
            random_state=1,
        ),
        'gradientboosting': xgboost.XGBRegressor(
            objective='reg:squarederror',
            colsample_bytree=0.3,
            learning_rate=0.1,
            max_depth=5,
            alpha=10,
            n_estimators=10,
        ),
        'knn': KNeighborsRegressor(n_neighbors=5),
        'gp': GaussianProcessRegressor(),
    }

    # regression_type is validated before this function is called, so this
    # branch is a safety net for direct internal calls.
    if regression_type not in models:
        raise ValueError(
            f"Unknown regression_type '{regression_type}'. "
            f"Must be one of: {sorted(models.keys())}."
        )
    return models[regression_type]
