
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
def regression(embeddings: pd.DataFrame,
               experimental_data: pd.DataFrame,
               regression_type: str = 'randomforest',
               random_seed: Optional[int] = None):
    """Train a regression model then predict activity of every variant."""

    X = embeddings.loc[experimental_data.index] # sorts embeddings according to the order in experimental_data
    y = experimental_data.iloc[:, 0]    # takes the first column of experimental_data that isn't the labels

    # --- Build and fit model ---
    model = _build_model(regression_type)
    model.fit(X=X, y=y)

    # make activity prediction for each item in the embeddings
    y_pred = model.predict(X=embeddings)

    # make this into a dataframe and combine with the experimental data
    predictions_df = pd.DataFrame(data={'y_pred': y_pred},
                                  index=embeddings.index.tolist())
    predictions_df['y_actual'] = experimental_data['activity']
    return predictions_df


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
