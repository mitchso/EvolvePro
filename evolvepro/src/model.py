
import warnings
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn import linear_model
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, RBF
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import xgboost

warnings.simplefilter(action='ignore', category=FutureWarning)

def zero_shot():
    pass


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


def regression(embeddings: pd.DataFrame,
               experimental_data: pd.DataFrame,
               regression_model = 'randomforest'):
    """Train a regression model then predict activity of every variant."""

    X = embeddings.loc[experimental_data.index] # sorts embeddings according to the order in experimental_data
    y = experimental_data    # use all columns of experimental_data

    y_dimensions = y.shape[1]
    y_names = y.columns.tolist()

    print(f"Fitting regression model to activity with {y_dimensions} dimensions.")

    if y_dimensions == 1: # if there is only one output
        y = y.iloc[:, 0]    # use this to prevent a warning

    if type(regression_model) is str:   # access default model implementations
        model = _build_model(regression_model, y_dimensions)
    else:   # allows for custom models to be passed in
        model = regression_model

    try:
        model.fit(X=X, y=y)
    except Exception as e:
        print(f"Error fitting model: {e}")

    # make activity prediction for each item in the embeddings
    y_pred = model.predict(X=embeddings)

    # make this into a dataframe and combine with the experimental data

    if y_dimensions == 1:   # 1 dimension is simpler
        pred_col = f'{y_names[0]}_pred'
        actual_col = f'{y_names[0]}_actual'

        predictions_df = pd.DataFrame(data={pred_col: y_pred}, index=embeddings.index.tolist())
        predictions_df.sort_values(by=pred_col, inplace=True, ascending=False)  # sort by predicted activity
        predictions_df[actual_col] = experimental_data[y_names[0]]  # add the actual activity
        predictions_df['rank'] = range(1, len(predictions_df) + 1)  # add 1-based rankings
        predictions_df = predictions_df[['rank', actual_col, pred_col]]  # reorganize columns

    else:   # multiple dimensions are more complex
        # generate a dictionary with the predicted activity for each dimension
        pred_cols = {f'{y_names[i]}_pred': y_pred[:, i] for i in range(y_dimensions)}
        pred_cols['sum_pred'] = y_pred.sum(axis=1)  # sum of all dimensions

        # convert to dataframe, add embeddings indices (all possible variants)
        predictions_df = pd.DataFrame(data=pred_cols, index=embeddings.index.tolist())
        predictions_df.sort_values(by='sum_pred', inplace=True, ascending=False)  # sort by overall

        # add back the actual data
        actual_cols = [f"{col}_actual" for col in y_names]
        predictions_df[actual_cols] = experimental_data[y_names]

        predictions_df['rank'] = range(1, len(predictions_df) + 1)  # add 1-based rankings
        predictions_df = predictions_df[['rank'] + actual_cols + list(pred_cols.keys())]  # reorganize columns

    return predictions_df


def _build_model(regression_model: str, y_dimensions: int):
    """Instantiate the chosen scikit-learn / XGBoost regression model."""

    # significantly changed the implementation parameters compared to evolvepro code
    models = {
        'ridge': linear_model.RidgeCV(alphas=np.logspace(-1, 6, 30)),    # penalizes complex parameterization to minimize overfitting
        'lasso': linear_model.LassoCV(max_iter=100_000, tol=1e-4), # shrinks less important coefficients to zero
        'elasticnet': linear_model.ElasticNetCV(max_iter=100_000, tol=1e-4), # combination of ridge and lasso
        'linear': linear_model.LinearRegression(),  # normal linear regression
        'neuralnet': MLPRegressor(
            hidden_layer_sizes=(64, 32),    # much smaller — more appropriate for n=130
            max_iter=2000,                   # more iterations for multi-output convergence
            activation='relu',
            solver='adam',
            alpha=0.01,                      # relax regularization slightly
            early_stopping=True,             # stop when validation loss stops improving
            validation_fraction=0.15,        # use 15% of training data for early stopping
            random_state=1,
        ),
        'randomforest': RandomForestRegressor(  # combine many basic decision trees
            n_estimators=200,
            criterion='friedman_mse',
            random_state=1,
        ),
        'gradientboosting': xgboost.XGBRegressor(   # builds trees sequentially, which correct errors from preceding trees
            objective='reg:squarederror',
            colsample_bytree=0.3,
            colsample_bylevel=0.8,
            colsample_bynode=1.0,
            learning_rate=0.05,
            max_depth=3,
            alpha=1,
            reg_lambda=1,
            n_estimators=200,
            subsample=0.8,
            random_state=1,
        ),
        'gradientboosting_PCA': Pipeline([      # underperforms regular gradient boosting so dont use this
            ('scaler', StandardScaler()),       # PCA is sensitive to feature scale
            ('pca', PCA(n_components=70)),      # 1152 → 70 dims, determined by doing PCA on ESM-C 600M embeddings
            ('gb', xgboost.XGBRegressor(
            objective='reg:squarederror',
            colsample_bytree=0.5,
            colsample_bylevel=0.8,
            colsample_bynode=1.0,
            learning_rate=0.05,
            max_depth=3,
            alpha=1,
            reg_lambda=1,
            n_estimators=200,
            subsample=0.8,
            random_state=1,
        )),
        ]),
        'knn': Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=0.95)),
            ('knn', KNeighborsRegressor(n_neighbors=10, weights='distance')),
            # weights='distance' means closer neighbors matter more, not equal votes
        ]),
        'gp': GaussianProcessRegressor(
            kernel=RBF(length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(),
            alpha=1e-10,
            normalize_y=True,
            n_restarts_optimizer=10,
        ),   # searches possible functions that could fit to data, provides confidence scoring as well
    }

    if y_dimensions > 1:    # update specific models that require a different implementation
        models['lasso'] = linear_model.MultiTaskLassoCV(max_iter=100_000, tol=1e-4)
        models['elasticnet'] = linear_model.MultiTaskElasticNetCV(max_iter=100_000, tol=1e-4)
        models['gradientboosting'] = MultiOutputRegressor(models['gradientboosting']) # wrap previous implementation
        models['gp'] = MultiOutputRegressor(models['gp'])   # wrap previous implementation

    if regression_model not in models:
        raise ValueError(
            f"Unknown regression_type '{regression_model}'. "
            f"Must be one of: {sorted(models.keys())}."
        )

    return models[regression_model]
