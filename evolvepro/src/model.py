
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
