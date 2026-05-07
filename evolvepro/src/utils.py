"""
utils.py — Shared utility functions for EvolvePro.
"""

import pandas as pd
from sklearn.decomposition import PCA


def pca_embeddings(embeddings_df: pd.DataFrame, n_components: int = 10) -> pd.DataFrame:
    """Reduce a variant embeddings DataFrame to its top principal components.

    Fits PCA on all rows of ``embeddings_df`` and returns a new DataFrame
    containing the projected coordinates.  The variant row index is preserved.

    Args:
        embeddings_df: Numeric DataFrame of variant embeddings, indexed by
            variant name.  Shape: (n_variants, n_embedding_dims).
        n_components: Number of principal components to retain.

    Returns:
        DataFrame of shape (n_variants, n_components) with columns
        ``"PCA 1"``, ``"PCA 2"``, …, ``"PCA <n_components>"``, and the
        original variant name index.
    """
    if embeddings_df.empty:
        raise ValueError(
            "embeddings_df is empty. Cannot perform PCA on an empty DataFrame."
        )

    if n_components < 1:
        raise ValueError(
            f"n_components must be at least 1, got {n_components}."
        )

    max_components = min(embeddings_df.shape)
    if n_components > max_components:
        raise ValueError(
            f"n_components ({n_components}) exceeds the maximum possible for this "
            f"embeddings matrix ({max_components}, which is min(n_variants={embeddings_df.shape[0]}, "
            f"n_dims={embeddings_df.shape[1]})). "
            f"Reduce n_components to at most {max_components}."
        )

    non_numeric_cols = [
        col for col in embeddings_df.columns
        if not pd.api.types.is_numeric_dtype(embeddings_df[col])
    ]
    if non_numeric_cols:
        raise ValueError(
            f"embeddings_df contains non-numeric column(s): {non_numeric_cols}. "
            f"All embedding dimensions must be numeric before PCA can be applied."
        )

    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(embeddings_df)[:, :n_components]
    return pd.DataFrame(
        reduced,
        index=embeddings_df.index,
        columns=[f'PCA {i}' for i in range(1, n_components + 1)],
    )
