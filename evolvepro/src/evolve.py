from evolvepro.src.model import regression
import pandas as pd
from typing import Optional
import os

def evolve(round_name: str,
           embeddings_files: list,
           round_files: list,
           output_dir: Optional[str] = None):

    # Load experimental data into df
    dfs = []
    for file in round_files:
        round_df = validate_csv(file, 'experimental')
        dfs.append(round_df)
    df_experimental = pd.concat(dfs, ignore_index=False)

    # Load embeddings into df
    dfs = []
    for file in embeddings_files:
        embeddings = validate_csv(file, 'embeddings')
        dfs.append(embeddings)
    df_embeddings = pd.concat(dfs, ignore_index=False)

    # check for duplicates in both dataframes
    for df in [df_experimental, df_embeddings]:
        duplicated_indices = df.index[df.index.duplicated()].tolist()
        if duplicated_indices:
            raise ValueError(f"Duplicate variant names found in the input data: {duplicated_indices[:10]}")

    # Do top-layer prediction
    predictions_df = regression(embeddings=df_embeddings, experimental_data=df_experimental)

    # Write results to file
    predictions_df.to_csv(os.path.join(output_dir, f"{round_name}_predictions.csv"))

    return predictions_df


def validate_csv(file_path: str, file_type: str) -> pd.DataFrame:
    """Load and validate a CSV file used by EvolvePro."""

    if file_type == 'embeddings':
        df = pd.read_csv(file_path, index_col=0)    # turns the variant name into the index
    elif file_type == 'experimental':
        df = pd.read_excel(file_path, index_col=0)   # turns the variant name into the index
    else:
        raise ValueError(f"Invalid file type: {file_type}")

    # Sort embeddings by index (variant name)
    df = df.sort_index()

    return df