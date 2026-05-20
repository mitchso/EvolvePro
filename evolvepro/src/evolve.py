import os
import sys
import pandas as pd
from evolvepro.src.model import regression

def evolve(embeddings_files: list,
           round_files: list,
           output_file: str = None,
           regression_model: str = 'randomforest'):

    # Load experimental data into df
    df_experimental = load_experimental(round_files)

    # Load embeddings into df
    df_embeddings = load_embeddings(embeddings_files)

    # check for duplicates in both dataframes
    for df in [df_experimental, df_embeddings]:
        duplicated_indices = df.index[df.index.duplicated()].tolist()
        if duplicated_indices:
            raise ValueError(f"{len(duplicated_indices)} duplicate variant name(s) found in the input data: {duplicated_indices[:10]}")

    # Make sure every activity score in df_experimental is represented by an embedding in df_embeddings
    missing_indices = df_experimental.index.difference(df_embeddings.index)
    if len(missing_indices) > 0:
        num_missing = len(missing_indices)
        raise ValueError(f"{num_missing} variant(s) in round_files not represented by "
                         f"embeddings in embeddings_files: {missing_indices.tolist()}")

    # Do top-layer prediction
    predictions_df = regression(embeddings=df_embeddings,
                                experimental_data=df_experimental,
                                regression_model=regression_model)

    # Write results
    predictions_df.to_csv(output_file)
    print(f"Wrote predictions to {output_file}")

    return predictions_df


def load_embeddings(files: str | list[str]) -> pd.DataFrame:
    """Loads a list of files into a single dataframe."""
    if type(files) != list:
        files = [files]

    dfs = []
    for file in files:
        try:
            df = pd.read_csv(file, index_col=0) # turns the variant name into the index
            dfs.append(df)
        except FileNotFoundError:
            print(f"Could not locate file: {file}")
            sys.exit()

    df = pd.concat(dfs, ignore_index=False)

    df = df.sort_index()

    print(f"Loaded embeddings with shape: {df.shape} -> (variants, embeddings_width)")
    print(f"Quick peek:")
    print(df.iloc[:5, :5])

    return df


def load_experimental(files: str | list[str]) -> pd.DataFrame:
    """Loads a list of files into a single dataframe."""
    if type(files) != list:
        files = [files]

    dfs = []
    for file in files:      # TODO: force explicit y_dim loading and checking
        try:
            df = pd.read_excel(file, index_col=0)  # turns the variant name into the index
            dfs.append(df)
        except FileNotFoundError:
            print(f"Could not locate file: {file}")
            sys.exit()

    df = pd.concat(dfs, ignore_index=False)

    df = df.sort_index()

    print(f"Loaded experimental data with shape: {df.shape} -> (variants, activity_dimensions)")
    print(f"Quick peek:")
    print(df.head())

    return df
