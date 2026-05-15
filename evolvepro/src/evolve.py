from evolvepro.src.data import *
from evolvepro.src.model import regression

def evolve(
        round_name: str,
        embeddings_files: list,
        round_files: list,
        output_dir: Optional[str] = None,
):
    """
    Original evolve_experimental() and evolve_experimental_multi() functions are the same thing but
    with slightly different handling. I'm cleaning them up and merging them into one thing function.

    Args:
        round_name:
        embeddings_file:
        round_files:
        wt_fasta_path:
        output_dir:

    Returns:

    """

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
