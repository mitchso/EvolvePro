"""
evolve.py — Top-level orchestration for EvolvePro.

- ``evolve_experimental`` — process one or more completed assay rounds and
  output ranked predictions for the *next* round (the primary function used in
  day-to-day experimental campaigns).

- ``evolve_experimental_multi`` — like ``evolve_experimental`` but supports
  embedding files that mix single-mutant and multi-mutant variants across rounds.
"""

from evolvepro.src.data import *
from evolvepro.src.model import top_layer
import sys

def evolve_experimental(
    round_name: str,
    embeddings_file: str,
    round_files: List[str],
    wt_fasta_path: str,
    output_dir: Optional[str] = None,
) -> None:

    print(f'Processing {round_name}')

    embeddings = validate_csv(embeddings_file, 'embeddings')

    all_experimental_data = []
    for fname in round_files:
        exp_data = load_experimental_data(fname, wt_fasta_path)
        all_experimental_data.append(exp_data)
        print(f'Loaded {exp_data.shape[0]} variants from {fname.split("/")[-1]}')

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

    if output_dir is not None:
        save_dir = os.path.join(output_dir, round_name)
        try:
            os.makedirs(save_dir, exist_ok=False)
            iteration.to_csv(os.path.join(save_dir, 'iteration.csv'))
            this_round_variants.to_csv(os.path.join(save_dir, 'this_round_variants.csv'))
            df_test.sort_values('y_pred', ascending=False).to_csv(
                os.path.join(save_dir, 'df_test.csv')
            )
            df_sorted_all.to_csv(os.path.join(save_dir, 'df_sorted_all.csv'))
            print(f'Results saved to {save_dir}')

        except FileExistsError:
            print(f"Output directory {save_dir} already exists. No files written.")



def evolve_experimental_multi(
    round_name: str,
    embeddings_files: List[str],
    round_files_single: List[str],
    round_files_multi: List[str],
    wt_fasta_path: str,
    output_dir: Optional[str] = None,
):

    print(f'Processing {round_name}')

    # --- Load and concatenate embedding files, keeping only one WT row ---
    embeddings_parts = []
    for i, fname in enumerate(embeddings_files):
        emb = load_experimental_embeddings(fname)
        if i > 0 and 'WT' in emb.index:
            warnings.warn(
                f"Embeddings file '{fname}' (file {i + 1} of "
                f"{len(embeddings_files)}) contains a 'WT' row. "
                f"WT is retained from the first file only; "
                f"this row will be dropped. To avoid this warning, omit WT from all "
                f"files except the first (see README, multi-mutant embeddings).",
                UserWarning,
                stacklevel=2,
            )
            emb = emb[emb.index != 'WT']
        embeddings_parts.append(emb)

    # --- Warn if embedding files have inconsistent dimensionality ---
    dims = [part.shape[1] for part in embeddings_parts]
    if len(set(dims)) > 1:
        dim_info = {embeddings_files[i]: dims[i] for i in range(len(dims))}
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
    for fname in round_files_single:
        exp_data = load_experimental_data(fname, wt_fasta_path)
        all_experimental_data.append(exp_data)
        print(f'Loaded {exp_data.shape[0]} variants from {fname.split("/")[-1]}')

    # --- Load multi-mutant rounds ---
    for fname in round_files_multi:
        exp_data = load_experimental_data(fname, wt_fasta_path)
        all_experimental_data.append(exp_data)
        print(f'Loaded {exp_data.shape[0]} variants from {fname.split("/")[-1]}')

    iteration, labels = create_iteration_dataframes(all_experimental_data, embeddings.index.tolist())

    this_round_variants, df_test, df_sorted_all = top_layer(
        iter_train=iteration['iteration'].unique().tolist(),
        iter_test=None,
        embeddings_pd=embeddings,
        labels_pd=labels,
        measured_var='activity',
        regression_type='randomforest',
        experimental=True,
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

    # return this_round_variants, df_test, df_sorted_all