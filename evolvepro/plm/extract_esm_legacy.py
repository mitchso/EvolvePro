#!/usr/bin/env python3 -u
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# This is meant to be used with the fair-esm package.
# Newer models (ESM-3, ESM-C) need to be accessed using the new esm package

import os
import shutil
import argparse
import pathlib
import pandas as pd
import torch
import re

from esm import Alphabet, FastaBatchedDataset, ProteinBertModel, pretrained, MSATransformer


def create_parser():
    parser = argparse.ArgumentParser(
        description="Extract per-token representations and model outputs for sequences in a FASTA file"
    )

    parser.add_argument(
        "model_location",
        type=str,
        help="PyTorch model file OR name of pretrained model to download (see README for models)",
    )
    parser.add_argument(
        "fasta_file",
        type=pathlib.Path,
        help="FASTA file on which to extract representations",
    )
    parser.add_argument(
        "output_dir",
        type=pathlib.Path,
        help="output directory for extracted representations",
    )

    parser.add_argument("--toks_per_batch", type=int, default=4096, help="maximum batch size")

    parser.add_argument(
        "--repr_layers",
        type=int,
        default=[-1],
        nargs="+",
        help="layers indices from which to extract representations (0 to num_layers, inclusive)",
    )
    parser.add_argument(
        "--include",
        type=str,
        nargs="+",
        choices=["mean", "per_tok", "bos", "contacts"],
        help="specify which representations to return",
        required=True,
    )

    parser.add_argument(
        "--truncation_seq_length",
        type=int,
        default=1022,
        help="truncate sequences longer than the given value",
    )

    parser.add_argument("--nogpu", action="store_true", help="Do not use GPU even if available")

    return parser


def get_device(nogpu=False):
    if nogpu:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_directory(path: pathlib.Path, print_error=True) -> pathlib.Path:
    try:
        path.mkdir(parents=True, exist_ok=False)
        print(f"Created directory {path}.")
        return path
    except FileExistsError:
        if print_error:
            print(f"Output directory {path} already exists.")
        dir_string = str(path)
        match = re.search(pattern=r'^(.*?)_(\d+)$', string=dir_string)
        if match:
            prefix, number = match.group(1), int(match.group(2))
            suffix = number + 1
            dir_string = f"{prefix}_{suffix}"
        else:
            suffix = 1
            dir_string = f"{path}_{suffix}"

        path = pathlib.Path(dir_string)
        # recursive until you find a path that doesn't already exist'
        return make_directory(path, print_error=False)


def run(args):
    model, alphabet = pretrained.load_model_and_alphabet(args.model_location)
    model.eval()
    if isinstance(model, MSATransformer):
        raise ValueError(
            "This script currently does not handle models with MSA input (MSA Transformer)."
        )

    device = get_device(args.nogpu)
    model = model.to(device, dtype=torch.float32)  # testing whether bfloat16 increases performance on my mac
    print(f"Using device: {device}")

    dataset = FastaBatchedDataset.from_file(args.fasta_file)
    batches = dataset.get_batch_indices(args.toks_per_batch, extra_toks_per_seq=1)
    data_loader = torch.utils.data.DataLoader(dataset, collate_fn=alphabet.get_batch_converter(), batch_sampler=batches)
    print(f"Read {args.fasta_file} with {len(dataset)} sequences")

    args.output_dir = make_directory(args.output_dir)
    print(f"Output directory: {args.output_dir}")

    return_contacts = "contacts" in args.include

    assert all(-(model.num_layers + 1) <= i <= model.num_layers for i in args.repr_layers)
    repr_layers = [(i + model.num_layers + 1) % (model.num_layers + 1) for i in args.repr_layers]

    with torch.no_grad():
        for batch_idx, (labels, strs, toks) in enumerate(data_loader):
            print(f"Processing {batch_idx + 1} of {len(batches)} batches ({toks.size(0)} sequences)")

            toks = toks.to(device=device)
            print(f"Device: {toks.device}")

            out = model(toks, repr_layers=repr_layers, return_contacts=return_contacts)

            # unused call to logits?
            # logits = out["logits"].to(device="cpu", dtype=torch.float32)
            # print(f"Device: {logits.device}, dtype: {logits.dtype}, shape: {logits.shape}")

            representations = {layer: t.to(device="cpu", dtype=torch.float32) for layer, t in out["representations"].items()}

            if return_contacts:
                contacts = out["contacts"].to(device="cpu", dtype=torch.float32)

            for i, label in enumerate(labels):
                args.output_file = args.output_dir / "pt" / f"{label}.pt"
                args.output_file.parent.mkdir(parents=True, exist_ok=True)
                result = {"label": label}
                truncate_len = min(args.truncation_seq_length, len(strs[i]))
                # Call clone on tensors to ensure tensors are not views into a larger representation
                # See https://github.com/pytorch/pytorch/issues/1995
                if "per_tok" in args.include:
                    result["representations"] = {
                        layer: t[i, 1 : truncate_len + 1].clone()
                        for layer, t in representations.items()
                    }
                if "mean" in args.include:
                    result["mean_representations"] = {
                        layer: t[i, 1 : truncate_len + 1].mean(0).clone()
                        for layer, t in representations.items()
                    }
                if "bos" in args.include:
                    result["bos_representations"] = {
                        layer: t[i, 0].clone() for layer, t in representations.items()
                    }
                if return_contacts:
                    result["contacts"] = contacts[i, : truncate_len, : truncate_len].clone()

                torch.save(result, args.output_file)

    print(f"Saved representations to {args.output_dir}")

def concatenate_files(output_dir, output_csv):

    # Get all .pt files in the output directory
    files = []
    for r, d, f in os.walk(output_dir / "pt"):
        for file in f:
            if '.pt' in file:
                files.append(os.path.join(r, file))

    # Load each file and append to a list of dataframes
    dataframes = []
    for file_path in files:
        file_data = torch.load(file_path, weights_only=True)
        label = file_data['label']
        representations = file_data['mean_representations']
        key, tensor = representations.popitem()
        row_name = label
        row_data = tensor.tolist()
        new_df = pd.DataFrame([row_data], index=[row_name])
        dataframes.append(new_df)

    # Concatenate all dataframes
    if dataframes:
        concatenated_df = pd.concat(dataframes)
        print("Shape of concatenated DataFrame:", concatenated_df.shape)
        concatenated_df.to_csv(output_dir / output_csv, index=True)
        print(f"Saved concatenated representations to {output_csv}")
    else:
        print("No data to concatenate.")

def main():
    parser = create_parser()
    args = parser.parse_args()
    
    run(args)

    fasta_file_name = args.fasta_file.stem
    output_csv = f"{fasta_file_name}_{args.model_location}.csv"
    concatenate_files(args.output_dir, output_csv)



if __name__ == "__main__":
    main()