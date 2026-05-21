# EVOLVEpro fork

This is my reworked version of EVOLVEpro (https://github.com/mat10d/EvolvePro).
The main differences between this repo and the original are:
- This codebase is trimmed down to only include necessary materials for running the evolvepro workflow on experimental datasets. The ability to analyze DMS datasets has been removed.
- This codebase now supports embedding extractions from ESM models only, but has added support for ESM-C family models, whereas the original EvolvePro supports ESM-1/2.
- Support for MPS on macs, to allow for GPU usage during embedding extraction.
- Maintained the underlying logic of the original code but restructured inputs, outputs, and function calls (new usages are described below).
- Support for arbitrary regression models and user-defined parameters as desired - however the default is still a random forest model as recommended by EvolvePro.
- Additional error-handling and documentation.

## Overview

The EVOLVEpro workflow consists of four main steps:

| Step                  | Description                                                                                                                                                                                                                                                                                                                                          | Input                                                  | Output                                                                                              |
|-----------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------|-----------------------------------------------------------------------------------------------------|
| Process               | Generate sequence files. This step defines the sequence space that you are searching in. Initially, this should be all single amino acid mutations to a wild-type sequence. <br/><br/>Later, you may wish to explore multi-mutants, and so you will need to produce sequences corresponding to the multi-mutants you want to include in your search. | Wild-type sequence in fasta format                     | Multifasta files, for example containing every possible single amino acid mutation to the wild-type |
| Embedding extraction  | Produce mean-pooled representations of each sequence                                                                                                                                                                                                                                                                                                 | Multifasta file (one entry per sequence)               | csv file (one row per sequence)                                                                     |
| Zero-shot suggestions | Suggest mutations to begin the experimental evolution process                                                                                                                                                                                                                                                                                        | embeddings csv file                                    | Suggested mutants                                                                                   |
| Regression            | Model the activity landscape to prioritize mutations for subsequent rounds                                                                                                                                                                                                                                                                           | embeddings csv file, experimental data in excel format | activity predictions in csv format                                                                  |

## Step-by-Step Description

### 1. Process

Generate FASTA files containing protein variant sequences.

For detailed instructions, see the [Experimental README](https://github.com/mitchso/EvolvePro/tree/main/evolvepro/src).

### 2. Embedding extraction

Extract protein language model embeddings for all variants using various PLM models.

For detailed instructions, see the [PLM README](plm/README.md).

### 3. Zero-shot suggestions

Use this workflow to generate initial mutations for the experimental evolution process.

For detailed instructions, see the [Experimental README](https://github.com/mitchso/EvolvePro/tree/main/evolvepro/src).

### 4. Regression

Apply the EVOLVEpro model to optimize protein activity.

#### Experimental Workflow
Use this workflow for iterative experimental optimization of protein activity.

For detailed instructions, see the [Experimental README](https://github.com/mitchso/EvolvePro/tree/main/evolvepro/src).

## Getting Started

### Install

```bash
git clone https://github.com/mitchso/EvolvePro.git
cd EvolvePro
```

### EVOLVEpro Environment

First, create and activate a conda environment with all necessary dependencies for EVOLVEpro:

```bash
conda env create -f envs/base.yml
conda activate evolvepro
```

### Protein Language Models Environments

Dependencies are different depending on which PLM you want to use. Because of this, there is one environment for ESM-1/ESM-2 family, and a different environment for ESM-C family.

```bash
sh setup_esm2.sh
conda activate esm2
```
```bash
sh setup_esmc.sh
conda activate esmc
```

These environments are kept separate to maintain clean dependencies and avoid conflicts between the core EVOLVEpro functionality and the various protein language models.

## Examples

See the [examples](examples) directory for example scripts.

## Citation

If you use this code in your research, please cite the original paper:

```
@ARTICLE
author={Jiang, Kaiyi and Yan, Zhaoqing and Di Bernardo, Matteo and Sgrizzi, Samantha R. and Villiger, Lukas and Kayabölen, Alişan and Kim, Byungji and Carscadden, Josephine K. and Hiraizumi, Masahiro and Nishimasu, Hiroshi and Gootenberg, Jonathan S. and Abudayyeh, Omar O.}
title={Rapid in silico directed evolution by a protein language model with EVOLVEpro}, 
year={2024},
DOI={10.1126/science.adr6006}
```