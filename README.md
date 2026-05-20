# EVOLVEpro - reworked

This is my reworked version of EVOLVEpro (https://github.com/mat10d/EvolvePro).
The main differences between this repo and the original are:
- This codebase is trimmed down to only include necessary materials for running the evolvepro workflow on experimental datasets. The ability to analyze DMS datasets has been completely removed.
- This codebase now supports embedding extractions from ESM models only, but has added support for modern ESM-3 and ESM-C family models, whereas the original EvolvePro supports only up to ESM-2.
- Support for MPS on macs, to allow for GPU usage during embedding extraction.
- Maintained the underlying logic of the original code but reimplemented to simplify inputs, outputs, and function calls (new usages are described below).
- Support for arbitrary regression models and user-defined parameters as desired - however the default is still a random forest model as recommended by EvolvePro.
- Various internal changes to improve clarity and error-handling.

## Overview

The EVOLVEpro workflow consists of three main steps:

1. **Process**: Generate sequence files for input.
2. **PLM**: Extract protein language model (PLM) embeddings for sequences
3. **Run EVOLVEpro**: Apply the model to experimental data

## Step-by-Step Description

### 1. Process

Generate and clean FASTA and CSV files containing protein variant sequences and their corresponding activity data.

For detailed instructions, see the [Process README](scripts/process/README.md).

### 2. PLM

Extract protein language model embeddings for all variants using various PLM models.

For detailed instructions, see the [PLM README](scripts/plm/README.md).

### 3. Run EVOLVEpro

Apply the EVOLVEpro model to optimize protein activity.

#### Experimental Workflow
Use this workflow for iterative experimental optimization of protein activity.

For detailed instructions, see the [Experimental README](scripts/exp/README.md).

## Getting Started

### Install

```bash
git clone https://github.com/mat10d/EvolvePro.git
cd EvolvePro
```

### EVOLVEpro Environment

First, create and activate a conda environment with all necessary dependencies for EVOLVEpro:

```bash
conda env create -f environment.yml
conda activate evolvepro
```

### Protein Language Models Environment

For installing all underlying protein language models, we use a different environment:

```bash
sh setup_plm.sh
conda activate plm
```

This environment includes:

- Deep learning frameworks (PyTorch)
- Protein language models that are installable via pip (ESM, ProtT5, UniRep, ankh, unirep)
- Protein language models that are only installable from github environments (proteinbert, efficient-evolution)

These environments are kept separate to maintain clean dependencies and avoid conflicts between the core EVOLVEpro functionality and the various protein language models.

## Examples

## Issues

## Citation

If you use this code in your research, please cite the original paper:

```
@ARTICLE
author={Jiang, Kaiyi and Yan, Zhaoqing and Di Bernardo, Matteo and Sgrizzi, Samantha R. and Villiger, Lukas and Kayabölen, Alişan and Kim, Byungji and Carscadden, Josephine K. and Hiraizumi, Masahiro and Nishimasu, Hiroshi and Gootenberg, Jonathan S. and Abudayyeh, Omar O.}
title={Rapid in silico directed evolution by a protein language model with EVOLVEpro}, 
year={2024},
DOI={10.1126/science.adr6006}
```