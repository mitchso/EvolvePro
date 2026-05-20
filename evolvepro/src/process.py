import os
from itertools import combinations
from typing import Callable, List, Optional, Tuple
import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

aa_alphabet = 'ACDEFGHIKLMNPQRSTVWY'

def _get_mutant_position(mutation: str, index=1) -> int:
    position = int(mutation[1:-1])
    if index == 1:
        return position
    elif index == 0:
        return position - 1


def generate_wt(wt_sequence: str, output_file: str) -> SeqRecord:
    """Write a single-record FASTA file containing the wild-type sequence."""
    record = SeqRecord(Seq(wt_sequence), id='WT', description='')
    with open(output_file, 'w') as f:
        SeqIO.write(record, f, 'fasta')
    return record


def generate_single_aa_mutants(wt_sequence: str, output_file: str) -> list[SeqRecord]:
    """Generate all possible single amino-acid substitutions and save as FASTA."""

    records = []

    for i in range(len(wt_sequence)):
        wt_aa = wt_sequence[i]
        for mut_aa in aa_alphabet:
            if mut_aa == wt_aa:
                continue
            mutant_seq = wt_sequence[:i] + mut_aa + wt_sequence[i + 1:]
            variant = f'{wt_aa}{i + 1}{mut_aa}'
            records.append(SeqRecord(Seq(mutant_seq), id=variant, description=''))

    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)

    with open(output_file, 'w') as f:
        SeqIO.write(records, f, 'fasta')

    return records


def generate_multi_mutants(wt_sequence: str, output_file: str, min_mutations: int, max_mutations: int, mutations_list: list[str]) -> list[SeqRecord]:
    """Given the mutations in mutations_list, generate all possible combinatorial mutants.
    min_mutations sets a lower bound on how many mutations must exist in each sequence
    max_mutations sets an upper bound on how many mutations must exist in each sequence"""

    records = []

    if len(mutations_list) == 0:
        print("You must provide specific mutations to combine. Exiting.")
        return

    for mutation in mutations_list:
        # check to make sure the wt position is correct
        mutation_pos = _get_mutant_position(mutation, index=0)
        wt_aa_original = wt_sequence[mutation_pos]
        wt_aa_mutated = mutation[0]
        if wt_aa_mutated != wt_aa_original:
            print(f"{mutation} is given but the original sequence is {wt_aa_original} at that position. Exiting.")
            return

    for n in range(min_mutations, max_mutations + 1):
        for combo in combinations(mutations_list, n):
            # Check for position conflicts (e.g. A42G and A42V in the same combo)
            positions = [_get_mutant_position(m) for m in combo]
            if len(positions) != len(set(positions)):
                continue

            # Apply all mutations in the combo to the WT sequence
            mutant_seq = list(wt_sequence)
            for mutation in combo:
                pos = _get_mutant_position(mutation, index=0)  # convert to 0-indexed
                mut_aa = mutation[-1]
                mutant_seq[pos] = mut_aa

            variant_id = '_'.join(combo)
            records.append(SeqRecord(Seq(''.join(mutant_seq)), id=variant_id, description=''))

    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)

    with open(output_file, 'w') as f:
        SeqIO.write(records, f, 'fasta')

    return records

