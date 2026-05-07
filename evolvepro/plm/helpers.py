from collections import defaultdict
from itertools import combinations, product
import re


def get_mutation_combinations(mutations: list[str], n: int) -> list[tuple[str, ...]]:
    """
    Generate all valid combinations of n mutations from a list, ensuring that
    mutations at the same position are never combined together.

    Args:
        mutations: List of mutations in [WT AA][position][Mutant AA] format (e.g., ["A107L", "A107M", "K4Y"])
        n: Number of mutations to combine together

    Returns:
        List of tuples, each containing a valid combination of n mutations
    """

    def parse_position(mutation: str) -> int:
        match = re.match(r'^[A-Za-z](\d+)[A-Za-z]$', mutation)
        if not match:
            raise ValueError(f"Invalid mutation format '{mutation}'.")
        return int(match.group(1))

    position_groups: dict[int, list[str]] = defaultdict(list)
    for mut in mutations:
        position_groups[parse_position(mut)].append(mut)

    unique_positions = sorted(position_groups.keys())
    n_positions = len(unique_positions)

    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}")
    if n > n_positions:
        raise ValueError(f"Cannot combine {n} mutations: only {n_positions} unique position(s) available")

    multi = {p: position_groups[p] for p in unique_positions if len(position_groups[p]) > 1}
    single_count = n_positions - len(multi)

    valid_combinations = []
    for pos_combo in combinations(unique_positions, n):
        for combo in product(*[position_groups[p] for p in pos_combo]):
            valid_combinations.append(combo)

    # --- Condensed summary ---
    print(f"Mutations: {len(mutations)} across {n_positions} positions | Combining: {n} at a time")
    if single_count:
        print(f"  {single_count} position(s) with 1 variant")
    for pos, muts in multi.items():
        print(f"  Position {pos}: {len(muts)} variants ({', '.join(muts)}) — never combined together")
    print(f"Total valid combinations: {len(valid_combinations)}")

    return valid_combinations


def parse_mutation(mutation: str) -> tuple[str, int, str]:
    match = re.match(r'^([A-Za-z])(\d+)([A-Za-z])$', mutation)
    if not match:
        raise ValueError(f"Invalid mutation format '{mutation}'. Expected format: [WT AA][position][Mutant AA] (e.g., 'A1G')")
    return match.group(1).upper(), int(match.group(2)), match.group(3).upper()

def apply_mutations(sequence: str, mutations: list[str]) -> str:
    """
    Apply one or more point mutations to an amino acid sequence.

    Args:
        sequence: The wild-type amino acid sequence (e.g., "MKTAYIAK")
        mutations: List of mutations in format [WT AA][position][Mutant AA] (e.g., ["A5G", "K8R"])

    Returns:
        The mutant amino acid sequence with all mutations applied

    Raises:
        ValueError: If any mutation format is invalid, position is out of range,
                    or wild-type residue doesn't match, or duplicate positions are provided
    """
    parsed = [parse_mutation(m) for m in mutations]

    positions = [pos for _, pos, _ in parsed]
    if len(positions) != len(set(positions)):
        seen, dupes = set(), set()
        for p in positions:
            (dupes if p in seen else seen).add(p)
        raise ValueError(f"Duplicate positions found: {sorted(dupes)}")

    for wt_aa, position, _ in parsed:
        if position < 1 or position > len(sequence):
            raise ValueError(f"Position {position} is out of range for sequence of length {len(sequence)}")
        actual_aa = sequence[position - 1].upper()
        if actual_aa != wt_aa:
            raise ValueError(f"Wild-type mismatch at position {position}: expected '{wt_aa}', found '{actual_aa}'")

    mutant_sequence = list(sequence)
    for _, position, mut_aa in parsed:
        mutant_sequence[position - 1] = mut_aa

    return "".join(mutant_sequence)