import pathlib
import torch
from Bio import SeqIO


def get_device(nogpu=False):
    if nogpu:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_fasta(fasta_file: pathlib.Path) -> list[tuple[str, str]]:
    """Parse a FASTA file into a list of (label, sequence) tuples."""
    return [(record.id, str(record.seq)) for record in SeqIO.parse(handle=fasta_file,
                                                                   format="fasta")]
