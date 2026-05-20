# extract_embeddings can also be called outside of a notebook for greater efficiency during memory intensive tasks
# in this example, embeddings are pulled from ESM-2 3B by running the following command:
#       PYTHONPATH=/path/to/EvolvePro python -u extract_esm2.py

from evolvepro.plm.extract_esm_legacy import *

extract_embeddings(model='esm2_t36_3B_UR50D',
                   fasta_files=['WT.fasta', 'single_mutants.fasta'],    # make sure to include the wild-type sequence as well
                   output_csv=f'embeddings_esm2_3b.csv',
                   seqs_per_batch=8)