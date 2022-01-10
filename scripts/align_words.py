from merge_word_alignments_to_annotations import merge_word_alignments
from forced_alignment import perform_forced_alignment
import multiprocessing
from tqdm import tqdm
from functools import partial
import argparse
from argparse import ArgumentParser

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Perform forced alignments of segments on the word level using the Montreal Forced Aligner and add the word alignments to the annotation json files.')
    parser.add_argument('-i','--input_dir', default="alignments", type=str, help="Input directory which contains the aligned segment json files, default: alignments")
    parser.add_argument('--work_dir', default="data/forced_alignments", type=str, help='Output directory where forced word alignments will be written to, default: data/forced_alignments')
    parser.add_argument('-o', '--output_dir', default="data/aligned", type=str, help='Output directory where word aligned json files will be written to, default: data/aligned')
    parser.add_argument('--num_chunks', default=1, type=int, help="Number of chunks to divide the workload to.")
    parser.add_argument('--num_threads', default=1, type=int, help="Number of threads to use.")
    parser.add_argument('--num_jobs', default=5, type=int, help="Number of jobs to use for MFA.")
    parser.add_argument('--max_retries', default=2, type=int, help="Maximum number of retries if MFA fails. Default: 2")
    parser.add_argument('--verbose', action="store_true", help="Enable logging")

    args = parser.parse_args()
    num_chunks = args.num_chunks
    workers = args.num_threads

    # Start the alignment process
    forced_alignment_partial = partial(perform_forced_alignment, args, num_chunks)

    with multiprocessing.Pool(workers) as pool:

    	for job in tqdm(pool.imap_unordered(forced_alignment_partial, range(num_chunks))):
    		pass


   	# Merge the alignments to the final output folder
    merge_partial = partial(merge_word_alignments, args, num_chunks)

    with multiprocessing.Pool(workers) as pool:

    	for job in tqdm(pool.imap_unordered(merge_partial, range(num_chunks))):
    		pass
