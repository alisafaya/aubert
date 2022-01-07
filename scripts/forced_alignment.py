from aligner import MfaAligner
from pathlib import Path
from glob import glob
import os
import json
from tqdm import tqdm
from argparse import ArgumentParser
import argparse
import numpy as np
import tokenizers
from sacremoses import MosesPunctNormalizer
import string


def main(args, num_chunks, chunk_id):
    input_dir = args.input_dir
    work_dir = args.work_dir
    num_jobs = args.num_jobs
    max_retries = args.max_retries

    print(f"Chunk id: {chunk_id}, Num Chunks: {num_chunks}")

    audiofiles, texts, prefixes = [], [], []
    alignments = glob(f"{input_dir}/*")

    for alignment in tqdm(alignments, desc="Processing alignment files"):
        alignment_jsn = json.load(open(alignment))
        for utterance in tqdm(alignment_jsn["utterances"], desc="Processing utterances", leave=False):
            transcript = alignment_jsn["raw"][utterance["start_idx"]:utterance["end_idx"]]
            audiofl = utterance["fname"]
            prefix = utterance["speaker_id"]
            texts.append(transcript)
            audiofiles.append(audiofl)
            prefixes.append(prefix)

    texts = np.array_split(texts, num_chunks)[chunk_id]
    audiofiles = np.array_split(audiofiles, num_chunks)[chunk_id]
    prefixes = np.array_split(prefixes, num_chunks)[chunk_id]


    punct_norm = MosesPunctNormalizer()
    tokenizer = tokenizers.pre_tokenizers.WhitespaceSplit()
    tokenize = lambda inp : [punct_norm.normalize(x[0]).upper().translate(str.maketrans("","",string.punctuation)) for x in tokenizer.pre_tokenize_str(inp)]

    output_path = os.path.join(work_dir, f"chunk_{chunk_id}")

    alner = MfaAligner()
    success = alner.align_batch(texts, audiofiles, workdir=output_path, prefixes=prefixes, tokenizer=tokenize, overwrite=True, validate=False, numjobs=num_jobs)

    if not success:
        for retry in range(max_retries):
            success = alner.align_batch(texts, audiofiles, workdir=output_path, prefixes=prefixes, tokenizer=tokenize, skipprep=True, overwrite=True, validate=False, numjobs=num_jobs)
            if success:
                break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Perform forced alignments of segments on the word level using the Montreal Forced Aligner.')
    parser.add_argument('--input_dir', default="alignments", type=str, help="Input directory which contains the aligned segment json files")
    parser.add_argument('--work_dir', default="data/forced_alignments", type=str, help='Output directory where forced word alignments will be written to, default: data/forced_alignments')
    parser.add_argument('--num_chunks', default=1, type=int, help="Number of chunks to use.")
    parser.add_argument('--chunk_id', default=0, type=int, help="ID of current chunk.")
    parser.add_argument('--num_jobs', default=5, type=int, help="Number of chunks to use.")
    parser.add_argument('--max_retries', default=2, type=int, help="Maximum number of retries. Default: 2")

    args = parser.parse_args()
    num_chunks = args.num_chunks
    chunk_id = args.chunk_id

    main(args, num_chunks, chunk_id)