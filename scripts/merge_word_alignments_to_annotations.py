import textgrid
from glob import glob
from pathlib import Path
import string
from sacremoses import MosesPunctNormalizer
import tokenizers
from tqdm import tqdm
import os
import json
import argparse
from argparse import ArgumentParser
import numpy as np

punct_norm = MosesPunctNormalizer()
tokenizer = tokenizers.pre_tokenizers.WhitespaceSplit()
tokenize = lambda inp : [(punct_norm.normalize(x[0]).lower().translate(str.maketrans("","",string.punctuation)), x[1]) for x in tokenizer.pre_tokenize_str(inp)]

def get_speakerid(txt):
    return Path(txt).parts[-2]

def get_audio_name(txt):
    return Path(txt).stem

def main(args, num_chunks, chunk_id):
    inpdir = args.input_dir
    workdir = args.work_dir
    outdir = args.output_dir

    print(f"Chunk id: {chunk_id}, Num Chunks: {num_chunks}")

    dir = os.path.join(workdir, "*/outputs/*/*TextGrid")
    all_annotations = glob(dir)
    annotation_dict = {get_audio_name(x) : x for x in all_annotations}

    has_problem=0

    for jsfl in tqdm(np.array_split(os.listdir(inpdir), num_chunks)[chunk_id]):
        j = json.load(open(os.path.join(inpdir, jsfl)))
        for utt in tqdm(j["utterances"], leave=False):
            if not get_audio_name(utt["fname"])in annotation_dict:
                continue
            
            textwords = tokenize(j["raw"][utt["start_idx"]:utt['end_idx']])
            textwords = [x for x in textwords if x[0]]

            annotation_fl = annotation_dict[get_audio_name(utt["fname"])]

            tg = textgrid.TextGrid.fromFile(annotation_fl)

            gridwords = [x for x in tg[0] if x.mark]

            if not len(gridwords) == len(textwords):
                has_problem+=1
                continue

            words = []

            for text, grid in zip(textwords, gridwords):
                word = text[0]
                span_beg = text[1][0] + utt["start_idx"]
                span_end = text[1][1] + utt["start_idx"]

                audio_beg = int(grid.minTime * 1000)
                audio_end = int(grid.maxTime * 1000)

                words.append([word, span_beg, span_end, audio_beg, audio_end])

            utt['words'] = words

        outfilename = os.path.join(outdir, jsfl)
        json.dump(j, open(outfilename, "w"), indent=2)

    print(has_problem)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Refine sliding window alignments, by looking for a local maximum around the span that maximizes the Levenshtein distance between the ASR output and the span from the original book.')
    parser.add_argument('--input_dir', default="ali_annotations", type=str, help='Output directory where refined outputs will be written to, default: ali_annotations')
    parser.add_argument('--work_dir', default="data/forced_alignments", type=str, help='Output directory where forced word alignments will be written to, default: data/forced_alignments')    
    parser.add_argument('--output_dir', default="data/aligned", type=str, help='Output directory where refined outputs will be written to, default: data/aligned')
    parser.add_argument('--num_chunks', default=1, type=int, help="Number of chunks to use. Default: 1 (no chunking)")
    parser.add_argument('--chunk_id', default=0, type=int, help="ID of current chunk, if chunking is used. Default: 0 (in case of no chunking)")

    args = parser.parse_args()

    num_chunks = args.num_chunks
    chunk_id = args.chunk_id

    main(args, num_chunks, chunk_id)


