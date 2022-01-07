import Levenshtein as levy
from Levenshtein import ratio
import json
import tokenizers
import argparse
from argparse import ArgumentParser
import argparse
import os
from tqdm import tqdm
import numpy as np


whitesplit = tokenizers.pre_tokenizers.WhitespaceSplit()

def tokenize(txt, tokenizer=whitesplit, skip=1):
    """
    txt: text to be tokenized
    tokenizer: tokenizer from tokenizers library
    skip: tokens to trim from each side
    returns an array of tuples of (word, (start_idx, end_idx))
    """
    return tokenizer.pre_tokenize_str(txt)


def refine_alignment(utterance, reference, start_idx, end_idx, var_ratio=0.4, verbose=False):
    
    N = int((end_idx-start_idx)*var_ratio)
    # Measure initial
    best_lev = levy.distance(utterance, reference[start_idx:end_idx])
    tokens = tokenize(reference[max(0, start_idx-N):end_idx+N])
    # Start with original indices as the ideal indices
    best_r = end_idx-start_idx
    best_l = 0
    r_scores = []
    rs = []
    # Start changing on the right side
    for token in tokens[::-1]:
        # If right boundary less than end_idx, stop
        #         st-N      st       nd    nd+N
        # |=========|========|========|=====|==========|
        #                        ^nd-N(end) ^start
        r = token[1][1]-N

        if r<(end_idx-start_idx)-N:
            break

        span_score = levy.distance(utterance, reference[start_idx:start_idx+r])

        if span_score < best_lev:
            best_r = r
            best_lev = span_score

        if verbose:
            r_scores.append(span_score)
            rs.append(r)
    
    if verbose:
        plt.plot(rs, r_scores)
    
    # Start changing on the left side
    l_scores = []
    ls = []

    for token in tokens:
        l = token[1][0]-N
        if l>N:
            break

        span_score = levy.distance(utterance, reference[start_idx+l:start_idx+best_r])
        if span_score < best_lev:
            best_l = l
            best_lev = span_score
        if verbose:
            l_scores.append(span_score)
            ls.append(l)

    if verbose:
        plt.plot(ls, l_scores)

    return best_l, best_r



def main():
    parser = argparse.ArgumentParser(description='Refine sliding window alignments, by looking for a local maximum around the span that maximizes the Levenshtein distance between the ASR output and the span from the original book.')
    parser.add_argument('--source_dir', default="alignments/", type=str, help='Source directory, where the alignments to be refined are located, default: alignments/')
    parser.add_argument('--output_dir', default="alignments/", type=str, help='Output directory where refined outputs will be written to, default: alignments/')
    parser.add_argument('--max_length', default=5000, type=int, help="Maximum number of characters in a window to avoid processing huge chunks, default: 5000")
    parser.add_argument('--retain_all_data', action="store_true", help="Retain the old data in the input. If not selected, parts of the original data such as the old spans will be overwritten with the improved spans.")
    parser.add_argument('--ratio', default=0.4, type=float, help="Range to look in within neighborhood. For a given ratio r, spans within the range start_idx-(r*N):end_idx+(r*N) will be examined, where N is the length of the span.") 
    parser.add_argument('--num_chunks', default=1, type=int, help="Number of chunks to use.")
    parser.add_argument('--chunk_id', default=0, type=int, help="ID of current chunk.")

    args = parser.parse_args()

    original_alignments_dir = args.source_dir
    output_dir = args.output_dir
    max_length = args.max_length
    retain_data = args.retain_all_data
    max_ratio = args.ratio
    num_chunks = args.num_chunks
    chunk_id = args.chunk_id

    print(f"Chunk id: {chunk_id}, Num Chunks: {num_chunks}")

    for annotation_name in tqdm(np.array_split(os.listdir(original_alignments_dir), num_chunks)[chunk_id]):
        dct = json.load(open(os.path.join(original_alignments_dir, annotation_name)))
        
        # Some indices have 
        bad_indices = set()

        for annotation_idx in tqdm(range(len(dct["utterances"])), leave=False):
            utterance_dct = dct["utterances"][annotation_idx]
            utterance = utterance_dct["utterance"]
            
            # Check if boundaries invalid or if length more than max
            if utterance_dct["start_idx"]>=utterance_dct["end_idx"] or (utterance_dct["end_idx"]-utterance_dct["start_idx"]) > max_length:
                bad_indices.add(annotation_idx)
                continue
            

            # Refine the boundaries using levenshtein distance
            ref_l, ref_r = refine_alignment(utterance, dct["raw"], utterance_dct["start_idx"], utterance_dct["end_idx"], var_ratio=max_ratio)
            
            # Back up old info if needed
            if retain_data:
                utterance_dct["old"] = {}
                utterance_dct["old"]["start_idx"] = utterance_dct["start_idx"]
                utterance_dct["old"]["end_idx"] = utterance_dct["end_idx"] 
                utterance_dct["old"]["similarity"] = utterance_dct["similarity"]
                utterance_dct["old"]["similarity_ratio"] = ratio(utterance_dct["utterance"], dct["raw"][utterance_dct["start_idx"]:utterance_dct["end_idx"]])
            
            utterance_dct["end_idx"] = utterance_dct["start_idx"]+ref_r
            utterance_dct["start_idx"] = utterance_dct["start_idx"]+ref_l
            utterance_dct["similarity"] = ratio(utterance_dct["utterance"], dct["raw"][utterance_dct["start_idx"]:utterance_dct["end_idx"]])
            
        
        dct["utterances"] = [x for idx, x in enumerate(dct["utterances"]) if idx not in bad_indices]

        # Save modified dictionary into output folder
        json.dump(dct, open(os.path.join(output_dir, annotation_name), "w"), indent=2)

if __name__ == "__main__":
    main()