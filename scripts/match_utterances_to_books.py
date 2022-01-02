"""
Match utterances to books using fuzzy substring matching.
"""
import argparse 
import os
import json
import string
import re

from tqdm import tqdm
from multiprocessing import Pool

from cdifflib import CSequenceMatcher as SM
import difflib
difflib.SequenceMatcher = SM

from nltk.util import ngrams
import numpy as np

regex = re.compile("|".join(re.escape(x) for x in string.punctuation.replace("'", ""))) # keep the appostrophes in the book

pass_keywords = [ "LIBREVOX", "RECORDING", ]

def get_match(args):

    book_name, utterances = args

    book = open(book_name).read()
    book_raw = re.sub(r'\s+', " ", book)
    processed = re.sub(regex, "", book_raw.lower())
    restore = [ (m.start(), m.end(), m.group(0)) for m in re.finditer(regex, book_raw) ]

    ## aligning the book before and after processing
    i = 0
    index_map = np.zeros(len(book_raw), dtype=np.int64)
    for k, (s, _, _) in enumerate(restore):
        index_map[i:s-k] = k 
        i = (s - k)
    index_map[i:] = k
    ##

    for utterance in utterances:

        query = utterance["utterance"]
        query_length = len(query.split())

        max_sim_string = ""
        counter, max_sim_val = 0, 0
        for i, ngram in enumerate(ngrams(processed.split(), query_length)):

            if counter > 0:
                if i - counter > 10:
                    break
            elif i % (query_length // 4) != 0:
                continue

            book_ngram = " ".join(ngram)
            similarity = SM(lambda x: x in string.punctuation, book_ngram, query).ratio() 

            if similarity > max_sim_val and similarity > 0.3:
                max_sim_val = similarity
                max_sim_string = book_ngram
                counter = i

        startidx = processed.find(max_sim_string)
        endidx = startidx + len(max_sim_string)
        startidx += index_map[startidx]
        endidx += index_map[endidx]

        # utterance["matched_processed"] = max_sim_string
        # utterance["matched_book_raw"] = book_raw[startidx:endidx]

        utterance["start_idx"] = int(startidx)
        utterance["end_idx"] = int(endidx)
        utterance["similarity"] = max_sim_val

    return book_name.split("/")[-1], utterances, book_raw


def get_book_name(audio_dir, book_meta):
    """
    Get the book name from the directory name.

    Audio directory names are of the form: ../temp-out-librivox/1053/778/lilacfairybook_03_lang_64kb_0009.flac
    Book meta is a dictionary of the form: {  
        "778": {
            "url": "http://www.gutenberg.org/etext/3454",
            "name": "lilac_fairy_0707_librivox_64kb_mp3.txt"},
        }
        ..
    }
    """
    try: 
        return book_meta[audio_dir.split("/")[-2]]["name"], audio_dir.split("/")[-3]
    except KeyError:
        return ""


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()

    argparser.add_argument('-u', '--utterances', type=str, help='Directory containing the utterances to match (.tsv file)')
    argparser.add_argument('-b', '--books-dir', type=str, help='Directory containing the books to match')
    argparser.add_argument('-m', '--book-meta', type=str, help='Directory containing the metadata for the books')
    argparser.add_argument('-p', '--num-processes', type=int, default=8, help='Number of processes for decoder')
    argparser.add_argument('-o', '--output-dir', type=str, help='Directory to save the output')
    
    args = argparser.parse_args()

    # Create the output directory
    output_dir = args.output_dir

    # Create the book list
    books = sorted(os.listdir(args.books_dir))
    book_meta = json.load(open(args.book_meta))

    # Create the utterance list
    utterances = {}
    with open(args.utterances, 'r') as f:
        for line in f:
            # check for pass keywords
            if any(keyword in line for keyword in pass_keywords):
                continue
            try:
                fname, utterance = line.strip().split("\t")
            except ValueError:
                # print("Error reading line : {}".format(line.strip()))
                continue

            book_name, speaker_id = get_book_name(fname, book_meta)
            if book_name not in books:
                continue

            book_name = os.path.join(args.books_dir, book_name)
            if book_name not in utterances:
                utterances[book_name] = []

            utterances[book_name].append({ "fname": fname, "utterance": utterance.lower(), "speaker_id": speaker_id })

    print("Found {} books and {} utterances".format(len(utterances), sum(len(v) for v in utterances.values())))

    # Create the pool of workers
    pool = Pool(args.num_processes)

    # Match the utterances to the books
    for book_name, match_list, book_raw in tqdm(pool.imap(get_match, utterances.items()), total=len(utterances)):
        with open(os.path.join(output_dir, f"{book_name}.align.json"), "w") as f:
            json.dump({ "book_name": book_name, "utterances": match_list, "raw": book_raw }, f, ensure_ascii=False, indent=2)
