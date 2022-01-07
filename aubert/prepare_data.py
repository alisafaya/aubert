"""
Data preparation for pre-training AuBERT.
"""

import os
import json
import logging
import argparse
import sys
import itertools
import random
from multiprocessing import Pool, Manager, Lock
from tqdm import tqdm

from nltk.util import ngrams
from functools import partial

import soundfile as sf
import numpy as np
import torch

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

logger.addHandler(handler)

tokenizer = {}
processor = {}

def get_speaker_dict(books, min_utterances=100):

    speakers = {}
    raw_books = {}
    for book in books:
        raw_books[book['book_name']] = book['raw']
        for utterance in book['utterances']:
            if utterance['similarity'] == 0 or 'words' not in utterance:
                continue

            if utterance['speaker_id'] not in speakers:
                speakers[utterance['speaker_id']] = []

            utt = dict.copy(utterance)
            utt.pop('speaker_id')
            utt.pop('similarity')

            utt['book'] = book['book_name']
            speakers[utterance['speaker_id']].append(utt)

    deleted_speakers = []
    for speaker_id, utterances in list(speakers.items()):
        if len(utterances) < min_utterances:
            del speakers[speaker_id]
            deleted_speakers.append((speaker_id, len(utterances)))

    return speakers, raw_books, deleted_speakers


def prepare_text_input(texts, args):

    global tokenizer
    if not tokenizer:
        logger.info("Loading tokenizer: %s" % args.tokenizer)
        from transformers import BertTokenizerFast
        tokenizer = BertTokenizerFast.from_pretrained(args.tokenizer)

    mask_ratio = args.mask_ratio

    def mask(text, span_start, span_end):

        pre_tokenized = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
    
        char_offset_map = np.zeros(len(text), dtype=np.int16)
        for i, (_, x) in enumerate(pre_tokenized):
            char_offset_map[x[0]:x[1]] = i + 1

        char_offset_map[char_offset_map == 0] = char_offset_map[np.minimum(np.where(char_offset_map == 0)[0] + 1, len(char_offset_map) - 1)]
        char_offset_map -= 1

        masked_span_start = char_offset_map[span_start]
        masked_span_end = char_offset_map[span_end]

        words = [ x[0] for x in pre_tokenized ]

        # word to [wo, ##rd] map
        offset_map = np.cumsum([ len(tokenizer.tokenize(word)) for word in words ])
        offset_map = np.array(list(ngrams(np.insert(offset_map, 0, 0), 2)))

        mask_indices = np.random.choice(len(words), int(len(words) * mask_ratio), replace=False)
        mask_indices = sorted(mask_indices)

        tokens_to_mask = list(itertools.chain(*[ range(*s) for s in offset_map[mask_indices] ]))

        return words, tokens_to_mask, (int(offset_map[masked_span_start][0] + 1), int(offset_map[masked_span_end][0] + 1)) # +1 to account for [CLS]

    words, tokens_to_mask, span_positions = list(zip(*[ mask(*x) for x in texts ]))

    encoded = tokenizer(list(words), is_split_into_words=True, add_special_tokens=True, return_tensors="np", max_length=args.max_seq_length,
                        truncation=True, padding=True)

    encoded['labels'] = encoded['input_ids'].copy()
    for batch, m in enumerate(tokens_to_mask):
        mask_indices = np.array(m) + 1
        encoded['input_ids'][batch][mask_indices[mask_indices < (args.max_seq_length - 2) ]] = tokenizer.mask_token_id # +1 to account for [CLS]

    for k in list(encoded.keys()):
        if k in ("input_ids", "labels"):
            encoded[k] = encoded[k].astype(np.int16)

        elif k in ("attention_mask", "token_type_ids"):
            encoded[k] = encoded[k].astype(np.bool8)

    return encoded, span_positions


def prepare_audio_input(audios, args):

    global processor
    if not processor:
        logger.info("Loading audio processor: %s" % args.audio_processor)
        from transformers import Wav2Vec2Processor
        processor = Wav2Vec2Processor.from_pretrained(args.audio_processor)

    encoded = processor(audios, return_tensors="np", sampling_rate=16000, padding='longest')
    encoded['attention_mask'] = encoded["attention_mask"].astype(np.bool8)

    return encoded


def prepare_speaker_ngram_data(instances, args):

    ngrams, contexts, audios = list(zip(*instances))
    text_encoded, span_indices = prepare_text_input(contexts, args)
    audio_encoded = prepare_audio_input(audios, args)

    return { 'text': text_encoded, 'audio': audio_encoded, 'ngrams': ngrams[0], 'spans': span_indices }


def get_audio(cache_dict, cache_queue, audio_fname):
    if audio_fname in cache_dict:
        return cache_dict[audio_fname]

    if len(cache_queue) > 4096:
        cache_dict.pop(cache_queue.pop(0))

    content = sf.read(audio_fname)[0]
    cache_dict[audio_fname] = content
    cache_queue.append(audio_fname)

    return content


def collate_single_ngram(tup, speaker_data, raw_books, args, cache_dict, cache_queue):
    ngram, indices = tup

    if len(indices) > args.max_samples_per_ngram:
        random.shuffle(indices)
        indices = indices[:args.max_samples_per_ngram]

    instances = []
    for i, j, ngram_size in indices:
        alignments = speaker_data[i]['words']
        _, start_idx, _, audio_starts, _ = alignments[j]
        _, _, end_idx, _, audio_ends = alignments[j + ngram_size - 1]

        if (audio_ends - audio_starts) < args.min_audio_length:
            continue

        audio_data = get_audio(cache_dict, cache_queue, speaker_data[i]['fname'].split("/")[-1])
        relative_start_idx = start_idx - max(0, start_idx - 512)
        relative_end_idx = end_idx - max(0, start_idx - 512)

        context_window = (raw_books[speaker_data[i]['book']][max(0, start_idx - 512):end_idx + 512], relative_start_idx, relative_end_idx)
        audio_segment = audio_data[audio_starts * 16:audio_ends * 16] # assume 16kHz sampling rate

        instances.append((ngram, context_window, audio_segment))

        # DEBUG
        # if i < 3:
            # sf.write(os.path.join("bin/temp-audio", "%s_%d.wav" % (ngram, i)), audio_segment, 16000)

    if instances:
        return prepare_speaker_ngram_data(instances, args)
    else:
        return []

def collate_speaker_data(speaker_data, raw_books, args, pool, speaker_id, cache_dict, cache_queue):

    speaker_vocab_map = {}
    for i, utterance in enumerate(speaker_data):
        if utterance.get('words', False):
            for ngram_size in range(args.max_ngram_size):
                for j, ngram in enumerate(ngrams(utterance['words'], ngram_size)):
                    ngram = " ".join(x[0] for x in ngram)
                    if ngram not in speaker_vocab_map:
                        speaker_vocab_map[ngram] = []

                    speaker_vocab_map[ngram].append((i, j, ngram_size))

    speaker_vocab_map = { k: v for k, v in speaker_vocab_map.items() if len(v) >= args.min_ngram_occurrences }
    speaker_vocab_map = list(speaker_vocab_map.items())
    random.shuffle(speaker_vocab_map)

    logger.info("Found %d vocabulary for speaker %s" % (len(speaker_vocab_map), speaker_id))

    if len(speaker_vocab_map) > args.max_ngrams_per_speaker:
        speaker_vocab_map = speaker_vocab_map[:args.max_ngrams_per_speaker]

    data_instances = []
    logger.info("Processing speaker's data...")
    for ngram_data in tqdm(pool.imap_unordered(partial(collate_single_ngram, speaker_data=speaker_data, raw_books=raw_books, args=args, cache_dict=cache_dict, cache_queue=cache_queue), speaker_vocab_map), 
                            total=len(speaker_vocab_map)):
        if ngram_data:
            data_instances.append(ngram_data)
        
        if len(data_instances) == args.chunk_size:
            yield data_instances
            data_instances = []

    if len(data_instances) > 0:
        yield data_instances

    return data_instances


if __name__ == '__main__':

    random.seed(42)
    np.random.seed(42)

    logger.info('-' * 100)
    logger.info('Starting data preparation')
    logger.info('-' * 100)

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--alignments', type=str, help='Directory to The JSON files for word-level aligned book data as prepared in the preprocessing step')
    argparser.add_argument('--audio-dir', type=str, help='Directory to the audio files')
    argparser.add_argument('--audio-processor', type=str, default="facebook/hubert-large-ls960-ft", help='The audio processor to use for preprocessing and padding audio segments')
    argparser.add_argument('--chunk-size', type=int, default=512, help='The max samples to save at a data batch')
    argparser.add_argument('--max-seq-length', type=int, default=256, help='The maximum sequence length of utterance contexts')
    argparser.add_argument('--mask-ratio', type=float, default=0.1, help='The ratio of masked tokens in the input sequence')
    argparser.add_argument('--max-ngram-size', type=int, default=4, help='The size of n-grams to use for pre-training AuBERT')
    argparser.add_argument('--min-utter-per-speaker', type=int, default=100, help='The minimum number of utterances per speaker to include in the dataset')
    argparser.add_argument('--min-ngram-occurrences', type=int, default=10, help='The minimum number of times a n-gram must occur in the dataset to be included')
    argparser.add_argument('--min-audio-length', type=int, default=400, help='The minimum length of audio segments to include in the dataset')
    argparser.add_argument('--max-samples-per-ngram', type=int, default=128, help='The maximum number of samples per n-gram to include in the dataset')
    argparser.add_argument('--max-ngrams-per-speaker', type=int, default=64, help='The maximum number of samples per n-gram to include in the dataset')
    argparser.add_argument('--num-workers', type=int, default=8, help='The number of workers to use for data loading')
    argparser.add_argument('--output-dir', type=str, help='Directory to save the output')
    argparser.add_argument('--tokenizer', type=str, default="bert-base-uncased", help='The tokenizer to use for tokenization')
    argparser.add_argument('--rank', type=int, default=0, help='Rank of the current process')
    argparser.add_argument('--world_size', type=int, default=1, help='World size: total nodes')

    args = argparser.parse_args()

    logger.info("Loading alignments...")
    books = [ json.load(open(os.path.join(args.alignments, book))) for book in os.listdir(args.alignments) ]
    logger.info("Loaded {} books".format(len(books)))

    logger.info("Fetching speaker data...")
    speakers, raw_books, deleted_speakers = get_speaker_dict(books, min_utterances=args.min_utter_per_speaker)
    logger.info("Found %d speakers, %d were deleted due to having less than %d utterances" % (len(speakers), len(deleted_speakers), args.min_utter_per_speaker))
    logger.info("Total ignored utterances: %d" % sum([utterance_count for _, utterance_count in deleted_speakers]))

    speakers = list(speakers.items())
    speakers = sorted(speakers, key=lambda x: x[0])
    random.shuffle(speakers)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    num_speakers = len(speakers)
    speakers_per_node = (len(speakers)  + 1) // args.world_size
    speakers = speakers[args.rank * speakers_per_node: (args.rank + 1) *  speakers_per_node]

    logger.info("Processing speakers: [ %s ]" % (",".join([x[0] for x in speakers ])))

    # Create the pool of workers
    with Pool(args.num_workers) as pool:
        total_vocab, total_samples = 0, 0
        for speaker, data in speakers:
            audio_dict, audio_cache = {}, []
            speaker_data = collate_speaker_data(data, raw_books, args, pool, speaker, audio_dict, audio_cache)
            for i, chunk in enumerate(speaker_data):
                torch.save(chunk, os.path.join(args.output_dir, speaker + f".{i}.pt"))
                logger.info("Saved chunk %d for speaker %s" % (i, speaker))
                total_vocab += len(chunk)
                total_samples += sum([len(x['spans']) for x in chunk])

    logger.info("Total vocab size: %d" % total_vocab)
    logger.info("Average vocab size per speaker: %d" % (total_vocab / len(speakers)))
    logger.info("Total samples: %d" % total_samples)
    logger.info("Average samples per speaker: %d" % (total_samples / len(speakers)))
    logger.info("Average samples per vocab: %d" % (total_samples / total_vocab))
    logger.info("Done!")
