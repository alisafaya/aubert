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

from nltk import tokenize
from tqdm import tqdm

from nltk.util import ngrams
from nltk.tokenize import word_tokenize
from transformers import BertTokenizerFast, Wav2Vec2Processor

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

tokenizer = ""
processor = ""

def get_speaker_dict(books, min_utterances=100):

    speakers = {}
    raw_books = {}
    for book in books:
        raw_books[book['book_name']] = book['raw']
        for utterance in book['utterances']:
            if utterance['similarity'] == 0:
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
        tokenizer = BertTokenizerFast.from_pretrained(args.tokenizer)

    mask_ratio = args.mask_ratio

    def mask(text):
        words = np.array(word_tokenize(text))

        offset_map = np.cumsum([ len(tokenizer.tokenize(word)) for word in words ])
        offset_map = np.array(list(ngrams(np.insert(offset_map, 0, 0), 2)))

        mask_indices = np.random.choice(len(words), int(len(words) * mask_ratio), replace=False)
        mask_indices = sorted(mask_indices)

        tokens_to_mask = list(itertools.chain(*[ range(*s) for s in offset_map[mask_indices] ]))
        return words.tolist(), tokens_to_mask

    words, tokens_to_mask = list(zip(*[ mask(x) for x in texts ]))

    encoded = tokenizer(list(words), is_split_into_words=True, add_special_tokens=True, return_tensors="pt", max_length=args.max_seq_length,
                        truncation=True, padding=True)

    encoded['labels'] = encoded['input_ids'].clone()
    for i, mask in enumerate(tokens_to_mask):
        mask_indices = np.array(mask) + 1
        encoded['input_ids'][i][mask_indices[mask_indices < (args.max_seq_length - 2) ]] = tokenizer.mask_token_id # +1 to account for [CLS]

    # TODO: 
    # return span index
    encoded['span_index'] = "???"

    return encoded


def prepare_audio_input(audios, args):

    global processor
    if not processor:
        logger.info("Loading audio processor: %s" % args.audio_processor)
        processor = Wav2Vec2Processor.from_pretrained(args.audio_processor)

    encoded = processor(audios, return_tensors="pt", sampling_rate=16000, padding='longest')

    return encoded


def prepare_speaker_ngram_data(instances, args):

    ngrams, contexts, audios = list(zip(*instances))

    text_encoded = prepare_text_input(contexts, args)
    audio_encoded = prepare_audio_input(audios, args)

    return { 'text': text_encoded, 'audio': audio_encoded, 'ngrams': ngrams }


def collate_speaker_data(speaker_data, raw_books, args):

    audio_data, speaker_vocab_map = {}, {}
    for i, utterance in enumerate(speaker_data):
        if utterance.get('words', False):
            for ngram_size in range(args.max_ngram_size):
                for j, ngram in enumerate(ngrams(utterance['words'], ngram_size)):
                    ngram = " ".join(x[0] for x in ngram)
                    if ngram not in speaker_vocab_map:
                        speaker_vocab_map[ngram] = []

                    speaker_vocab_map[ngram].append((i, j, ngram_size))

            audio_data[i] = sf.read(os.path.join(args.audio_dir, utterance['fname'].split("/")[-1]))[0]

    speaker_vocab_map = { k: v for k, v in speaker_vocab_map.items() if len(v) >= args.min_ngram_occurrences }

    data_instances = []
    for ngram, indices in tqdm(speaker_vocab_map.items()):
        if len(indices) > args.max_samples_per_ngram:
            random.shuffle(indices)
            indices = indices[:args.max_samples_per_ngram]

        instances = []
        for i, j, ngram_size in indices:
            alignments = speaker_data[i]['words']
            _, start_idx, _, audio_starts, _ = alignments[j]
            _, _, end_idx, _, audio_ends = alignments[j + ngram_size - 1]

            context_window = raw_books[speaker_data[i]['book']][max(0, start_idx - 1024):end_idx + 1024]
            audio_segment = audio_data[i][audio_starts * 16:audio_ends * 16] # assume 16kHz sampling rate

            instances.append((ngram, context_window, audio_segment))

        data_instances.append(prepare_speaker_ngram_data(instances, args))

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
    argparser.add_argument('--max-seq-length', type=int, default=512, help='The maximum sequence length of utterance contexts')
    argparser.add_argument('--mask-ratio', type=float, default=0.1, help='The ratio of masked tokens in the input sequence')
    argparser.add_argument('--max-ngram-size', type=int, default=3, help='The size of n-grams to use for pre-training AuBERT')
    argparser.add_argument('--min-utter-per-speaker', type=int, default=100, help='The minimum number of utterances per speaker to include in the dataset')
    argparser.add_argument('--min-ngram-occurrences', type=int, default=10, help='The minimum number of times a n-gram must occur in the dataset to be included')
    argparser.add_argument('--max-samples-per-ngram', type=int, default=256, help='The maximum number of samples per n-gram to include in the dataset')
    argparser.add_argument('--output-dir', type=str, help='Directory to save the output')
    argparser.add_argument('--tokenizer', type=str, default="bert-base-uncased", help='The tokenizer to use for tokenization')
    args = argparser.parse_args()

    logger.info("Loading alignments...")
    books = [ json.load(open(os.path.join(args.alignments, book))) for book in os.listdir(args.alignments) ]
    logger.info("Loaded {} books".format(len(books)))

    logger.info("Fetching speaker data...")
    speakers, raw_books, deleted_speakers = get_speaker_dict(books, min_utterances=args.min_utter_per_speaker)
    logger.info("Found %d speakers, %d were deleted due to having less than %d utterances" % (len(speakers), len(deleted_speakers), args.min_utter_per_speaker))
    logger.info("Total ignored utterances: %d" % sum([utterance_count for _, utterance_count in deleted_speakers]))

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    for speaker in speakers:
        speaker_data = collate_speaker_data(speakers[speaker], raw_books, args)
        logger.info("Speaker vocabulary size: {}".format(len(speaker_data)))
        logger.info("Total size of speaker data: {}".format(sum([len(x['text']) for x in speaker_data])))
        logger.info("Saving speaker data to {}".format(os.path.join(args.output_dir, speaker + ".pt")))

        torch.save(speaker_data, os.path.join(args.output_dir, speaker + ".pt"))
