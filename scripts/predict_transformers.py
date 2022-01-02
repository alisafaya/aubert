import torch
from transformers import Wav2Vec2Processor, HubertForCTC
import soundfile as sf
from glob import glob
from multiprocessing import Pool
import argparse
import os
from tqdm import tqdm
import numpy as np
import sys
import time
import itertools
from pyctcdecode import build_ctcdecoder
import tracemalloc

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(threadName)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

logger.addHandler(handler)

def load_audio(file):
    speech, _ = sf.read(file)
    return speech.astype(np.float32), file

def grouper_it(iterable, n):
    it = iter(iterable)
    while True:
        chunk_it = itertools.islice(it, n)
        try:
            first_el = next(chunk_it)
        except StopIteration:
            return

        yield itertools.chain((first_el,), chunk_it)


if __name__ == '__main__':
 
    # starting the monitoring
    tracemalloc.start()

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--input-dir', type=str, help='Directory containing the dataset')
    argparser.add_argument('--output-dir', type=str, help='Directory to save the output')
    argparser.add_argument('--num-processes', type=int, default=4, help='Number of processes for decoder')
    argparser.add_argument('--rank', type=int, default=0, help='Rank of the current process')
    argparser.add_argument('--num-speakers', type=int, default=200, help='Number of speakers per process')
    argparser.add_argument('--lm-path', type=str, help='Path to the language model')
    argparser.add_argument('--batch-size', type=int, default=8, help='Batch size for decoding')
    argparser.add_argument('--device', type=str, default='cuda', help='Device to use for decoding')

    args = argparser.parse_args()

    device = torch.device(args.device)

    # Working on num_speakers speakers per process
    speakers = sorted(os.listdir(args.input_dir))
    num_speakers = len(speakers)
    print(f"Found {num_speakers} speakers in total")
    speaker_list = speakers[args.rank *  args.num_speakers: (args.rank + 1) *  args.num_speakers]

    # Create the output directory
    output_dir = os.path.join(args.output_dir, f"transcriptions_{args.rank}.tsv")

    # Create the dataset by traversing the input directory
    file_names = []
    for speaker in speaker_list:
        file_names.extend(glob(os.path.join(args.input_dir, speaker, "**/*.flac"), recursive=True))

    print(f"Found {len(file_names)} files")

    # Create the decoder
    processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")
    model = HubertForCTC.from_pretrained("facebook/hubert-large-ls960-ft")
    model.to(device)

    decoder = build_ctcdecoder(
        list(processor.tokenizer.get_vocab()),
        args.lm_path,
        alpha=0.5,   
        beta=1.0,
    )

    # Create the pool of workers
    decoding_pool = Pool(args.num_processes)

    # Decode the dataset
    with open(output_dir, 'w') as f:
        reading_pool = Pool(args.num_processes)
        
        # Create the iterator for reading the dataset
        with tqdm(grouper_it(file_names, args.batch_size), total=len(file_names) // args.batch_size) as pbar:

            for fnames in pbar:
                chunk = reading_pool.map(load_audio, fnames)
                audios, files = zip(*chunk)
                batch = processor(audios, padding='longest', max_length=1600_000, truncation=True, pad_to_multiple_of=8, sampling_rate=16000, return_tensors='pt').input_values
                with torch.cuda.amp.autocast():
                    with torch.no_grad():
                        logits = model(batch.to(device)).logits.cpu().numpy()

                textlist = decoder.decode_batch(decoding_pool, logits, beam_width=20)
                for file_name, text in zip(files, textlist):
                    print(f"{file_name}\t{text}", file=f)

                pbar.set_postfix(mem_usage=tracemalloc.get_traced_memory()[1])

    # stopping the library
    tracemalloc.stop()
