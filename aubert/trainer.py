import time
import math
import random
import logging
import sys

from glob import glob
from contextlib import contextmanager

import torch
import numpy as np
import torch.nn as nn

from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from tensorboardX import SummaryWriter

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(message)s')
handler.setFormatter(formatter)

logger.addHandler(handler)


class DummyDDPWrapper(nn.Module):
    def __init__(self, module):
        super(DummyDDPWrapper, self).__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    @contextmanager
    def join(self):
        yield


def load_states_from_checkpoint(checkpoint, model=None, optimizer=None, scaler=None, scheduler=None):
    if model is not None and "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
        logger.info('Loaded model checkpoint.')
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
        logger.info('Loaded optimizer checkpoint.')
    if scaler is not None and "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])
        logger.info('Loaded scaler checkpoint.')
    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])
        logger.info('Loaded scheduler checkpoint.')


def get_batch(ngram_data, batch_size, sample=True, device=torch.device('cuda')):

    ngram_instance_length = len(ngram_data['spans'])
    indices = np.arange(len(ngram_data['spans']))

    if sample:
        np.random.shuffle(indices)
    indices = indices[:batch_size]

    batch = {
        'spans': np.array(ngram_data['spans'])[indices],
        'ngrams': ngram_data['ngrams'],
        'audio': {
            'input_values': torch.from_numpy(ngram_data['audio']['input_values'][indices]).to(device),
            'attention_mask': torch.from_numpy(ngram_data['audio']['attention_mask'][indices]).int().to(device)
        },
        'text': {
            'input_ids': torch.from_numpy(ngram_data['text']['input_ids'][indices]).long().to(device),
            'attention_mask': torch.from_numpy(ngram_data['text']['attention_mask'][indices]).long().to(device),
            'token_type_ids': torch.from_numpy(ngram_data['text']['token_type_ids'][indices]).long().to(device),
            'labels': torch.from_numpy(ngram_data['text']['labels'][indices]).long().to(device)
        }
    }

    return batch


def evaluate(model, batch_path, device, args, global_step, writer=None):

    logger.info(f"Global Step = {global_step:0>6} | Evaluating on {batch_path}")

    model.eval()
    with torch.no_grad():
        eval_data = torch.load(batch_path)
        total_samples, total_loss = 0, 0
        for batch in eval_data:
            batch_inputs = get_batch(batch, args.batch_size, sample=False, device=device)

            # calculate loss
            with torch.cuda.amp.autocast(enabled=args.no_amp):
                loss = model(batch_inputs['text'], batch_inputs['audio'], batch_inputs['spans'])

            total_loss += loss.item()
            total_samples += 1

    total_loss /= total_samples

    if writer is not None:
        writer.add_scalar(f'{batch_path}/loss', loss, global_step)

    logger.info("Finished evaluation")
    logger.info(f'| step {global_step:0>6} | loss {total_loss:5.2f} |')

    return total_loss


def train(model, batch_path, optimizer, scaler, args, global_step=0, writer=None, device=torch.device("cuda"), lr_scheduler=None, rank=0):

    # Disabled for now.
    # static_clip=True,
    # history_size=100, # this is used only if static_clip is False 

    max_steps = args.max_steps
    log_interval = args.log_interval
    clip_value = args.clip_value

    train_data = torch.load(batch_path)
    logger.info(f"Loaded {batch_path}, containing {len(train_data)} instances.")

    model.train()
    losses, grad_history = [], []
    total_loss, minibatch = 0, 0
    with model.join(), torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=2, warmup=4, active=4, repeat=2),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('bin/log/profile'),
        record_shapes=True,
        with_stack=True) as prof:

        for batch in train_data:
            # zero out gradients from last pass, this prevents gradient accumulation!
            model.zero_grad()

            # parse batch content into torch tensors on device.
            batch_inputs = get_batch(batch, args.batch_size, sample=False, device=device)

            # calculate loss
            with torch.cuda.amp.autocast(enabled=args.no_amp):
                loss = model(batch_inputs['text'], batch_inputs['audio'], batch_inputs['spans'])

            # do backward pass
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            # calculate dynamic gradient clip threshold
            _norm = float(model.module._get_grad_norm())
            grad_norm = _norm if math.isfinite(_norm) else 0.0

            # if not static_clip:
            #     grad_history.append(grad_norm)
            #     grad_history = grad_history[-history_size:]
            #     clip_value = np.mean(grad_history)  

            # clip gradients
            if clip_value != -1:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)

            # update weights
            scaler.step(optimizer)

            # if lr scheduler is used
            if lr_scheduler is not None:
                lr_scheduler.step()

            # update scaler's scale value and other parameters
            scaler.update()
            if scaler.get_scale() > 2.0**17:
                scaler.update(2.0**18)

            # profiling
            prof.step()

            total_loss += loss.item()
            minibatch += 1
            global_step += 1

            if minibatch % log_interval == 0:
                total_loss = min(total_loss / log_interval, 1e1) # max loss value logged is 10 to prevent logging big values.
                grad_norm /= log_interval
                lr = optimizer.param_groups[0]['lr']

                if writer is not None and rank == 0:
                    writer.add_scalar('training/loss', total_loss, global_step)
                    writer.add_scalar('training/gnorm', grad_norm, global_step)
                    writer.add_scalar('training/scale', scaler.get_scale(), global_step)
                    writer.add_scalar('training/lr', lr, global_step)

                logger.info(f'| rank = {rank} | global step = {global_step:0>6} | lr = {lr:2.6f} | loss = {total_loss:2.4f} | gnorm = {grad_norm:2.4f} |')
                total_loss = 0
                grad_norm = 0

            if global_step >= max_steps:
                return global_step

    return global_step


def main(args):

    device = torch.device(args.device)
    training_batches = glob(args.train_dir)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "cuda":
        torch.cuda.manual_seed(args.seed)

    # Initialize model
    from .model import AuBERT
    model = AuBERT(
    	args.text_model, 
    	args.audio_model, 
    	proj_dim=args.proj_dim,
    	dropout=0.1,
    	activation="gelu"
    )
    no_params = sum(p.numel() for p in model.parameters())

    logger.info(f"Initialized AuBERT with text encoder: '{args.text_model}', and audio encoder: '{args.audio_model}'")
    logger.info(f"No of parameters {no_params}")

    model.to(device)
    model = DummyDDPWrapper(model)
    scaler = torch.cuda.amp.GradScaler(enabled=args.no_amp)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.base_lr)
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                    num_warmup_steps = args.warmup,
                                                    num_training_steps = args.max_steps,
                                                    num_cycles = 0.45
                                                  )

    best_loss, global_step = 1e3, args.global_step
    if args.load_checkpoint:
        load_states_from_checkpoint(torch.load(args.load_checkpoint), model, optimizer, scaler, lr_scheduler)
        global_step = next(iter(optimizer.state.values()))["step"]
        logger.info(f"Loaded checkpoint model {args.load_checkpoint}")

    writer = SummaryWriter(args.writer_dir)

    for epoch in range(args.epochs):
        random.shuffle(training_batches)
        for batch_path in training_batches:
            # global step is a counter of total update steps
            logger.info(f"Processing {batch_path}")
            global_step = train(
                model,
                batch_path,
                optimizer,
                scaler,
                args,
                global_step=global_step,
                writer=writer,
                device=device,
                lr_scheduler=lr_scheduler
            )

            logger.info(f"Global Step = {global_step:0>6} : finished batch {batch_path}")

            if global_step >= args.max_steps:
                break

        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scaler': scaler.state_dict(),
            'scheduler': lr_scheduler.state_dict()
        }

        # model.module.save(args.checkpoint_path + f"_{global_step}")
        torch.save(checkpoint, args.checkpoint_path + "_last.ckpt")

        evaluate(model.module, args.val_dir, device, args, global_step, writer=writer)
        if args.test_dir:
            evaluate(model.module, args.test_dir, device, args, global_step, writer=writer)

        logger.info("-"*60)
        logger.info(f"Finished epoch {epoch}")
        logger.info("-"*60)

        if global_step >= args.max_steps:
            break


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()

    parser.add_argument("--base_lr", type=float, default=5e-5)
    parser.add_argument("--clip_value", type=float, default=1.0)
    parser.add_argument("--warmup", type=int, default=1000)

    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no_amp", action="store_false")

    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=50000)
    parser.add_argument("--batch_size", type=int, default=16)

    parser.add_argument("--train_dir", type=str, required=True)
    parser.add_argument("--val_dir", type=str, required=True)
    parser.add_argument("--test_dir", type=str, default="")

    parser.add_argument("--checkpoint_path", type=str, default="bin/checkpoints/aubert-checkpoint")
    parser.add_argument("--writer_dir", type=str, default="bin/runs/aubert-trainer")
    parser.add_argument("--load_checkpoint", type=str, default="")
    parser.add_argument("--global_step", type=int, default=0)

    parser.add_argument("--text_model", type=str, default="bert-base-uncased")
    parser.add_argument("--audio_model", type=str, default="facebook/hubert-large-ls960-ft")
    parser.add_argument("--proj_dim", type=int, default=512)

    args = parser.parse_args()

    logger.info("-"*60)
    logger.info("Starting AuBERT Single GPU Trainer")
    logger.info("NameSpace arguments:\n" + json.dumps(vars(args), indent=2))
    logger.info("-"*60)

    main(args)
