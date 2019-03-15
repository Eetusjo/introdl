import argparse
import logging
import models
import os
import torch
import torch.nn as nn
import re
import numpy as np

from data import read_datasets, WORD_START, UNK, get_minibatch, PADDING
from paths import data_dir

logging.basicConfig(format="%(asctime)s: %(message)s",
                    datefmt="%d/%m/%Y %I:%M:%S %p",
                    level=logging.DEBUG)


def train(model, data_train, optimizer, loss_fn, steps, log_interval=100,
          valid_interval=-1, id2char_map=None, data_valid=None,
          save_dir=None, exp_name=None, device=None):
        """"TODO."""
        # Make sure training mode is on
        model.train()
        # Keep track of running loss
        running_loss = 0

        # Keep track of best valid loss for saving models
        best_valid_loss = 100
        step = 0
        while step < steps:
            for batch in data_train:
                x, x_lengths, y = batch

                # Move batch to device if using gpu
                if device:
                    x, x_lengths, y = x.to(device=device), \
                                      x_lengths.to(device=device), \
                                      y.to(device=device)

                optimizer.zero_grad()
                output = model(x, x_lengths, device)

                # Sum losses from each decoder timestep
                loss = sum(
                    [loss_fn(output[t], y_t) for t, y_t in enumerate(y)]
                )

                loss.backward()
                optimizer.step()

                running_loss = 0.1*running_loss + 0.9*loss.item()/len(y)

                if step % log_interval == 0:
                    logging.info(
                        'Training step {}/{} ({:.0f}%) Running loss: {:.6f}'.format(
                            step, steps, 100*step/steps, running_loss
                        )
                    )

                if valid_interval > 0 and data_valid and step % valid_interval == 0:
                    val_metrics = evaluate(model, data_valid, id2char_map,
                                           loss_fn, device)
                    logging.info(
                        'Validation step {}/{} ({:.0f}%) loss: {:.6f} accuracy: {:.1f}'.format(
                            step, steps, 100*step/steps, val_metrics["loss"],
                            val_metrics["accuracy"]
                        )
                    )
                    # Only save new model if better than earlier
                    if val_metrics["loss"] < best_valid_loss:
                        torch.save(model.state_dict(), "{}/{}-{}.pt".format(
                            save_dir, exp_name, step
                        ))
                        best_valid_loss = val_metrics["loss"]

                step += 1
                if step >= steps:
                    break


def evaluate(model, data, id2char_map, loss_fn, device):
    correct = 0
    total = 0

    model.eval()

    batches = 0
    loss_sum = 0
    with torch.no_grad():
        for batch in data:
            x, x_lengths, y = batch

            if device:
                x, x_lengths, y = x.to(device=device), \
                    x_lengths.to(device=device), \
                    y.to(device=device)

            output = model(x, x_lengths, device)
            batch_size = len(output[0])

            # turn x to a list of strings
            x_strings = [''] * batch_size
            for time_step in output:
                top_chars = torch.argmax(time_step, dim=1)
                for i, index in enumerate(top_chars):
                    x_strings[i] += str(id2char_map[index.item()])

            # if '<w>' and '</w>' are in output, choose the characters between them
            for i, x in enumerate(x_strings):
                if re.search('<w>(.*)</w>', x):
                    x_strings[i] = re.search('<w>(.*)</w>', x).group(1)


            # turn y to a list of strings
            y_strings = [''] * batch_size
            for time_step in y:
                for i, char_id in enumerate(time_step):
                    y_strings[i] += str(id2char_map[char_id.item()])

            y_strings = [re.search('<w>(.*)</w>', y).group(1) for y in y_strings]

            total += batch_size
            n_correct = len([x for x, y in zip(x_strings, y_strings) if x==y])
            correct += n_correct

            # Sum losses from each decoder timestep
            loss = sum(
                [loss_fn(output[t], y_t) for t, y_t in enumerate(y)]
            )
            loss_sum += loss.item()/len(y)
            batches += 1

    model.train()

    return {"loss": loss_sum/batches, "accuracy": 100.0 * correct / total}


class DataIterator:
    def __init__(self, data, batch_size, char_map):
        self.data = data
        self.batch_size = batch_size
        self.char_map = char_map

        self.n = len(data)
        self.b = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.b > self.n:
            self.b = 0
            raise StopIteration

        curr = self.b
        self.b += self.batch_size

        batch = sorted(self.data[curr:curr + self.batch_size],
                       key=lambda x: len(x["SOURCE"].split(" ")),
                       reverse=True)
        x, y = get_minibatch(batch, self.char_map, None)
        x_lengths = torch.tensor([len(b["SOURCE"].split(" ")) for b in batch])

        return x, x_lengths, y

    def __len__(self):
        return len(self.data)


def main(args):
    use_gpu = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda") if use_gpu else None

    logging.info("Using gpu: {}".format(use_gpu))
    # setup data_loader instances
    data, character_map, id2char_map = read_datasets(args.language + '-task1', data_dir)
    trainset = [datapoint for datapoint in data['training']]
    train_iter = DataIterator(trainset, args.bsize, character_map)

    validset = [datapoint for datapoint in data['dev']]
    valid_iter = DataIterator(validset, args.valid_bsize, character_map)

    # build model architecture
    model = models.Seq2SeqModel(embedding_dim=args.embedding_dim,
                                hidden_dim=args.hidden_dim,
                                vocabulary_size=len(character_map),
                                max_target_length=50,
                                sos=character_map[WORD_START])
    if use_gpu:
        model.to(device=device)

    if args.resume:
        model.load_state_dict(torch.load(args.resume))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    train(model=model, data_train=train_iter, optimizer=optimizer,
          loss_fn=nn.NLLLoss(ignore_index=character_map[PADDING]),
          steps=args.steps, log_interval=args.log_interval,
          valid_interval=args.valid_interval, data_valid=valid_iter,
          save_dir=args.save_dir, exp_name=args.exp, id2char_map=id2char_map,
          device=device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Placeholder')
    parser.add_argument("--exp", required=True,
                        help="Name for experiment. Used in saving model.")
    parser.add_argument("--language", choices=["finnish", "german", "navajo"],
                        required=True, help="Language to train.")
    parser.add_argument("--embedding_dim", type=int, default=128,
                        help="Dimensionality of char embeddings")
    parser.add_argument("--hidden_dim", type=int, default=256,
                        help="Dimensionality of LSTM hidden states")
    parser.add_argument("--bsize", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--valid_bsize", type=int, default=128,
                        help="Validation batch size")
    parser.add_argument("--steps", type=int, default=100000,
                        help="Training steps")
    parser.add_argument("--log_interval", type=int, default=500,
                        help="Interval for logging")
    parser.add_argument("--valid_interval", type=int, default=2000,
                        help="Interval for validating adn saving model.")
    parser.add_argument("--lr", type=float, default=0.0001,
                        help="Learning rate for optimizer")
    parser.add_argument("--cuda", action="store_true", help="Use gpu")
    parser.add_argument("--save_dir", required=True,
                        help="Directory for saving the model")
    parser.add_argument("--resume", default=None,
                        help="Path to model checkpoint if resuming training")
    args = parser.parse_args()

    main(args)
