import argparse
import logging
import models
import torch
import torch.nn as nn

from data import read_datasets, WORD_START, UNK, get_minibatch, PADDING
from paths import data_dir

logging.basicConfig(format="%(asctime)s: %(message)s",
                    datefmt="%d/%m/%Y %I:%M:%S %p",
                    level=logging.DEBUG)


def train(model, data_train, optimizer, loss_fn, steps, log_interval=100,
          valid_interval=-1, data_valid=None, device=None):
        """"TODO."""
        model.train()
        running_loss = 0

        step = 0
        while step < steps:
            for batch in data_train:
                x, x_lengths, y = batch

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

                running_loss = 0.1*running_loss + 0.9*loss.item()

                if step % log_interval == 0:
                    logging.info(
                        'Training step {}/{} ({:.0f}%) Running loss: {:.6f}'.format(
                            step, steps, 100*step/steps, loss.item()
                        )
                    )

                if valid_interval > 0 and data_valid and step % valid_interval == 0:
                    val_metrics = evaluate(model, data_valid, device)
                    logging.info(
                        'Validation step {}/{} ({:.0f}%) loss: {:.6f} accuracy: {:.1f}'.format(
                            step, steps, 100*step/steps, val_metrics["loss"],
                            val_metrics["accuracy"]
                        )
                    )

                step += 1
                if step >= steps:
                    break


def evaluate(model, data, device):
    model.eval()
    with torch.no_grad():
        for batch in data:
            x, x_lengths, y = batch

            if device:
                x, x_lengths, y = x.to(device=device), \
                    x_lengths.to(device=device), \
                    y.to(device=device)

            output = model(x, x_lengths, device)

    model.train()

    return {"loss": 0., "accuracy": 0.}


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
            raise StopIteration

        curr = self.b
        self.b += self.batch_size

        batch = sorted(self.data[curr:curr + self.batch_size],
                       key=lambda x: len(x["SOURCE"].split(" ")),
                       reverse=True)
        x, y = get_minibatch(batch, self.char_map, None)
        x_lengths = torch.tensor([len(b["SOURCE"].split(" ")) for b in batch])

        return x, x_lengths, y


def main(args):
    use_gpu = args.cuda and torch.cuda.is_available()
    logging.info("Using gpu: {}".format(use_gpu))
    print("Using gpu: {}".format(use_gpu))
    # setup data_loader instances
    data, character_map = read_datasets(args.language + '-task1', data_dir)
    trainset = [datapoint for datapoint in data['training']]
    train_iter = DataIterator(trainset, 16, character_map)

    validset = [datapoint for datapoint in data['dev']]
    valid_iter = DataIterator(validset, 128, character_map)

    # build model architecture
    model = models.Seq2SeqModel(embedding_dim=128, hidden_dim=512,
                                vocabulary_size=len(character_map),
                                max_target_length=50,
                                sos=character_map[WORD_START])
    if use_gpu:
        model.to(device="cuda")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    train(model=model, data_train=train_iter, optimizer=optimizer,
          loss_fn=nn.NLLLoss(ignore_index=character_map[PADDING]),
          steps=1000, log_interval=100, valid_interval=100,
          data_valid=valid_iter, device=torch.device("cuda") if use_gpu else None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Placeholder')
    parser.add_argument("--language", choices=["finnish", "german", "navajo"],
                        required=True, help="Language to train.")
    parser.add_argument("--cuda", action="store_true", help="Use gpu")
    args = parser.parse_args()

    main(args)
