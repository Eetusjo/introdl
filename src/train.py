import argparse
import logging
import models
import torch
import torch.nn as nn

from data import read_datasets, WORD_BOUNDARY, UNK, get_minibatch
from paths import data_dir

logging.basicConfig(format="%(asctime)s: %(message)s",
                    datefmt="%d/%m/%Y %I:%M:%S %p",
                    level=logging.DEBUG)


def train(model, data_train, optimizer, loss_fn, steps, log_interval):
        """"TODO."""
        model.train()
        running_loss = 0

        step = 0
        while step < steps:
            for batch in data_train:
                x, x_lengths, y = batch

                optimizer.zero_grad()
                output = model(x, x_lengths)

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

                step += 1
                if step >= steps:
                    break

            # if do_validation:
            #     val_log = self._valid_epoch(epoch)
            #     log = {**log, **val_log}

            # if lr_scheduler is not None:
            #     lr_scheduler.step()

            # if (teacherf_decrease > 0) and model.teacherf_ratio > 0:
            #     model.teacherf_ratio -= teacherf_decrease
            #     if model.teacherf_ratio < 0:
            #         model.teacherf_ratio = 0
            #     logging.info(
            #         'Teacher forcing ratio set to: {}'.format(model.teacherf_ratio)
            #     )


def validate(model, data_valid):
    model.eval()
    model.train()

    raise NotImplementedError()


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

        curr = self.b
        self.b += self.batch_size

        batch = sorted(self.data[curr:curr + self.batch_size],
                       key=lambda x: len(x["SOURCE"].split(" ")),
                       reverse=True)
        x, y = get_minibatch(batch, self.char_map, None)
        x_lengths = [len(b["SOURCE"].split(" ")) for b in batch]

        return x, x_lengths, y


def main(lang):
    # setup data_loader instances
    data, character_map = read_datasets(lang + '-task1', data_dir)
    trainset = [datapoint for datapoint in data['training']]
    train_iter = DataIterator(trainset, 16, character_map)

    # build model architecture
    model = models.Seq2SeqModel(embedding_dim=128, hidden_dim=512,
                                vocabulary_size=len(character_map),
                                max_target_length=50,
                                sos=character_map[WORD_BOUNDARY])

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train(model=model, data_train=train_iter, optimizer=optimizer,
          loss_fn=nn.NLLLoss(ignore_index=character_map[WORD_BOUNDARY]),
          steps=10000, log_interval=100)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument("--placeholder", action="store_true")
    args = parser.parse_args()

    main("finnish")
