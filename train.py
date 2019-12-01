import numpy as np
import torch
import torch.nn as nn
import os
import config
import dataloading
import model
import pdb
import time
import matplotlib.pyplot as plt
import csv

def print_stats(batch_idx, after, before,
                batch_loss, running_loss, accuracy,
                batch_accuracy):
    print("Stats: batch %d" % batch_idx)
    print("Time: ", after - before)
    print("Batch loss", batch_loss)
    print("Running loss", running_loss)
    print("Batch accuracy", batch_accuracy)
    print("Accuracy", accuracy)
    print('\n')

    return

class Trainer(object):
    def __init__(self, model, optimizer):
        super(Trainer, self).__init__()
        torch.cuda.empty_cache()
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optimizer
        self.GPU = torch.cuda.is_available()
        self.batch_size = config.BATCH_SIZE

        if self.GPU:
            print("Model to cuda")
            self.model = self.model.cuda()

    def test(self, test_loader):
        correct, epoch_loss, total = 0., 0., 0.

        before = time.time()
        print(len(test_loader), "batches of size", self.batch_size)
        out_embeddings = []
        for batch_idx, (data, label) in enumerate(test_loader):
            if self.GPU: data = data.cuda(); label = label.cuda();
            data = data.float()
            out = self.model(data, embedding = True)
            out = out.detach().cpu().numpy()

            out_embeddings.append(out)

        arr = np.array(out_embeddings)
        embeddings = np.concatenate(arr, axis = 0)
        np.save(config.EMBEDDINGS_OUTPUT_FILE, embeddings)
        return embeddings

    def train(self, n_epochs, train_loader):
        for epoch in range(n_epochs):
            correct, epoch_loss, total = 0., 0., 0.

            before = time.time()
            print(len(train_loader), "batches of size", self.batch_size)
            for batch_idx, (data, label) in enumerate(train_loader):
                if batch_idx == 0:
                    one_before = time.time()

                data = data.float()

                if self.GPU:
                    data = data.cuda()
                    label = label.cuda()

                self.optimizer.zero_grad()
                out = self.model(data)
                loss = self.criterion(out, label)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

                forward_res = out.detach().cpu().numpy()
                label = label.detach().cpu().numpy()
                predictions = np.argmax(forward_res, axis=1)

                batch_correct = np.sum((predictions == label))
                correct += batch_correct
                total += data.size(0)

                if batch_idx == 0:
                    one_after = time.time()
                    print(one_after - one_before)

                if batch_idx % 100 == 0:
                    after = time.time()
                    print_stats(batch_idx, after, before,
                                loss.item(), epoch_loss / (batch_idx+1),
                                correct / total,
                                float(batch_correct / self.batch_size))
                    before = after

            torch.save(self.model, config.MODELS_PATH +
                                   config.SAVE_MODEL_NAME +
                                   ' epoch%d' % epoch)

        return


def main():
    input_size = config.INPUT_SIZE
    classes = config.N_CLASSES
    neural_net = model.BaselineModel(input_size, classes)
    optim = torch.optim.Adam(neural_net.parameters(),
                             lr = 1e-3)

    trainer = Trainer(neural_net, optim)

    if config.TRAIN_FLAG:
        trainer.train(config.N_EPOCHS,
                      dataloading.train_loader)

    if config.TEST_FLAG:
        trainer.test(dataloading.test_loader)

    return



if __name__ == '__main__':
    main()
