import numpy as np
import torch
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
       
        if self.gpu:
            print("Model to cuda")
            self.model = self.model.cuda()

    def validation(self, val_loader):
        correct, epoch_loss, total = 0., 0., 0.

        before = time.time()
        print(len(val_loader), "batches of size", self.batch_size)
        for batch_idx, (data, label) in enumerate(val_loader):
            if self.gpu: data = data.cuda(); label = label.cuda();

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

            if batch_idx % 100 == 0:
                after = time.time()
                print_stats(batch_idx, after, before, 
                            loss.item(), epoch_loss / (batch_idx+1), 
                            correct / total, WRITE_FILE_FLAG)
                before = after

        print("Done validating stats:")
        print("epoch_loss:", epoch_loss/batch_idx+1)
        print("accuracy:", correct/total)
        print("\n")
        return 


    def train(self, n_epochs, train_loader, val_loader, val_flag):
        for epoch in range(n_epochs):
            correct, epoch_loss, total = 0., 0., 0.

            before = time.time()
            print(len(train_loader), "batches of size", self.batch_size)
            for batch_idx, (data, label) in enumerate(train_loader):
                if batch_idx == 0:
                    one_before = time.time()

                if self.gpu:
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
            if val_flag:
                self.validation(val_loader)

        return 


def main():
    model = model.Baseline_Model()
    optim = torch.optim.Adam(model.parameters(), 
                             lr = 1e-3)

    trainer = Trainer(model, optim)

    TRAIN_FLAG = True
    VAL_FLAG = True
    if TRAIN_FLAG:
        trainer.train(config.N_EPOCHS, 
                      dataloading.train_loader, 
                      dataloading.val_loader,
                      VAL_FLAG)
    
    TEST_FLAG = False
    if TEST_FLAG:
        trainer.test(dataloading.test_loader)

    return 



if __name__ == '__main__':
    main()

