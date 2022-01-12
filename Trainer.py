"""This module defines a class to train the models on the fruits 360 dataset."""
import os
import time
import datetime

import matplotlib.pyplot as plt
from DataPrep import DataPrep
import torch


class Trainer:
    def __init__(self, model_name: str, model: torch.nn.Module, batch_size: int, val_size: float):
        """Initialize a Trainer object.

        Args:
            model_name: The name of the model to include in the run name.
            model: The pytorch model.
            batch_size: Batch size for the pytorch data loader.
            val_size: Percentage of all images used for validation.
        """
        # Set instance attributes
        self.seed = 43
        self.model_name = model_name
        self.batch_size = batch_size
        self.val_size = val_size
        self.device = torch.device('cuda:0' if torch.cuda.device_count() > 0 else 'cpu')
        print('Using device:', self.device)

        # Move model to device
        self.model = model.to(self.device)

    def train(self, max_epochs: int, lr: float, momentum: float):
        """Train the model on the fruits-360 dataset.

        Args:
            max_epochs: Max number of epochs to run.
            lr: Learning rate.
            momentum: Momentum for SGD.
        """
        print('LOADING TRAINING DATA...')
        trn_loader, val_loader = self.load_data_trn_val()

        # Generate a name for this run and create run folder
        date = datetime.date.today().strftime('%Y-%m-%d')
        self.run_name = '{}_{}_trn{}_val{}_seed{}_epochs{}_batch{}_lr{}_momentum{}'.format(
            date, self.model_name, len(trn_loader.dataset), self.val_size,
            self.seed, max_epochs, self.batch_size, lr, momentum)

        run_path = os.path.join('runs', self.run_name)
        if not os.path.exists(run_path):
            os.makedirs(run_path)

        # Save a summary of the model in the run folder
        f = open(f'runs/{self.run_name}/model_summary.txt', 'a')
        f.write(self.model.__str__())
        f.close()

        # Initialize pytorch optimizer and loss function
        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)
        err_fn = torch.nn.CrossEntropyLoss()

        # Initial lists to store metrics for each epoch
        err_trn_epochs = []
        acc_trn_epochs = []
        err_val_epochs = []
        acc_val_epochs = []

        print('TRAINING...')
        for epoch in range(max_epochs):
            tik = time.time()

            ## RUN MODEL ON TRAIN DATA ##
            self.model.train()  # set pytorch model to training mode

            # These variables are used to compute average error and accuracy at the end of the epoch
            err_trn_running = 0
            acc_trn_running = 0
            counter_trn = 0

            # Loop over each batch, computing output, backpropagating, and taking a step with SGD
            for batch_idx, (input, target) in enumerate(trn_loader):
                self.model.zero_grad()  # set gradients to zero
                input, target = input.to(self.device), target.to(self.device)

                output = self.model(input)

                err_trn_batch = err_fn(output, target)
                acc_trn_batch = self.compute_acc(output, target)

                err_trn_running += err_trn_batch.item() * len(target)
                acc_trn_running += acc_trn_batch * len(target)
                counter_trn += len(target)

                # Backpropagate gradients and take a step with SGD
                err_trn_batch.backward()
                optimizer.step()

            # Compute trn metrics and append to metrics lists
            err_trn_epoch = err_trn_running / counter_trn
            acc_trn_epoch = acc_trn_running / counter_trn
            err_trn_epochs.append(err_trn_epoch)
            acc_trn_epochs.append(acc_trn_epoch)

            ## RUN MODEL ON VALIDATION DATA ##
            err_val_epoch, acc_val_epoch = self.evaluate(err_fn, val_loader)
            err_val_epochs.append(err_val_epoch)
            acc_val_epochs.append(acc_val_epoch)

            ## PLOT AND PRINT METRICS ##
            if (epoch + 1) % 10 == 0 or (epoch + 1) == max_epochs:
                # Save error and accuracy plots every 10 epochs and after last epoch
                self.save_plot(err_trn_epochs, err_val_epochs, 'Error Plot', run_path, 'err_plot', 3, 5)
                self.save_plot(acc_trn_epochs, acc_val_epochs, 'Accuracy Plot', run_path, 'acc_plot', 0, 1)

            tok = time.time()
            time_taken = tok - tik
            print('Epoch {}/{} | err_trn: {}, acc_trn: {}, err_val: {}, acc_val: {}, time: {} secs'.format(
                epoch + 1, max_epochs, round(err_trn_epoch, 4), round(acc_trn_epoch, 4),
                round(err_val_epoch, 4), round(acc_val_epoch, 4), round(time_taken, 2)))

        # Save pytorch model parameters
        torch.save(self.model.state_dict(), os.path.join(run_path, f'model_epoch{epoch + 1}.pt'))

        ## RUN MODEL ON TEST DATA ##
        err_test, acc_test = self.test_model()
        print('TEST RESULTS | err: {}, acc: {}'.format(round(err_test, 4), round(acc_test, 4)))

        ## SAVE RUN RESULTS SUMMARY ##
        f = open(f'runs/{self.run_name}/run_summary.txt', 'a')
        f.write('err_trn: {}, acc_trn: {}, \nerr_val: {}, acc_val: {} \ntest_err: {}, test_acc: {}'.format(
            round(err_trn_epoch, 4), round(acc_trn_epoch, 4),
            round(err_val_epoch, 4), round(acc_val_epoch, 4),
            round(err_test, 4), round(acc_test, 4)))
        f.close()

        # Rename run folder to include val error and accuracy
        os.rename(run_path, run_path + f'_TestErr{round(err_test, 2)}_TestAcc{round(acc_test, 4)}')

    def test_model(self):
        """Run model on test data (must be run after training and in self.train())."""
        print('LOADING TEST DATA...')
        loader = self.load_data_test()

        print('TESTING...')
        err_fn = torch.nn.CrossEntropyLoss()
        err, acc = self.evaluate(err_fn, loader)
        return err, acc

    def evaluate(self, err_fn, loader):
        """Evaluate model on data in loader and return error + accuracy.

        Args:
            err_fn: Pytorch error function.
            loader: Pytorch data loader containing data to evaluate model on.
        """
        # Used to compute metrics
        err_val_running = 0
        acc_val_running = 0
        counter_val = 0

        # Set model to evaluate mode and run model on data without storing gradients
        self.model.eval()
        self.model.zero_grad()
        with torch.no_grad():
            for batch_idx, (input, target) in enumerate(loader):
                input, target = input.to(self.device), target.to(self.device)

                output = self.model(input)

                err_val_batch = err_fn(output, target)
                acc_val_batch = self.compute_acc(output, target)

                err_val_running += err_val_batch.item() * len(target)
                acc_val_running += acc_val_batch * len(target)
                counter_val += len(target)

        err_val = err_val_running / counter_val
        acc_val = acc_val_running / counter_val
        return err_val, acc_val

    def load_data_trn_val(self):
        """Load data into pytorch trn and val data loaders."""
        trn_loader, val_loader = DataPrep.load_data_into_loaders(
            'fruits-360/Training', batch_size=self.batch_size, shuffle=True,
            seed=self.seed, split_data=True, val_size=self.val_size)
        return trn_loader, val_loader

    def load_data_test(self):
        """Load data into a single pytorch data loader."""
        loader = DataPrep.load_data_into_loaders(
            'fruits-360/Test', batch_size=self.batch_size, shuffle=True,
            seed=self.seed, split_data=False)
        return loader

    def compute_acc(self, output: torch.Tensor, target: torch.Tensor):
        """Compute accuracy between output and target tensors.

        Args:
            output: Model output tensor (vectors of length 113, one for each item in the batch).
            target: Target class numbers (one for each item in the batch).
        """
        correct = 0

        # Loop over images in batch and add 1 to correct if correct
        for i in range(len(target)):
            pred = torch.argmax(output[i])
            label = target[i]

            if pred == label:
                correct += 1

        return correct / len(target)

    @staticmethod
    def save_plot(trn_all: list, val_all: list, title: str, save_path: str, filename: str,
                  ylim_bottom: float, ylim_top: float):
        """Plot train and validation metric lists.

        Args:
            trn_all (list): Train values to plot.
            val_all (list): Validation values to plot.
            title (str): Title of plot.
            save_path (str): Path to save plot.
            filename (str): Plot filename.
            ylim_bottom (float): Lower limit of y-axis.
            ylim_top (float): Upper limit of y-axis.
        """
        # Clear pyplot image before saving
        plt.clf()

        # Set figure size to 512 by 512 pixels (assuming monitor dpi is 96)
        dpi = 96
        plt.figure(figsize=(512 / dpi, 512 / dpi), dpi=dpi)

        # Set y-axis limit, figure title, x and y axes labels
        plt.ylim(ylim_bottom, ylim_top)
        plt.title(title, fontsize=8)
        plt.xlabel("Epoch")
        plt.ylabel("Error")

        # Plot val curves
        plt.plot(val_all, linewidth=2, color=(0.93, 0.75, 0.41))

        # Plot trn curves
        plt.plot(trn_all, linewidth=2, color=(0.41, 0.62, 1))

        # Add legend
        plt.legend(("VAL", "TRN"), loc="upper right")

        # Save and close figure
        plt.savefig("{}.png".format(os.path.join(save_path, filename), format="png"))
        plt.close()
