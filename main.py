"""This module is the main driver for the program."""
from CNNModels import *
from Dense1HiddenModels import *
from Dense2HiddenModels import *
from Trainer import Trainer


def main():
    """Run different models on the data and summarize the results in different run folders."""
    max_epochs = 20
    lr = 0.001
    momentum = 0.9
    batch_size = 256  # lower if memory usage is too high

    ## RUN Dense1Hidden Models ##
    model = Dense1Hidden_Tiny()
    trainer = Trainer('Dense1Hidden_Tiny', model, batch_size, 0.2)
    trainer.train(max_epochs, lr, momentum)

    trainer.model = Dense1Hidden_Small().to(trainer.device)
    trainer.model_name = 'Dense1Hidden_Small'
    trainer.train(max_epochs, lr, momentum)

    trainer.model = Dense1Hidden_Medium().to(trainer.device)
    trainer.model_name = 'Dense1Hidden_Medium'
    trainer.train(max_epochs, lr, momentum)

    trainer.model = Dense1Hidden_Big().to(trainer.device)
    trainer.model_name = 'Dense1Hidden_Big'
    trainer.train(max_epochs, lr, momentum)

    ## RUN Dense2Hidden Models ##
    model = Dense2Hidden_Tiny()
    trainer = Trainer('Dense2Hidden_Tiny', model, batch_size, 0.2)
    trainer.train(max_epochs, lr, momentum)

    trainer.model = Dense2Hidden_Small().to(trainer.device)
    trainer.model_name = 'Dense2Hidden_Small'
    trainer.train(max_epochs, lr, momentum)

    trainer.model = Dense2Hidden_Medium().to(trainer.device)
    trainer.model_name = 'Dense2Hidden_Medium'
    trainer.train(max_epochs, lr, momentum)

    trainer.model = Dense2Hidden_Big().to(trainer.device)
    trainer.model_name = 'Dense2Hidden_Big'
    trainer.train(max_epochs, lr, momentum)

    ## RUN CNN Models ##
    model = CNN_Tiny()
    trainer = Trainer('CNN_Tiny', model, batch_size, 0.2)
    trainer.train(max_epochs, lr, momentum)

    model = CNN_Small()
    trainer = Trainer('CNN_Small', model, batch_size, 0.2)
    trainer.train(max_epochs, lr, momentum)

    model = CNN_Medium()
    trainer = Trainer('CNN_Medium', model, batch_size, 0.2)
    trainer.train(max_epochs, lr, momentum)

    model = CNN_Big()
    trainer = Trainer('CNN_Big', model, batch_size, 0.2)
    trainer.train(max_epochs, lr, momentum)

    ## RUN CNN BIG (best model) with other learning rates (already used 0.001) ##
    model = CNN_Big()
    trainer = Trainer('CNN_Big', model, batch_size, 0.2)
    trainer.train(max_epochs, 0.0001, momentum)
    trainer.train(max_epochs, 0.01, momentum)
    trainer.train(max_epochs, 0.1, momentum)


if __name__ == '__main__':
    # Run this method to train the models in main() and get the results shown in the report.
    # Please lower batch size variable in main() if memory usage is too high.
    # Also, note that this takes a fairly long time to run on a gpu.
    main()
