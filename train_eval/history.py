import pickle

import matplotlib.pyplot as plt


class History:
    TRAIN_LOSS = "Train Loss"
    VAL_LOSS = "Val Loss"

    def __init__(self):
        self.history = {
            self.TRAIN_LOSS: [],
            self.VAL_LOSS: []
        }

    @staticmethod
    def load_from_disk(file_name):
        filehandler = open(file_name, 'rb')
        obj = pickle.load(filehandler)
        return obj

    def save_on_disk(self, file_name):
        filehandler = open(file_name, 'wb')
        pickle.dump(self, filehandler)

    def save_new_data(self, train_loss, val_loss):
        self.history[self.TRAIN_LOSS].append(train_loss)
        self.history[self.VAL_LOSS].append(val_loss)

    def append(self, history):
        self.history[self.TRAIN_LOSS].extend(history.history[self.TRAIN_LOSS])
        self.history[self.VAL_LOSS].extend(history.history[self.VAL_LOSS])

    def display_loss(self, title=None):
        epoch = len(self.history[self.TRAIN_LOSS])
        epochs = [x for x in range(1, epoch + 1)]
        if title is None:
            plt.title('Training loss')
        else:
            plt.title(title)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.plot(epochs, self.history[self.TRAIN_LOSS], label='Train')
        plt.plot(epochs, self.history[self.VAL_LOSS], label='Validation')
        plt.legend()
        plt.show()

    def to_experiment(self, experiment):
        train_loss = self.history[self.TRAIN_LOSS]
        for x in train_loss:
            experiment.log_scalar("training.loss", x)
        val_loss = self.history[self.VAL_LOSS]
        for y in val_loss:
            experiment.log_scalar("validation.loss", y)
