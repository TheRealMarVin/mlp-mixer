import torch
import torch.nn as nn

import numpy as np

import os
from os import path

import torchvision
from matplotlib.pyplot import imshow
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from torch import optim

from torch.utils.data import SubsetRandomSampler
from torchvision import transforms

from mlp_mixer.mlp_mixer import MlpMixer
from train_eval.eval import evaluate
from train_eval.history import History
from train_eval.training import train


def get_mnist_sets(train_transform, test_transform):
    train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=train_transform)
    test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=test_transform)
    return train_set, test_set


def get_cifar10_sets(train_transform, test_transform):
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)

    return train_set, test_set


def run_experiment():
    n_epochs = 200
    batch_size = 64
    learning_rate = 0.00005

    model_name = "mlp_mixer"

    out_folder = "models/{}".format(model_name)
    if not path.exists(out_folder):
        os.makedirs(out_folder)

    save_file = "{}/best.model".format(out_folder)

    train_transform = transforms.Compose([

                                          transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.1307,), (0.3081,))
                                          ])
    test_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((0.1307,), (0.3081,))])

    train_set, test_set = get_mnist_sets(train_transform, test_transform)
    # train_set, test_set = get_cifar10_sets(train_transform, test_transform)
    # train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, num_workers=4,
    #                                            shuffle=True)

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)

    model = MlpMixer(image_input_size=28,
                     nb_channels=1,
                     patch_size=4,
                     nb_blocks=4,
                     out_size=10,
                     hidden_size=64,
                     dropout=0.1)

    if torch.cuda.is_available():
        model = model.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    history = History()
    train(model, train_set, optimizer, criterion, None, batch_size, n_epochs, True, history, save_file, early_stop=None,
          true_index=1)
    print('Finished Training')

    y_pred, y_true, valid_loss = evaluate(model, test_loader, criterion)
    y_pred = np.array(y_pred).argmax(1)
    print(classification_report(y_true, y_pred))
    print(confusion_matrix(y_true, y_pred))

    print("Done")


if __name__ == '__main__':
    run_experiment()
