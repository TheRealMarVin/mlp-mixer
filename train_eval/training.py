from datetime import timedelta

import torch
import time

from common.train_eval.eval import evaluate


def train(model, train_dataset, optimizer, criterion, scheduler, batch_size, n_epochs, shuffle, history, save_file,
          early_stop=None, train_ratio=0.85, true_index=1):
    tc = int(len(train_dataset) * train_ratio)
    x, y = torch.utils.data.random_split(train_dataset, [tc, len(train_dataset) - tc])
    train_iterator = torch.utils.data.DataLoader(x, batch_size=batch_size, num_workers=4, shuffle=shuffle)
    valid_iterator = torch.utils.data.DataLoader(y, batch_size=batch_size, num_workers=4,
                                                                 shuffle=shuffle)

    best_valid_loss = float('inf')

    for epoch in range(n_epochs):
        start_time = time.time()

        train_loss = train_epoch(model, train_iterator, optimizer, criterion, true_index=true_index)
        _, _, valid_loss = evaluate(model, valid_iterator, criterion, true_index=true_index)
        if scheduler is not None:
            scheduler.step(valid_loss)

        end_time = time.time()

        delta_time = timedelta(seconds=(end_time - start_time))

        history.save_new_data(train_loss, valid_loss)
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            if save_file is not None:
                torch.save(model, save_file)

        print("Current Epoch: {} -> train_eval time: {}\n\tTrain Loss: {:.3f} - Validation Loss: {:.3f}".format(epoch + 1, delta_time, train_loss, valid_loss))

        if early_stop is not None:
            if early_stop.should_stop(valid_loss):
                break

    return best_valid_loss, history


def train_epoch(model, iterator, optimizer, criterion, true_index = 1):
    model.train()

    epoch_loss = 0

    for i, batch in enumerate(iterator):
        src = batch[0]
        y_true = batch[true_index]

        if len(y_true.shape) == 1:
            y_true = y_true.type('torch.LongTensor')

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if device.type == 'cuda':
            src = src.cuda()
            y_true = y_true.cuda()

        optimizer.zero_grad()

        y_pred = model(src)
        loss = criterion(y_pred, y_true)

        loss.backward()

        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss / len(iterator)
