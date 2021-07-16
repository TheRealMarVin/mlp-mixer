import numpy as np
import torch


def reshape_prediction_for_compatibility(raw_output):
    reshaped_res = []
    for x in range(raw_output[0].shape[0]):
        curr_out = []
        for y in raw_output:
            curr_out.append(y[x])

        reshaped_res.append(curr_out)

    return np.array(reshaped_res)


def evaluate(model, iterator, criterion, true_index = 1):
    model.eval()

    epoch_loss = 0

    with torch.no_grad():
        all_pred = []
        all_true = []

        for i, batch in enumerate(iterator):
            src = batch[0]
            y_true = batch[true_index]

            if len(y_true.shape) == 1:
                y_true = y_true.type('torch.LongTensor')

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            if device.type == 'cuda':
                src = src.cuda()
                y_true = y_true.cuda()

            y_pred = model(src)
            loss = criterion(y_pred, y_true)

            if type(y_pred) is tuple:
                y_pred, _ = y_pred

            all_pred.extend(y_pred.detach().cpu().numpy())
            all_true.extend(y_true.detach().cpu().numpy())

            epoch_loss += loss.item()

    return all_pred, all_true, epoch_loss / len(iterator)


def convert_string(tokenizer, device, field, text):
    tokenized_text = tokenizer(text)
    res = torch.LongTensor([field.vocab.stoi[x] for x in tokenized_text]).to(device)
    res = res.unsqueeze(-1)
    return res
