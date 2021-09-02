import numpy
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from characterdataset import CharacterDataset
from model import Model
from torch.utils.data import DataLoader


def label_transform(z):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return torch.tensor(z).to(device)


def transform(z):
    tmp = torch.tensor(z).unsqueeze(0).to('cuda' if torch.cuda.is_available() else 'cpu')
    tmp.requires_grad_()
    return tmp


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f'loss: {loss:>7f}  [{current:>5d}/{size:>5d}]')


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f'Test Error \n Accuracy: {(100 * correct):>1f}%, Avg loss: {test_loss:>8f}')


def main():
    learning_rate = 1e-3
    momentum = 0.9
    batch_size = 64
    epochs = 18

    train_data = CharacterDataset('data/Train_Set_Handwritten.csv', transform=transform, target_transform=label_transform)
    test_data = CharacterDataset('data/Test_Set_Handwritten.csv', transform=transform, target_transform=label_transform)
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'using {device} device')

    model = Model().to(device)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer)
        test_loop(test_dataloader, model, loss_fn)
    print("Done!")
    torch.save(model.state_dict(), 'model_weights.pth')


if __name__ == '__main__':
    main()
