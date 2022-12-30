import os
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.optim as optim

from tqdm import tqdm


class NeuralNetwork(nn.Module):

    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(12000, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 5),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        pred_probab = nn.Softmax(dim=1)(logits)
        # y_pred = pred_probab.argmax(1)
        return pred_probab


def train_simple_net(trainloader, testloader):
    model = NeuralNetwork()
    print(model)
    total_epochs = 30
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # TODO: Tidy up the TQDM sections here
    with tqdm(range(total_epochs)
              ) as all_epochs:  # loop over the dataset multiple times
        for epoch in all_epochs:
            all_epochs.set_description(
                f"Training for {epoch} / {total_epochs} epochs")
            running_loss = 0.0
            running_acc = 0.0
            with tqdm(trainloader, unit="batch") as tepoch:
                for i, data in enumerate(tepoch):
                    tepoch.set_description(f"Epoch {epoch}")

                    # get the inputs; data is a list of [inputs, labels]
                    inputs, labels = data

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward + backward + optimize
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    _, predicted = torch.max(outputs.data, 1)
                    running_acc += 100 * (
                        (predicted == labels).sum().item() / labels.size(0))
                    # print statistics
                    running_loss += loss.item()
                    if i % 1 == 0:
                        tepoch.set_postfix(
                            loss=f"{running_loss / (i + 1):.5f}",
                            acc=f"{running_acc / (i + 1):.2f}%")

    # TODO: Add the validation to get an accuracy as well / ROC for comparisson
    print('Finished Training')
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            # calculate outputs by running images through the network
            outputs = model(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(
        f'Accuracy of the network on the test images: {100 * correct // total} %'
    )


def test_simple_net(test_loader, path_to_weights):
    pass
