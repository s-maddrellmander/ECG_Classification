import torch
from torch import optim
from torch import nn
import logging
from tqdm import tqdm

from torchsummary import summary


class ConvNet(nn.Module):

    def __init__(
        self,
        seq_len,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        num_classes,
    ) -> None:
        super(ConvNet, self).__init__()
        self.num_classes = num_classes
        self.conv_1 = nn.Conv1d(in_channels,
                                out_channels,
                                kernel_size=kernel_size,
                                stride=stride,
                                padding=padding)
        self.LeakyReLU = nn.LeakyReLU(0.01)

        self.conv_2 = nn.Conv1d(out_channels,
                                out_channels,
                                kernel_size=kernel_size,
                                stride=stride,
                                padding=padding)
        self.LeakyReLU_2 = nn.LeakyReLU(0.01)

        self.conv_3 = nn.Conv1d(out_channels,
                                out_channels,
                                kernel_size=kernel_size,
                                stride=stride,
                                padding=padding)
        self.LeakyReLU_3 = nn.LeakyReLU(0.01)

        self.conv_4 = nn.Conv1d(out_channels,
                                out_channels,
                                kernel_size=kernel_size,
                                stride=stride,
                                padding=padding)
        self.conv_5 = nn.Conv1d(out_channels,
                                out_channels,
                                kernel_size=kernel_size,
                                stride=stride,
                                padding=padding)
        self.LeakyReLU_4 = nn.LeakyReLU(0.01)
        self.LeakyReLU_5 = nn.LeakyReLU(0.01)

        # TODO: Assuming padding keeps things the right length
        l_out = out_channels
        fc_size = int(seq_len * l_out)
        self.fully_connected = nn.Linear(in_features=fc_size,
                                         out_features=num_classes)
        self.logSoftmax = nn.LogSoftmax(dim=0)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.LeakyReLU(x)
        x = self.conv_2(x)
        x = self.LeakyReLU_2(x)
        x = self.conv_3(x)
        x = self.LeakyReLU_3(x)
        x = self.conv_4(x)
        x = self.LeakyReLU_4(x)
        x = self.conv_5(x)
        x = self.LeakyReLU_5(x)

        x = nn.Flatten()(x)
        x = self.fully_connected(x)
        output = self.logSoftmax(x)
        return output


def train_conv_net(trainloader, testloader):
    seq_len = trainloader.dataset.data[0].shape[1]
    in_channels = trainloader.dataset.data[0].shape[0]
    out_channels = 128
    kernel_size = 5
    stride = 1
    padding = 2
    num_classes = trainloader.dataset.targets.max() + 1
    print((seq_len, in_channels, out_channels, kernel_size, stride, padding,
           num_classes))
    model = ConvNet(seq_len, in_channels, out_channels, kernel_size, stride,
                    padding, num_classes)
    summary(model, input_size=(12, 1000), batch_size=-1)
    total_epochs = 10
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # TODO: Tidy up the TQDM sections here
    with tqdm(range(total_epochs), colour="#FF6F79"
              ) as all_epochs:  # loop over the dataset multiple times
        for epoch in all_epochs:
            all_epochs.set_description(
                f"Training for {epoch} / {total_epochs} epochs")
            running_loss = 0.0
            running_acc = 0.0
            with tqdm(trainloader, unit="batch", colour="#B5E4EB") as tepoch:
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
                            acc=f"{running_acc / (i + 1):.1f}%")

            # TODO: Add the validation to get an accuracy as well / ROC for comparisson
            # print('Finished Training')
            correct = 0
            total = 0
            val_loss = 0
            # since we're not training, we don't need to calculate the gradients for our outputs
            with torch.no_grad():
                for i, data in enumerate(testloader, 1):
                    images, labels = data
                    # calculate outputs by running images through the network
                    outputs = model(images)
                    val_loss += criterion(outputs, labels)
                    # the class with the highest energy is what we choose as prediction
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            all_epochs.set_postfix(
                val_acc=f'{100 * correct / total:.1f}%',
                val_loss=f"{val_loss / len(testloader):.4f}")
