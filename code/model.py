import torch
from torch import nn


class PlanetModel(nn.Module):

    def __init__(self):
        super(PlanetModel, self).__init__()

        series_features = 0
        other_features = 171 # + 38729
        final_features = 110

        # self.series_features = CNN()
        # series_features = self.series_features.num_features

        self.linear_layers = nn.Sequential(
            nn.BatchNorm1d(series_features+other_features),
            linear_module(series_features+other_features, final_features),
            # linear_module(200, final_features),
        )

        self.out_radii = nn.Linear(final_features, 55)
        self.out_sma = nn.Linear(final_features, 1)
        self.out_incl = nn.Linear(final_features, 1)

    def forward(self, matrix, others):
        # m = self.series_features(matrix)
        # x = torch.cat([m, others], dim=1)
        x = self.linear_layers(others)
        return self.out_radii(x).squeeze(), self.out_sma(x).squeeze(), self.out_incl(x).squeeze()


class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        self.channels = 550
        self.num_features = self.channels
        self.matrix_layers = nn.Sequential(
            nn.Conv1d(in_channels=55, out_channels=self.channels, kernel_size=7),
            nn.AdaptiveMaxPool1d(output_size=1),
        )

        # self.channels = 3
        # self.num_features = self.channels * 55
        # self.matrix_layers = nn.Sequential(
        #     nn.Conv2d(in_channels=1, out_channels=self.channels, kernel_size=(1, 7)),
        #     nn.AdaptiveMaxPool2d(output_size=(55, 1)),
        # )

    def forward(self, matrix):
        m = self.matrix_layers(matrix).squeeze()
        m = m.view(matrix.size(0), -1)
        return m


class RNN(nn.Module):

    def __init__(self):
        super(RNN, self).__init__()
        hidden1 = 128
        hidden2 = 128
        self.rnn = nn.GRU(input_size=110, hidden_size=hidden1, batch_first=True,
                          num_layers=5, dropout=0.2)

    def forward(self, matrix):
        x, _ = self.rnn(matrix)
        return x


def linear_module(in_size, out_size):
    return nn.Sequential(
        nn.Linear(in_size, out_size, bias=False),
        nn.ReLU(),
        nn.BatchNorm1d(out_size),
        nn.Dropout(p=0.2)
    )
