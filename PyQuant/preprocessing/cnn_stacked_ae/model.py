import torch.nn as nn
import torch

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 64, 3),
            nn.ReLU(),
            #nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 3),
            nn.ReLU(),
            #nn.MaxPool1d(2),
            nn.Conv1d(128, 256, 3),
            nn.ReLU(),
            #nn.MaxPool1d(2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(256, 128, 3),
            nn.ReLU(),
            nn.ConvTranspose1d(128, 64, 3),
            nn.ReLU(),
            nn.ConvTranspose1d(64, 1, 3),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

if __name__ == '__main__':
    m = Model()
    a = m.forward(torch.arange(100).reshape(1,1,100).float())
    print(a.shape)