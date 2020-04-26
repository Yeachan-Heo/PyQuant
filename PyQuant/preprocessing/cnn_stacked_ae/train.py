from PyQuant.preprocessing.cnn_stacked_ae.model import Model
from PyQuant.preprocessing.cnn_stacked_ae.dataloader import get_dataloader
from torch.utils.tensorboard.writer import SummaryWriter
import torch
import torch.nn.functional as F

def train(model, loss_f, x, y, optimizer):
    y_ = model.forward(x)
    loss = loss_f(target=y.squeeze(), input=y_.squeeze())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

def test(model, loss_f, test_dataset):
    losses = []
    for x, y in test_dataset:
        y_ = model.forward(x)
        loss = loss_f(y_, y)
        losses.append(loss)
    mean_loss = torch.stack(losses).mean().item()
    return mean_loss

def main(epoch:int=100, lr=1e-7):
    # log
    writer = SummaryWriter()
    # auto device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model & optim
    model = Model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # data
    train_dl, test_dl = get_dataloader()
    # stp
    global_step = 0
    # loss function
    loss_f = torch.nn.BCELoss()
    print("train start")
    for ep in range(epoch):

        for idx, (x, y) in enumerate(train_dl):
            train_loss = train(model, loss_f, x, y, optimizer)
            writer.add_scalar("loss/train", train_loss, global_step=global_step)
            global_step += 1

        test_loss = test(model, loss_f, test_dl)
        writer.add_scalar("loss/test", test_loss, global_step=ep)

        if ep % 10 == 0:
            torch.save(model.state_dict(), f"./models/model{ep}.pth")
            for key, value in model.state_dict():
                writer.add_histogram(key, value.numpy(), global_step=ep)

if __name__ == '__main__':
    main()







