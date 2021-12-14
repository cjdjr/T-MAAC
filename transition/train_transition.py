
import torch
import numpy as np
from torch import optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, dataloader
from torch.utils.data import random_split

from torch.optim import lr_scheduler
import argparse

from model import transition_model, transition_model_linear

NUM_EPOCHS = 100
LR = 0.001
SAVE_INTERVAL = 10

class TransitionDataset(Dataset):

    def __init__(self, data):
        self.state = torch.from_numpy(data['state']).to(torch.float32)
        self.q = torch.from_numpy(data['q']).to(torch.float32)
        self.res_v = torch.from_numpy(data['res_v']).to(torch.float32)

    def __getitem__(self, index):
        data = torch.cat((self.state[index],self.q[index]),dim=0)
        return data, self.res_v[index]
    def __len__(self):
        return self.state.shape[0]
        

def get_args():
    parser = argparse.ArgumentParser(description="Train rl agent.")
    parser.add_argument("--scenario", type=str, nargs="?", default="case322_3min_final", help="Please input the valid name of an environment scenario.")
    args = parser.parse_args()
    args.num_epochs = NUM_EPOCHS
    args.lr = LR
    args.save_interval = SAVE_INTERVAL
    return args

if __name__=="__main__":
    args = get_args()

    _data = np.load(args.scenario+'.npy',allow_pickle=True).item()
    dataset = TransitionDataset(_data)
    len_train = int(len(dataset) * 0.8)
    len_val = len(dataset) - len_train
    train_dataset, valid_dataset = random_split(
        dataset=dataset,
        lengths=[len_train, len_val],
        generator=torch.Generator().manual_seed(0)
    )
    train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True, pin_memory=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=256, shuffle=False, pin_memory=True)

    # model=transition_model().cuda()
    model=transition_model_linear().cuda()

    loss_func=nn.MSELoss()
    optm=torch.optim.Adam(model.parameters(),args.lr)
    scheduler = lr_scheduler.StepLR(optm, step_size=10, gamma=0.1)
    train_epochs_loss = []
    valid_epochs_loss = []
    # acc=acc_func()
    for epoch in range(args.num_epochs):
        train_epoch_loss = []
        scheduler.step()
        for idx,(data_x, data_y) in enumerate(train_dataloader,0):
            data_x = data_x.cuda()
            data_y = data_y.cuda()
            outputs = model(data_x)
            optm.zero_grad()
            loss = loss_func(data_y,outputs)
            loss.backward()
            optm.step()
            train_epoch_loss.append(loss.item())
            # train_loss.append(loss.item())
            if idx%(len(train_dataloader)//2)==0:
                print("epoch={}/{},{}/{}of train, loss={}".format(
                    epoch, NUM_EPOCHS, idx, len(train_dataloader),loss.item()))
        train_epochs_loss.append(np.average(train_epoch_loss))
    
        #=====================valid============================

        valid_epoch_loss = []
        with torch.no_grad():
            for idx,(data_x,data_y) in enumerate(valid_dataloader,0):
                data_x = data_x.cuda()
                data_y = data_y.cuda()
                outputs = model(data_x)
                loss = torch.mean(torch.abs(outputs - data_y))
                valid_epoch_loss.append(loss.item())
        print("val epoch = {}   :    {}".format(epoch,np.average(valid_epoch_loss)))
        valid_epochs_loss.append(np.average(valid_epoch_loss))

        if epoch % args.save_interval ==0 :
            path = args.scenario + '.lin_model{}'.format(epoch)
            print("SAVE model to {}".format(path))
            torch.save(model.state_dict(), path)

    print(valid_epochs_loss)