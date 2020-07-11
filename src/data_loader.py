from torch.utils.data import DataLoader, Dataset
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np


#######################


class Data(Dataset):

    def __init__(self, x, y, num_channels=8, width=20, Mode='2D'):
        self.x = np.load(x)
        self.y = np.load(y)
        # normalize
        self.Mode = Mode
        if Mode == '2D':
            self.x = self.x.transpose((0, 2, 1))
            self.x = self.x.reshape((-1, num_channels, 1, width))
            self.x = self.x.astype(np.float32)
            self.x = (self.x - np.mean(self.x)) / np.std(self.x)
        if Mode == 'S+C':
            self.x = self.x.astype(np.float32)
            self.x = (self.x - np.mean(self.x)) / np.std(self.x)
        if Mode == 'C+S':
            self.x = self.x.transpose((0, 2, 1))
            self.x = self.x.astype(np.float32)
            self.x = (self.x - np.mean(self.x)) / np.std(self.x)
        self.y = self.y
        self.y = self.y.astype(np.long)
        self.y = self.y.reshape((-1))
        self.y = self.y - 1

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        # load image as ndarray type (Height * Width * Channels)
        # be carefull for converting dtype to np.uint8 [Unsigned integer (0 to 255)]
        # in this example, i don't use ToTensor() method of torchvision.transforms
        # so you can convert numpy ndarray shape to tensor in PyTorch (H, W, C) --> (C, H, W)
        if self.Mode == '2D':
            x = self.x[index, :, :, :]
        if self.Mode == 'S+C' or self.Mode == 'C+S':
            x = self.x[index, :, :]
        y = self.y[index]
        return x, y


class DataSet(Dataset):
    def __init__(self, path_to_x, path_to_y):
        self.x = np.load(path_to_x)
        self.y = np.load(path_to_y)
        self.x = self.x.astype(np.float32)
        self.y = self.y.astype(np.long)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]
        return x, y


class DataSetNP(Dataset):
    def __init__(self, x_np, y_np):
        self.x = x_np
        self.y = y_np
        self.x = self.x.astype(np.float32)
        self.y = self.y.astype(np.long)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]
        return x, y


class Config:

    def __init__(self, path_x=None, path_y=None, input_width=20, input_height=1, channel=8, num_classes=6,
                 batch_size=16, num_epochs=20, learning_rate=0.001, shuffle=True):
        self.input_width = input_width
        self.input_height = input_height
        self.channel = channel
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.path_to_x = path_x
        self.path_to_y = path_y
        self.num_classes = num_classes
        self.dataset = Data(self.path_to_x, self.path_to_y, num_channels=channel, width=input_width)
        self.shuffle = shuffle

    def data_loader(self):
        train_loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=self.shuffle)
        return train_loader


def generate_data_loader(batch_size, dataset='MNIST', is_main=True):
    if dataset == 'MNIST':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        if is_main:
            root_dir = '../data'
        else:
            root_dir = '../../data'
        trainset = torchvision.datasets.MNIST(root=root_dir, train=True,
                                              download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                  shuffle=True, num_workers=0)
        testset = torchvision.datasets.MNIST(root=root_dir, train=False,
                                             download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                 shuffle=False, num_workers=0)
    elif dataset == 'CIFAR10':
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        if is_main:
            root_dir = '../data'
        else:
            root_dir = '../../data'

        trainset = torchvision.datasets.CIFAR10(root=root_dir, train=True,
                                                download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                  shuffle=True, num_workers=0)
        testset = torchvision.datasets.CIFAR10(root=root_dir, train=False,
                                               download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                 shuffle=False, num_workers=0)
    elif dataset == 'EMG':

        # path_to_x_train = "/Users/zber/ProgramDev/data_process_jupyter/EMG/data_gen/npy_data/train_X.npy"
        # path_to_y_train = "/Users/zber/ProgramDev/data_process_jupyter/EMG/data_gen/npy_data/train_y.npy"
        # path_to_x_test = "/Users/zber/ProgramDev/data_process_jupyter/EMG/data_gen/npy_data/test_X.npy"
        # path_to_y_test = "/Users/zber/ProgramDev/data_process_jupyter/EMG/data_gen/npy_data/test_y.npy"

        # path_to_x_train = "/Users/zber/ProgramDev/data_process_jupyter/EMG/data_gen/npy_data_w100/train_X.npy"
        # path_to_y_train = "/Users/zber/ProgramDev/data_process_jupyter/EMG/data_gen/npy_data_w100/train_y.npy"
        # path_to_x_test = "/Users/zber/ProgramDev/data_process_jupyter/EMG/data_gen/npy_data_w100/test_X.npy"
        # path_to_y_test = "/Users/zber/ProgramDev/data_process_jupyter/EMG/data_gen/npy_data_w100/test_y.npy"

        path_to_x_train = "/Users/zber/ProgramDev/data_process_jupyter/EMG/data_gen/npy_data_part/train_X.npy"
        path_to_y_train = "/Users/zber/ProgramDev/data_process_jupyter/EMG/data_gen/npy_data_part/train_y.npy"
        path_to_x_test = "/Users/zber/ProgramDev/data_process_jupyter/EMG/data_gen/npy_data_part/test_X.npy"
        path_to_y_test = "/Users/zber/ProgramDev/data_process_jupyter/EMG/data_gen/npy_data_part/test_y.npy"

        trainset = DataSet(path_to_x_train, path_to_y_train)
        testset = DataSet(path_to_x_test, path_to_y_test)
        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
        testloader = DataLoader(testset, batch_size=batch_size, shuffle=True)

    elif dataset == 'Har':

        # path_to_x_train = '/Users/zber/ProgramDev/data_process_jupyter/Har/data_gen/data_npy/train_X.npy'
        # path_to_y_train = '/Users/zber/ProgramDev/data_process_jupyter/Har/data_gen/data_npy/train_y.npy'
        # path_to_x_test = '/Users/zber/ProgramDev/data_process_jupyter/Har/data_gen/data_npy/test_X.npy'
        # path_to_y_test = '/Users/zber/ProgramDev/data_process_jupyter/Har/data_gen/data_npy/test_y.npy'

        path_to_x_train = '/Users/zber/ProgramDev/data_process_jupyter/Har/data_gen/data_npy_shuffle/train_X.npy'
        path_to_y_train = '/Users/zber/ProgramDev/data_process_jupyter/Har/data_gen/data_npy_shuffle/train_y.npy'
        path_to_x_test = '/Users/zber/ProgramDev/data_process_jupyter/Har/data_gen/data_npy_shuffle/test_X.npy'
        path_to_y_test = '/Users/zber/ProgramDev/data_process_jupyter/Har/data_gen/data_npy_shuffle/test_y.npy'

        # path_to_x_train = '/Users/zber/ProgramDev/data_process_jupyter/Har/data_gen/data_npy_part/train_X.npy'
        # path_to_y_train = '/Users/zber/ProgramDev/data_process_jupyter/Har/data_gen/data_npy_part/train_y.npy'
        # path_to_x_test = '/Users/zber/ProgramDev/data_process_jupyter/Har/data_gen/data_npy_part/test_X.npy'
        # path_to_y_test = '/Users/zber/ProgramDev/data_process_jupyter/Har/data_gen/data_npy_part/test_y.npy'

        trainset = DataSet(path_to_x_train, path_to_y_train)
        testset = DataSet(path_to_x_test, path_to_y_test)
        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
        testloader = DataLoader(testset, batch_size=batch_size, shuffle=True)

    elif dataset == 'myHealth':
        path_to_x_test = "/Users/zber/ProgramDev/data_process_jupyter/MyHealth/gen_data/data_npy/test_X.npy"
        path_to_y_test = "/Users/zber/ProgramDev/data_process_jupyter/MyHealth/gen_data/data_npy/test_y.npy"
        path_to_x_train = "/Users/zber/ProgramDev/data_process_jupyter/MyHealth/gen_data/data_npy/train_X.npy"
        path_to_y_train = "/Users/zber/ProgramDev/data_process_jupyter/MyHealth/gen_data/data_npy/train_y.npy"

        trainset = DataSet(path_to_x_train, path_to_y_train)
        testset = DataSet(path_to_x_test, path_to_y_test)
        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
        testloader = DataLoader(testset, batch_size=batch_size, shuffle=True)

    return trainloader, testloader
