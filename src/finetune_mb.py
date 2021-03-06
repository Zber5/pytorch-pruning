import torch.nn as nn
import torch.optim as optim
from src import dataset
from src.prune import *
import argparse
from operator import itemgetter
from heapq import nsmallest
from data_loader import generate_data_loader
from model import NetS

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ModifiedVGG16Model(torch.nn.Module):
    def __init__(self):
        super(ModifiedVGG16Model, self).__init__()

        model = models.vgg16(pretrained=True)
        self.features = model.features

        for param in self.features.parameters():
            param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(25088, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 2))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class FilterPrunner:
    def __init__(self, model):
        self.model = model
        self.reset()

    def reset(self):
        self.filter_ranks = {}

    def forward(self, x):
        self.activations = []
        self.gradients = []
        self.grad_index = 0
        self.activation_to_layer = {}

        activation_index = 0
        for layer, (name, module) in enumerate(self.model.model._modules.items()):
            x = module(x)
            if isinstance(module, torch.nn.modules.Sequential):
                x.register_hook(self.compute_rank)
                self.activations.append(x)
                self.activation_to_layer[activation_index] = layer
                activation_index += 1

        #
        # for layer, (name, module) in enumerate(self.model.classifier._modules.items()):
        #     x = module(x)
        #     if isinstance(module, torch.nn.modules.linear.Linear):
        #         x.register_hook(self.compute_rank)
        #
        #

        return self.model.fc(x.view(x.size(0), -1))

    def compute_rank(self, grad):
        activation_index = len(self.activations) - self.grad_index - 1
        activation = self.activations[activation_index]

        taylor = activation * grad
        # Get the average value for every filter, 
        # accross all the other dimensions
        taylor = taylor.mean(dim=(0, 2, 3)).data

        if activation_index not in self.filter_ranks:
            self.filter_ranks[activation_index] = \
                torch.FloatTensor(activation.size(1)).zero_()

            if args.use_cuda:
                self.filter_ranks[activation_index] = self.filter_ranks[activation_index].cuda()

        self.filter_ranks[activation_index] += taylor
        self.grad_index += 1

    def lowest_ranking_filters(self, num):
        data = []
        for i in sorted(self.filter_ranks.keys()):
            for j in range(self.filter_ranks[i].size(0)):
                data.append((self.activation_to_layer[i], j, self.filter_ranks[i][j]))

        return nsmallest(num, data, itemgetter(2))

    def normalize_ranks_per_layer(self):
        for i in self.filter_ranks:
            v = torch.abs(self.filter_ranks[i])
            v = v / np.sqrt(torch.sum(v * v))
            self.filter_ranks[i] = v.cpu()

    def get_prunning_plan(self, num_filters_to_prune):
        filters_to_prune = self.lowest_ranking_filters(num_filters_to_prune)

        # After each of the k filters are prunned,
        # the filter index of the next filters change since the model is smaller.
        filters_to_prune_per_layer = {}
        for (l, f, _) in filters_to_prune:
            if l not in filters_to_prune_per_layer:
                filters_to_prune_per_layer[l] = []
            filters_to_prune_per_layer[l].append(f)

        for l in filters_to_prune_per_layer:
            filters_to_prune_per_layer[l] = sorted(filters_to_prune_per_layer[l])
            for i in range(len(filters_to_prune_per_layer[l])):
                filters_to_prune_per_layer[l][i] = filters_to_prune_per_layer[l][i] - i

        filters_to_prune = []
        for l in filters_to_prune_per_layer:
            for i in filters_to_prune_per_layer[l]:
                filters_to_prune.append((l, i))

        return filters_to_prune


class PrunningFineTuner_VGG16:
    def __init__(self, train, test, model):
        # self.train_data_loader = dataset.loader(train_path)
        # self.test_data_loader = dataset.test_loader(test_path)
        self.train_data_loader = train
        self.test_data_loader = test
        self.model = model
        self.criterion = torch.nn.CrossEntropyLoss()
        self.prunner = FilterPrunner(self.model)
        self.model.train()

    def test(self):
        self.model.eval()
        correct = 0
        total = 0

        for i, (batch, label) in enumerate(self.test_data_loader):
            if args.use_cuda:
                batch = batch.cuda()
            output = model(Variable(batch))
            pred = output.data.max(1)[1]
            correct += pred.cpu().eq(label).sum()
            total += label.size(0)

        print("Accuracy :", float(correct) / total)

        self.model.train()
        return float(correct) / total

    def train(self, optimizer=None, epoches=10):
        if optimizer is None:
            # optimizer = optim.SGD(model.classifier.parameters(), lr=param['lr'], momentum=0.9)
            optimizer = optim.Adam(self.model.parameters(), lr=config.lr)

        for i in range(epoches):
            print("Epoch: ", i)
            self.train_epoch(optimizer)
            self.test()
        print("Finished fine tuning.")

    def train_batch(self, optimizer, batch, label, rank_filters):

        if args.use_cuda:
            batch = batch.cuda()
            label = label.cuda()

        self.model.zero_grad()
        input = Variable(batch)

        if rank_filters:
            output = self.prunner.forward(input)
            self.criterion(output, Variable(label)).backward()
        else:
            self.criterion(self.model(input), Variable(label)).backward()
            optimizer.step()

    def train_epoch(self, optimizer=None, rank_filters=False):
        for i, (batch, label) in enumerate(self.train_data_loader):
            self.train_batch(optimizer, batch, label, rank_filters)

    def get_candidates_to_prune(self, num_filters_to_prune):
        self.prunner.reset()
        self.train_epoch(rank_filters=True)
        self.prunner.normalize_ranks_per_layer()
        return self.prunner.get_prunning_plan(num_filters_to_prune)

    # def total_num_filters(self):
    #     filters = 0
    #     for name, module in self.model.features._modules.items():
    #         if isinstance(module, torch.nn.modules.conv.Conv2d):
    #             filters = filters + module.out_channels
    #     return filters

    def total_num_filters(self):
        filters = 0
        for module in self.model.modules():
            if isinstance(module, torch.nn.modules.conv.Conv2d):
                filters = filters + module.out_channels
        return filters

    def prune(self):
        # Get the accuracy before prunning
        self.test()
        self.model.train()

        # Make sure all the layers are trainable
        # for param in self.model.features.parameters():
        #     param.requires_grad = True

        number_of_filters = self.total_num_filters()
        num_filters_to_prune_per_iteration = config.num_filters_to_prune_per_iteration
        iterations = int(float(number_of_filters) / num_filters_to_prune_per_iteration)

        iterations = int(iterations * config.prune_percentage)

        print("{} pruning iterations to reduce {:.2f}% filters".format(iterations, config.prune_percentage * 100))

        for _ in range(iterations):
            print("Ranking filters.. ")
            prune_targets = self.get_candidates_to_prune(num_filters_to_prune_per_iteration)
            layers_prunned = {}
            for layer_index, filter_index in prune_targets:
                if layer_index not in layers_prunned:
                    layers_prunned[layer_index] = 0
                layers_prunned[layer_index] = layers_prunned[layer_index] + 1

            print("Layers that will be prunned", layers_prunned)
            print("Prunning filters.. ")
            model = self.model.cpu()
            for layer_index, filter_index in prune_targets:
                model = prune_mbnet_layer(model, layer_index, filter_index)

            self.model = model
            if args.use_cuda:
                self.model = self.model.cuda()

            message = str(100 * float(self.total_num_filters()) / number_of_filters) + "%"
            print("Filters prunned", str(message))
            print("New model size is {}".format(model_size(find_modules(self.model))))
            self.test()
            print("Fine tuning to recover from prunning iteration.")

            # optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
            optimizer = optim.Adam(self.model.parameters(), lr=config.lr)
            self.train(optimizer, epoches=config.retrian_epoch)
            acc = self.test()
            if acc < config.threshold:
                break

        print("Finished. Going to fine tune the model a bit more")
        self.train(optimizer, epoches=config.finetune_epoch)
        # torch.save(model.state_dict(), "model_prunned")

    def save_model(self, model_path):
        torch.save(self.model.state_dict(), model_path)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", dest="train", action="store_true")
    parser.add_argument("--prune", dest="prune", action="store_true")
    parser.add_argument("--train_path", type=str, default="train")
    parser.add_argument("--test_path", type=str, default="test")
    parser.add_argument('--use-cuda', action='store_true', default=False, help='Use NVIDIA GPU acceleration')
    parser.set_defaults(train=False)
    parser.set_defaults(prune=False)
    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()

    return args


def find_modules(model):
    modules = []
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            modules.append(module)
    return modules


def find_modules_short(model):
    modules = []
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            modules.append(module)
    return modules


def model_size(layers):
    size = []
    for module in layers:
        size.append(module.weight.data.shape[0])
    return size


def obtain_ff(dic, width=128):
    strides = [2, 1, 2, 2]
    for i in range(4):
        width = (width - dic['kernel_size']) // strides[i] + 1
    return width


if __name__ == '__main__':
    set_mode("MobileNet")
    args = get_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()

    # args = get_args()
    #
    # if args.train:
    #     model = ModifiedVGG16Model()
    # elif args.prune:
    #     model = torch.load("model", map_location=lambda storage, loc: storage)
    #
    # if args.use_cuda:
    #     model = model.cuda()
    #
    # fine_tuner = PrunningFineTuner_VGG16(args.train_path, args.test_path, model)
    #
    # if args.train:
    #     fine_tuner.train(epoches=10)
    #     torch.save(model, "model")
    #
    # elif args.prune:
    #     fine_tuner.prune()

    # dataset = "MNIST"
    dataset = "Fin_Sitting"
    # dataset = "Fin_SitRun"

    class config:
        lr = 0.0005
        # lr = 0.001
        epoch = 20
        batch_size = 64
        is_bn = False
        num_filters_to_prune_per_iteration = 10
        prune_percentage = 0.8
        threshold = 0.97
        retrian_epoch = 5
        finetune_epoch = 10


    trainloader, testloader = generate_data_loader(batch_size=config.batch_size, dataset=dataset)

    # kwargs = {'out1': 20, 'out2': 50, 'fc1': 500}
    #
    # model_path = "/Users/zber/ProgramDev/exp_pyTorch/results/Models/model_standard_20_50_500_.ckpt"
    # har_path = ""
    #
    #
    # # model = LeNet5_GROW_P(**kwargs)

    # net.load_state_dict(torch.load('models/convnet_pretrained.pkl'))
    # EMG

    if dataset == "Har":
        input_shape = (9, 1, 128)
        width = 128
        kwargs = {
            'in_channel': 9,
            'out1_channel': 32,
            'out2_channel': 64,
            'out3_channel': 128,
            'out4_channel': 256,
            'out_classes': 6,
            'kernel_size': 14,
            'avg_factor': 2
        }

        ff = obtain_ff(kwargs, width)
        kwargs['avg_factor'] = ff

    elif dataset == "EMG":
        input_shape = (8, 1, 100)
        width = 100
        kwargs = {
            'in_channel': 8,
            'out1_channel': 32,
            'out2_channel': 64,
            'out3_channel': 128,
            'out4_channel': 256,
            'out_classes': 6,
            'kernel_size': 12,
            'avg_factor': 2
        }

        ff = obtain_ff(kwargs, width)
        kwargs['avg_factor'] = ff

    elif dataset == 'myHealth':
        width = 100
        input_shape = (23, 1, 100)
        kwargs = {
            'in_channel': 23,
            'out1_channel': 32,
            'out2_channel': 64,
            'out3_channel': 128,
            'out4_channel': 256,
            'out_classes': 11,
            'kernel_size': 12,
            'avg_factor': 2
        }

        f_kwargs = {
            'in_channel': 23,
            'out1_channel': 32,
            'out2_channel': 64,
            'out3_channel': 128,
            'out4_channel': 256,
            'out_classes': 11,
            'kernel_size': 12,
            'avg_factor': 1
        }
        ff = obtain_ff(kwargs, width)
        kwargs['avg_factor'] = ff

    elif dataset in ['FinDroid', 'Fin_Sitting', 'Fin_SitRun']:
        width = 150
        input_shape = (6, 1, 150)

        # kwargs = {
        #     'in_channel': 6,
        #     'out1_channel': 32,
        #     'out2_channel': 64,
        #     'out3_channel': 128,
        #     'out4_channel': 256,
        #     'out_classes': 6,
        #     'kernel_size': 14,
        #     'avg_factor': 2
        # }

        kwargs = {
            'in_channel': 6,
            'out1_channel': 6,
            'out2_channel': 6,
            'out3_channel': 11,
            'out4_channel': 27,
            'out_classes': 6,
            'kernel_size': 14,
            'avg_factor': 2
        }

        ff = obtain_ff(kwargs, width)
        kwargs['avg_factor'] = ff

    elif dataset == 'HUA':
        width = 200
        input_shape = (7, 1, 200)
        path_to_model= "/Users/zber/ProgramDev/pytorch-pruning/data/HUA/model.pt"
        kwargs = {
            'in_channel': 7,
            'out1_channel': 32,
            'out2_channel': 64,
            'out3_channel': 128,
            'out4_channel': 256,
            'out_classes': 3,
            'kernel_size': 10,
            'avg_factor': 2
        }

        ff = obtain_ff(kwargs, width)
        kwargs['avg_factor'] = ff

    model = NetS(**kwargs)

    # model.load_state_dict(torch.load(path_to_model, map_location=lambda storage, loc: storage))
    # model.load_state_dict(torch.load("/Users/zber/ProgramDev/pytorch-pruning/src/mobile_fullsize.pt"))
    model.load_state_dict(torch.load("/Users/zber/ProgramDev/pytorch-pruning/src/mobile_prunedsize.pt"))

    if args.use_cuda:
        model = model.cuda()

    # create fine tuner object
    fine_tuner = PrunningFineTuner_VGG16(trainloader, testloader, model)

    # test before training
    fine_tuner.test()

    fine_tuner.train(epoches=config.epoch)
    # torch.save(model.state_dict(), path_to_model)

    # prune
    # fine_tuner.prune()

    # save model
    # fine_tuner.save_model("mobile_prunedsize.pt")


