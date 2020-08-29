import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models
# import cv2
import sys
import numpy as np
import time
import math


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def replace_layers(model, i, indexes, layers):
    if i in indexes:
        return layers[indexes.index(i)]
    return model[i]


def prune_vgg16_conv_layer(model, layer_index, filter_index, use_cuda=False):

    # replace batchnormal layer
    model = replace_bn_layers(model, layer_index, filter_index)

    _, conv = list(model.features._modules.items())[layer_index]
    next_conv = None
    offset = 1

    while layer_index + offset < len(model.features._modules.items()):
        res = list(model.features._modules.items())[layer_index + offset]
        if isinstance(res[1], torch.nn.modules.conv.Conv2d):
            next_name, next_conv = res
            break
        offset = offset + 1

    new_conv = \
        torch.nn.Conv2d(in_channels=conv.in_channels, \
                        out_channels=conv.out_channels - 1,
                        kernel_size=conv.kernel_size, \
                        stride=conv.stride,
                        padding=conv.padding,
                        dilation=conv.dilation,
                        groups=conv.groups,
                        bias=(conv.bias is not None))

    old_weights = conv.weight.data.cpu().numpy()
    new_weights = new_conv.weight.data.cpu().numpy()

    new_weights[: filter_index, :, :, :] = old_weights[: filter_index, :, :, :]
    new_weights[filter_index:, :, :, :] = old_weights[filter_index + 1:, :, :, :]
    new_conv.weight.data = torch.from_numpy(new_weights)
    if use_cuda:
        new_conv.weight.data = new_conv.weight.data.cuda()

    bias_numpy = conv.bias.data.cpu().numpy()

    bias = np.zeros(shape=(bias_numpy.shape[0] - 1), dtype=np.float32)
    bias[:filter_index] = bias_numpy[:filter_index]
    bias[filter_index:] = bias_numpy[filter_index + 1:]
    new_conv.bias.data = torch.from_numpy(bias)
    if use_cuda:
        new_conv.bias.data = new_conv.bias.data.cuda()

    if not next_conv is None:
        next_new_conv = \
            torch.nn.Conv2d(in_channels=next_conv.in_channels - 1, \
                            out_channels=next_conv.out_channels, \
                            kernel_size=next_conv.kernel_size, \
                            stride=next_conv.stride,
                            padding=next_conv.padding,
                            dilation=next_conv.dilation,
                            groups=next_conv.groups,
                            bias=(next_conv.bias is not None))

        old_weights = next_conv.weight.data.cpu().numpy()
        new_weights = next_new_conv.weight.data.cpu().numpy()

        new_weights[:, : filter_index, :, :] = old_weights[:, : filter_index, :, :]
        new_weights[:, filter_index:, :, :] = old_weights[:, filter_index + 1:, :, :]
        next_new_conv.weight.data = torch.from_numpy(new_weights)
        if use_cuda:
            next_new_conv.weight.data = next_new_conv.weight.data.cuda()

        next_new_conv.bias.data = next_conv.bias.data

    if not next_conv is None:
        features = torch.nn.Sequential(
            *(replace_layers(model.features, i, [layer_index, layer_index + offset], \
                             [new_conv, next_new_conv]) for i, _ in enumerate(model.features)))
        del model.features
        del conv

        model.features = features

    else:
        # Prunning the last conv layer. This affects the first linear layer of the classifier.
        model.features = torch.nn.Sequential(
            *(replace_layers(model.features, i, [layer_index], \
                             [new_conv]) for i, _ in enumerate(model.features)))
        layer_index = 0
        old_linear_layer = None
        for _, module in model.classifier._modules.items():
            if isinstance(module, torch.nn.Linear):
                old_linear_layer = module
                break
            layer_index = layer_index + 1

        if old_linear_layer is None:
            raise BaseException("No linear laye found in classifier")
        params_per_input_channel = old_linear_layer.in_features // conv.out_channels

        new_linear_layer = \
            torch.nn.Linear(old_linear_layer.in_features - params_per_input_channel,
                            old_linear_layer.out_features)

        old_weights = old_linear_layer.weight.data.cpu().numpy()
        new_weights = new_linear_layer.weight.data.cpu().numpy()

        new_weights[:, : filter_index * params_per_input_channel] = \
            old_weights[:, : filter_index * params_per_input_channel]
        new_weights[:, filter_index * params_per_input_channel:] = \
            old_weights[:, (filter_index + 1) * params_per_input_channel:]

        new_linear_layer.bias.data = old_linear_layer.bias.data

        new_linear_layer.weight.data = torch.from_numpy(new_weights)
        if use_cuda:
            new_linear_layer.weight.data = new_linear_layer.weight.data.cuda()

        classifier = torch.nn.Sequential(
            *(replace_layers(model.classifier, i, [layer_index], \
                             [new_linear_layer]) for i, _ in enumerate(model.classifier)))

        del model.classifier
        del next_conv
        del conv
        model.classifier = classifier

    return model


def replace_bn_layers(model, layer_index, filter_index):
    del_num = 1
    _, current_layer  = list(model.features._modules.items())[layer_index+1]
    new_num_features = current_layer.num_features - del_num

    new_bn = nn.BatchNorm2d(
        num_features=new_num_features,
        eps=current_layer.eps,
        momentum=current_layer.momentum,
        affine=current_layer.affine,
        track_running_stats=current_layer.track_running_stats
    )

    # bias
    new_bn.bias.data[:filter_index] = current_layer.bias.data[:filter_index]
    new_bn.bias.data[filter_index:] = current_layer.bias.data[filter_index+1:]

    # weight
    new_bn.weight.data[:filter_index] = current_layer.weight.data[:filter_index]
    new_bn.weight.data[filter_index:] = current_layer.weight.data[filter_index+1:]

    # find old layer's position
    seq, index = find_seq_index(model, current_layer)
    seq[index] = new_bn

    return model


def find_seq(model):
    seqs = []
    for module in model.modules():
        if isinstance(module, torch.nn.modules.Sequential):
            seqs.append(module)
    return seqs


def find_seq_index(model, layer):
    sequence = None
    index = None
    for seq in model.modules():
        if isinstance(seq, nn.Sequential):
            if layer in seq:
                sequence = seq
                for i in range(len(seq)):
                    if layer is seq[i]:
                        index = i
                        return sequence, index


def find_layers(model):
    layers = []
    for module in model.modules():
        if isinstance(module, torch.nn.modules.Conv2d) or isinstance(module, torch.nn.modules.Linear):
            layers.append(module)
    return layers


def del_cov_filter(old_conv_layer, filter_index, is_first=True):
    if old_conv_layer is None:
        raise BaseException("No Conv layer found in classifier")

    # delete filter number is 1
    del_num = 1

    old_weights = old_conv_layer.weight.data.cpu()

    # create new_conv_layer
    if is_first:
        new_conv_layer = torch.nn.Conv2d(in_channels=(old_conv_layer.in_channels - del_num) if old_conv_layer.groups > 1 else old_conv_layer.in_channels,
                                         out_channels=old_conv_layer.out_channels - del_num,
                                         kernel_size=old_conv_layer.kernel_size,
                                         stride=old_conv_layer.stride,
                                         padding=old_conv_layer.padding,
                                         dilation=old_conv_layer.dilation,
                                         groups=(old_conv_layer.groups - del_num) if old_conv_layer.groups > 1 else old_conv_layer.groups,
                                         bias=(old_conv_layer.bias is not None))

        # depth-wise conv layer
        new_conv_layer.weight.data[:filter_index] = old_weights[:filter_index]
        new_conv_layer.weight.data[filter_index:] = old_weights[filter_index + 1:]

    else:
        new_conv_layer = torch.nn.Conv2d(in_channels=old_conv_layer.in_channels - del_num,
                                         out_channels=old_conv_layer.out_channels,
                                         kernel_size=old_conv_layer.kernel_size,
                                         stride=old_conv_layer.stride,
                                         padding=old_conv_layer.padding,
                                         dilation=old_conv_layer.dilation,
                                         groups=(old_conv_layer.groups - del_num) if old_conv_layer.groups > 1 else old_conv_layer.groups,
                                         bias=(old_conv_layer.bias is not None))

        # copy new weights to new_conv_layer
        new_conv_layer.weight.data[:, :filter_index] = old_weights[:, :filter_index]
        new_conv_layer.weight.data[:, filter_index:] = old_weights[:, filter_index + 1:]

    return new_conv_layer


def del_linear_filter(old_linear_layer, filter_index):
    # delete filter number is 1
    del_num = 1

    new_linear_layer = torch.nn.Linear(old_linear_layer.in_features - del_num,
                                       old_linear_layer.out_features)

    new_linear_layer.weight.data[:, :filter_index] = old_linear_layer.weight.data[:, :filter_index]
    new_linear_layer.weight.data[:, filter_index:] = old_linear_layer.weight.data[:, filter_index + 1:]

    return new_linear_layer


def prune_mbnet_layer(model, layer_index, filter_index):
    # find layers
    layers = find_layers(model)

    # convert layer index into correct format
    layer_index = layer_index * 2

    old_layer = layers[layer_index]
    old_layer2 = None
    old_layer3 = None

    # first
    new_layer = del_cov_filter(old_layer, filter_index, is_first=True)

    if layer_index < len(layers) - 2:
        # select next two layers which are needed to adapt to new size.
        old_layer2 = layers[layer_index + 1]
        old_layer3 = layers[layer_index + 2]

        new_layer2 = del_cov_filter(old_layer2, filter_index, is_first=True)

        # third
        new_layer3 = del_cov_filter(old_layer3, filter_index, is_first=False)

        old_layers = [old_layer, old_layer2, old_layer3]
        new_layers = [new_layer, new_layer2, new_layer3]

    else:
        # second fc
        old_layer2 = layers[layer_index + 1]
        new_layer2 = del_linear_filter(old_layer2, filter_index)

        old_layers = [old_layer]
        new_layers = [new_layer]

        model.fc = new_layer2

    # then replace old layers by new layers
    for old, new in zip(old_layers, new_layers):
        seq, i = find_seq_index(model, old)
        seq[i] = new

    del old_layer
    del old_layer2
    del old_layer3

    return model


if __name__ == '__main__':
    model = models.vgg16(pretrained=True)
    model.train()

    t0 = time.time()
    model = prune_vgg16_conv_layer(model, 28, 10)
    print("The prunning took", time.time() - t0)
