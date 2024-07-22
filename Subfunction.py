import torch
import numpy
import torch.nn as nn
import random

device = 'cuda'
patch_length = 64
l_train_image_type = 4

class Net_hid2_1(nn.Module):
    def __init__(self, n_input, n_hidden1, n_hidden2, n_output):
        super(Net_hid2_1, self).__init__()
        self.linear_ANN_stack = nn.Sequential(
            nn.Linear(n_input, n_hidden1),
            nn.LeakyReLU(),
            nn.Linear(n_hidden1, n_hidden2),
            nn.LeakyReLU(),
            nn.Linear(n_hidden2, n_output),
            nn.ReLU()
        )

    def forward(self, a):
        logits = self.linear_ANN_stack(a)
        return logits


class Net_hid3_1(nn.Module):
    def __init__(self, n_input, n_hidden1, n_hidden2, n_hidden3, n_output):
        super(Net_hid3_1, self).__init__()
        self.linear_ANN_stack = nn.Sequential(
            nn.Linear(n_input, n_hidden1),
            nn.LeakyReLU(),
            nn.Linear(n_hidden1, n_hidden2),
            nn.LeakyReLU(),
            nn.Linear(n_hidden2, n_hidden3),
            nn.LeakyReLU(),
            nn.Linear(n_hidden3, n_output),
            nn.ReLU()
        )

    def forward(self, a):
        logits = self.linear_ANN_stack(a)
        return logits


def assemble_from_patches(num, patches, return_dict):
    image_thick = int(patches.size()[0] / (253 * 253) + 3)
    return_image = torch.zeros([image_thick, 256, 256])
    n = 0
    for i in range(image_thick - 3):
        for j in range(253):
            for k in range(253):
                return_image[i:i+4, j:j+4, k:k+4] = patches[n, :].reshape(4, 4, 4)
                n += 1
    return_dict[num] = return_image


def training_nor(input_patch, label_patch, normalization_factor):
    print('input', input_patch.size())
    print('label', label_patch.size())
    print('max', torch.max(input_patch))
    print('min', torch.min(input_patch))
    print('max', torch.max(label_patch))
    print('min', torch.min(label_patch))
    train_max, index = torch.max(input_patch, dim=1)
    train_max += normalization_factor
    for i in range(input_patch.size(0)):
        input_patch[i, :] = input_patch[i, :] / train_max[i]
    for i in range(label_patch.size(0)):
        label_patch[i, :] = label_patch[i, :] / train_max[i]
    # train_mean = torch.mean(input_patch, dim=1)
    # for i in range(input_patch.size(0)):
    #     input_patch[i, :] = (input_patch[i, :] - train_mean[i]) / 2 + 0.5
    # for i in range(label_patch.size(0)):
    #     label_patch[i, :] = (label_patch[i, :] - train_mean[i]) / 2 + 0.5
    print('max', torch.max(input_patch))
    print('min', torch.min(input_patch))
    print('max', torch.max(label_patch))
    print('min', torch.min(label_patch))

    return input_patch, label_patch


def training_nor_2(input_patch, label_patch, normalization_factor):
    print('before normalization')
    print('max', torch.max(input_patch))
    print('min', torch.min(input_patch))
    print('max', torch.max(label_patch))
    print('min', torch.min(label_patch))

    input_patch /= normalization_factor
    label_patch /= normalization_factor
    # train_mean = torch.mean(input_patch, dim=1)
    # for i in range(input_patch.size(0)):
    #     input_patch[i, :] = (input_patch[i, :] - train_mean[i]) / 2 + 0.5
    # for i in range(label_patch.size(0)):
    #     label_patch[i, :] = (label_patch[i, :] - train_mean[i]) / 2 + 0.5
    print('after normalization')
    print('max', torch.max(input_patch))
    print('min', torch.min(input_patch))
    print('max', torch.max(label_patch))
    print('min', torch.min(label_patch))

    return input_patch, label_patch


def network_128to64():
    model = Net_hid3_1(128, 256, 192, 96, 64)
    return model


def network_128to8():
    hiddenlayers1 = 32
    hiddenlayers2 = 128
    hiddenlayers3 = 512
    model = Net_hid3_1(128, hiddenlayers3, hiddenlayers2, hiddenlayers1, 8)
    return model


def network_1024to512():
    model = Net_hid2_1(1024, 768, 768, 512)
    return model
