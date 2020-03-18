##
## 部署测试代码
from __future__ import print_function
import os
import cv2
from models.focal_loss import *
from models.metrics import *
from models.resnet import *
import torch
import numpy as np
import time
import torch.nn.functional as F
from config.config import *  # Config
from torch.nn import DataParallel
import torch.nn as nn
from torch.nn import Parameter


def get_data_list(pair_list):
    with open(pair_list, 'r') as fd:
        pairs = fd.readlines()
    data_list = []
    for pair in pairs:
        splits = pair.split()

        if splits[0] not in data_list:
            data_list.append(splits[0])

        if splits[1] not in data_list:
            data_list.append(splits[1])
    return data_list


def load_image(img_path):
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (128, 128))
    if image is None:
        return None
    # image = np.dstack((image, np.fliplr(image)))
    print(image.shape)
    # image = image.transpose((2, 0, 1))
    image = image[np.newaxis, np.newaxis, :, :]
    image = image.astype(np.float32, copy=False)
    image -= 127.5
    image /= 127.5
    return image


class Arcface(nn.Module):
    def __init__(self):
        super(Arcface, self).__init__()
        self.weight = Parameter(torch.FloatTensor(opt.num_classes, 512))

    def forward(self, input_):
        metric_fc = F.linear(F.normalize(input_), F.normalize(self.weight))
        return metric_fc


if __name__ == '__main__':

    opt = Config()
    if opt.backbone == 'resnet18':
        model = resnet_face18(opt.use_se)
    elif opt.backbone == 'resnet34':
        model = resnet34()
    elif opt.backbone == 'resnet50':
        model = resnet50()
    metric_fc = Arcface()
    model = DataParallel(model)
    metric_fc = DataParallel(metric_fc)

    device = torch.device('cpu')

    model_dict = model.state_dict()
    pretrained_dict = torch.load(opt.test_model_path, map_location=device)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    # cosine = F.linear(F.normalize(input), F.normalize(self.weight))

    metric_fc_dict = metric_fc.state_dict()
    # print(metric_fc_dict)
    pretrained_dict1 = torch.load("./checkpoints/arcface_196.pth", map_location=device)
    pretrained_dict1 = {k: v for k, v in pretrained_dict1.items() if k in metric_fc_dict}
    metric_fc_dict.update(pretrained_dict1)
    metric_fc.load_state_dict(metric_fc_dict)

    img_dir = "./input0/politicalfacetest1align/lixueju/89a695b364ad722d.jpg"  #
    # img_dir = "./input0/politicalfacetrain1align/caiqi/9ae59669d423cd92.jpg"#19
    image = load_image(img_dir)
    model.eval()
    if image is None:
        print('read {} error'.format(img_dir))
    else:
        data = torch.from_numpy(image)
        data = data.to(torch.device("cpu"))
        print(data.shape)
        output = model(data)
        output1 = metric_fc(output)
        output = output.data.cpu().numpy()
        output1 = output1.data.cpu().numpy()
        print(output.shape)
        # fe_1 = output[::2]
        # fe_2 = output[1::2]
        # feature = np.hstack((fe_1, fe_2))
        print(output1.shape)
        # print(output1)
        max_val = np.argmax(output1, 1)
        print(max_val)
        # print(output1.shape)
        # print(feature.shape)

##data_list
##




