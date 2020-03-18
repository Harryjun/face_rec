# encoding=utf-8
## 部署测试代码

from __future__ import print_function
import os
import cv2 as cv
from models.focal_loss import *
from models.metrics import *
from models.resnet import *
import torch
import numpy as np
import time
import torch.nn.functional as F
from config.config import *  # Config
from torch.nn import DataParallel
from socket import *
import socket
import subprocess
import os
import sys
import time
import torch.nn as nn
import json

reload(sys)

sys.setdefaultencoding('utf8')
from torch.nn import Parameter

data_label_path = "./data_label.txt"
f = open(data_label_path, 'r')
data_label = []
for line in f.readlines():
    data = line.split(' ')
    data_label.append(data[0])
print(data_label)
print(len(data_label))


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
    image = cv.imread(img_path)
    image = cv.resize(image, (128, 128))
    if image is None:
        return None
    # image = np.dstack((image, np.fliplr(image)))
    print(image.shape)
    image = image.transpose((2, 0, 1))
    image = image[np.newaxis, :, :, :]
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
    pretrained_dict = torch.load("./resnet18_0.pth", map_location=device)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    metric_fc_dict = metric_fc.state_dict()
    pretrained_dict1 = torch.load("./checkpoints/arcface_196.pth", map_location=device)
    pretrained_dict1 = {k: v for k, v in pretrained_dict1.items() if k in metric_fc_dict}
    metric_fc_dict.update(pretrained_dict1)
    metric_fc.load_state_dict(metric_fc_dict)

    # 1.创建套接字 socket
    tcp_sever_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # 2.绑定本地信息 bind
    tcp_sever_socket.bind(("", 90))  # 7890))
    # 3.让默认的套接字由主动变为被动 listen
    tcp_sever_socket.listen(128)
    # 4.等待客户端的链接 accept
    while True:
        print('wait')
        new_client_socket, client_addr = tcp_sever_socket.accept()
        print('connect!')
        # img_dir = "./test1.jpg"#
        # img_dir = "./input0/politicalfacetrain1align/caiqi/9ae59669d423cd92.jpg"#19
        url_image = new_client_socket.recv(1024).decode("utf-8")
        print("客户端%s待处理的文件是：%s" % (str(client_addr), url_image))
        # try:
        file_time = time.time()
        subprocess.call('wget -P ./tmp/%s/ %s' % (file_time, url_image), shell=True)
        file_content = None
        index = os.listdir('./tmp/%s/' % file_time)
        img_dir = os.path.join('./tmp/%s/' % file_time, index[0])
        ## opencv face detect
        detector = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
        detector2 = cv.CascadeClassifier('haarcascade_profileface.xml')
        firsttime = time.time()
        img = cv.imread(img_dir)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        print("gray")
        faces = detector2.detectMultiScale(gray, 1.3, 5)
        faces2 = detector.detectMultiScale(gray, 1.3, 5)
        count = 0
        print("face", faces2)
        for (x, y, w, h) in faces2:
            imgs = img[y:y + h, x:x + w]
            img_dir = os.path.join('./tmp/%s/' % file_time, (str(count) + '.jpg'))
            cv.imwrite(img_dir, imgs)
            count += 1
            print(count)
            # cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        # cv.imwrite(img_dir,img)
        print("save")
        json_out = {}
        json_out['img_url'] = url_image
        # json_out["object"] = {}
        objects = []
        print(count)
        images = None
        for index in range(count):
            img_dir = os.path.join('./tmp/%s/' % file_time, (str(index) + '.jpg'))
            print("load")
            image = load_image(img_dir)
            if images is None:
                images = image
            else:
                images = np.concatenate((images, image), axis=0)
        image = images
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
            print(output1.shape)
            max_val = np.argmax(output1, 1)
            print(max_val)
            for i in range(len(max_val)):
                data = {}
                data['class'] = str(data_label[max_val[i]])
                data['pos'] = {'x': int(faces2[i][0]), 'y': int(faces2[i][1]), 'w': int(faces2[i][2]),
                               'h': int(faces2[i][3])}
                data['grade'] = float(output1[i][max_val[i]])
                objects.append(data)
        print("ok")
        json_out['object'] = objects
        # new_client_socket.send(str(data_label[max_val[0]]).encode("utf-8"))
        print(json_out)
        str_data = json.dumps(json_out)
        print(str_data)
        new_client_socket.send(str_data.encode("utf-8"))
        print(time.time() - firsttime)
        # except:
        #    new_client_socket.send("ERROR".encode("utf-8"))
        #    print("Error!")
    # 6.关闭套接字
    new_client_socket.close()
    tcp_sever_socket.close()

