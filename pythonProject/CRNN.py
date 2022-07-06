import torch
import os, json
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
import torch.optim
from torch import Tensor
from torch.nn import CTCLoss
from torchvision.transforms import Compose
import string
from collections import OrderedDict
from tqdm import tqdm_notebook, tqdm
import click
import editdistance
import glob
import pandas as pd

def text_collate(batch):
    img = list()
    seq = list()
    seq_len = list()
    for sample in batch:
        img.append(torch.from_numpy(sample["img"].transpose((2, 0, 1))).float())
        seq.extend(sample["seq"])
        seq_len.append(sample["seq_len"])
    img = torch.stack(img)
    seq = torch.Tensor(seq).int()
    seq_len = torch.Tensor(seq_len).int()
    batch = {"img": img, "seq": seq, "seq_len": seq_len}
    return batch

class ToTensor(object):
    def __call__(self, sample):
        sample["img"] = torch.from_numpy(sample["img"].transpose((2, 0, 1))).float()
        sample["seq"] = torch.Tensor(sample["seq"]).int()
        return sample


class Resize(object):
    def __init__(self, size=(320, 32)):
        self.size = size

    def __call__(self, sample):
        sample["img"] = cv2.resize(sample["img"], self.size)
        return sample


class Rotation(object):
    def __init__(self, angle=5, fill_value=0, p = 0.5):
        self.angle = angle
        self.fill_value = fill_value
        self.p = p

    def __call__(self, sample):
        if np.random.uniform(0.0, 1.0) < self.p or not sample["aug"]:
            return sample
        h,w,_ = sample["img"].shape
        #print(sample["img"].shape)
        ang_rot = np.random.uniform(self.angle) - self.angle/2  # 0-5之间随机采样并减2.5
        transform = cv2.getRotationMatrix2D((w/2, h/2), ang_rot, 1)     # 数据增强函数，绕图像中心点旋转，旋转ang_rot角度，无缩放
        sample["img"] = cv2.warpAffine(sample["img"], transform, (w,h), borderValue = self.fill_value)  # 对img进行仿射变换，变换方式为transform，原图大小，边界填充值为0
        return sample


class Translation(object):
    def __init__(self, fill_value=0, p = 0.5):
        self.fill_value = fill_value
        self.p = p

    def __call__(self, sample):
        if np.random.uniform(0.0, 1.0) < self.p or not sample["aug"]:
            return sample
        h,w,_ = sample["img"].shape
        trans_range = [w / 10, h / 10]
        tr_x = trans_range[0]*np.random.uniform()-trans_range[0]/2
        tr_y = trans_range[1]*np.random.uniform()-trans_range[1]/2
        transform = np.float32([[1,0, tr_x], [0,1, tr_y]])
        sample["img"] = cv2.warpAffine(sample["img"], transform, (w,h), borderValue = self.fill_value)
        return sample


class Scale(object):
    def __init__(self, scale=[0.5, 1.2], fill_value=0, p = 0.5):
        self.scale = scale
        self.fill_value = fill_value
        self.p = p

    def __call__(self, sample):
        if np.random.uniform(0.0, 1.0) < self.p or not sample["aug"]:
            return sample
        h, w, _ = sample["img"].shape
        scale = np.random.uniform(self.scale[0], self.scale[1])     # 0.5-1.2间随机抽样
        transform = np.float32([[scale, 0, 0],[0, scale, 0]])
        sample["img"] = cv2.warpAffine(sample["img"], transform, (w,h), borderValue = self.fill_value)
        return sample

class TextDataset(Dataset):
    def __init__(self, data_path, data_label, transform=None):
        super().__init__()
        self.data_path = data_path
        self.data_label = data_label
        self.transform = transform

    def abc_len(self):
        return len('0123456789')

    def get_abc(self):
        return '0123456789'

    def set_mode(self, mode):
        self.mode = mode

    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, idx):
        text = self.data_label[idx]     # 读取标签中的文字

        img = cv2.imread(self.data_path[idx])
        seq = self.text_to_seq(text)    # 得到text所串成的list
        sample = {"img": img, "seq": seq, "seq_len": len(seq), "aug": 1}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def text_to_seq(self, text):
        seq = []
        for c in text:
            seq.append(self.get_abc().find(str(c)) + 1)     # 循环找到text在0-9中的索引并形成列表
        return seq

class CRNN(nn.Module):
    def __init__(self,
                 abc='0123456789',
                 backend='resnet18',
                 rnn_hidden_size=256,
                 rnn_num_layers=1,
                 rnn_dropout=0,
                 seq_proj=[0, 0]):
        super(CRNN, self).__init__()

        self.abc = abc
        self.num_classes = len(self.abc)

        self.feature_extractor = getattr(models, backend)(pretrained=True)  # 将resnet18得到
        self.cnn = nn.Sequential(
            self.feature_extractor.conv1,
            self.feature_extractor.bn1,
            self.feature_extractor.relu,
            self.feature_extractor.maxpool,
            self.feature_extractor.layer1,
            self.feature_extractor.layer2,
            self.feature_extractor.layer3,
            self.feature_extractor.layer4
        )

        self.fully_conv = seq_proj[0] == 0
        if not self.fully_conv:
            self.proj = nn.Conv2d(seq_proj[0], seq_proj[1], kernel_size=1)

        self.rnn_hidden_size = rnn_hidden_size
        self.rnn_num_layers = rnn_num_layers
        self.rnn = nn.GRU(self.get_block_size(self.cnn),
                          rnn_hidden_size, rnn_num_layers,
                          batch_first=False,
                          dropout=rnn_dropout, bidirectional=True)
        self.linear = nn.Linear(rnn_hidden_size * 2, self.num_classes + 1)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x, decode=False):
        hidden = self.init_hidden(x.size(0), next(self.parameters()).is_cuda)
        features = self.cnn(x)  # 得到特征图
        features = self.features_to_sequence(features)  # 得到特征序列
        seq, hidden = self.rnn(features, hidden)
        seq = self.linear(seq)
        if not self.training:
            seq = self.softmax(seq)
            if decode:
                seq = self.decode(seq)
        return seq

    def init_hidden(self, batch_size, gpu=False):
        h0 = Variable(torch.zeros( self.rnn_num_layers * 2,
                                   batch_size,
                                   self.rnn_hidden_size))
        if gpu:
            h0 = h0.cuda()
        return h0

    def features_to_sequence(self, features):
        features = features.mean(2)
        b, c, w = features.size()
        features = features.reshape(b, c, 1, w)
        b, c, h, w = features.size()
        #print(b, c, h, w)   # 20 512 1 7
        assert h == 1, "the height of out must be 1"
        if not self.fully_conv:
            features = features.permute(0, 3, 2, 1)  #维度换位：b,w,h,c
            features = self.proj(features)
            features = features.permute(1, 0, 2, 3)
        else:
            features = features.permute(3, 0, 2, 1)  #w为时间序列，b为bitch_size，c为通道数
        features = features.squeeze(2)
        return features

    def get_block_size(self, layer):
        return layer[-1][-1].bn2.weight.size()[0]

    def pred_to_string(self, pred):
        seq = []
        for i in range(pred.shape[0]):
            label = np.argmax(pred[i])
            seq.append(label - 1)
        out = []
        for i in range(len(seq)):
            if len(out) == 0:
                if seq[i] != -1:
                    out.append(seq[i])
            else:
                if seq[i] != -1 and seq[i] != seq[i - 1]:
                    out.append(seq[i])
        out = ''.join(self.abc[i] for i in out)
        return out

    def decode(self, pred):
        pred = pred.permute(1, 0, 2).cpu().data.numpy()
        seq = []
        for i in range(pred.shape[0]):
            seq.append(self.pred_to_string(pred[i]))
        return seq


def load_weights(target, source_state):
    new_dict = OrderedDict()
    for k, v in target.state_dict().items():
        if k in source_state and v.size() == source_state[k].size():
            new_dict[k] = source_state[k]
        else:
            new_dict[k] = v
    target.load_state_dict(new_dict)


def load_model(abc, device, seq_proj=[0, 0], backend='resnet18', snapshot=None):
    net = CRNN(abc=abc, seq_proj=seq_proj, backend=backend)
    # net = nn.DataParallel(net)
    if snapshot is not None:
        load_weights(net, torch.load(snapshot))     # 载入权重
    #if cuda:
    net = net.to(device)
    return net

class StepLR(object):
    def __init__(self, optimizer, step_size=1000, max_iter=10000):
        self.optimizer = optimizer
        self.max_iter = max_iter
        self.step_size = step_size
        self.last_iter = -1
        self.base_lrs = list(map(lambda group: group['lr'], optimizer.param_groups))    # 得到初始学习率

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']

    def step(self, last_iter=None):
        if last_iter is not None:
            self.last_iter = last_iter
        if self.last_iter + 1 == self.max_iter:
            self.last_iter = -1
        self.last_iter = (self.last_iter + 1) % self.max_iter
        for ids, param_group in enumerate(self.optimizer.param_groups):
            param_group['lr'] = self.base_lrs[ids] * 0.8 ** ( self.last_iter // self.step_size )    # 调整学习率


def test(net, data, abc, device, batch_size=50):
    data_loader = DataLoader(data, batch_size=batch_size, num_workers=4, shuffle=False, collate_fn=text_collate)

    count = 0
    tp = 0
    avg_ed = 0
    iterator = tqdm_notebook(data_loader)   # 显示进度条
    for sample in iterator:
        imgs = Variable(sample["img"])
        #if cuda:
        imgs = imgs.to(device)
        out = net(imgs, decode=True)
        gt = (sample["seq"].numpy() - 1).tolist()
        lens = sample["seq_len"].numpy().tolist()
        pos = 0
        key = ''
        for i in range(len(out)):
            gts = ''.join(abc[c] for c in gt[pos:pos+lens[i]])
            pos += lens[i]
            if gts == out[i]:
                tp += 1
            else:
                avg_ed += editdistance.eval(out[i], gts)
            count += 1
        iterator.set_description("acc: {0:.4f}; avg_ed: {0:.4f}".format(tp / count, avg_ed / count))

    acc = tp / count
    avg_ed = avg_ed / count
    return acc, avg_ed


train_json = json.load(open('../input/mchar_train.json'))
train_label = [train_json[x]['label'] for x in train_json.keys()]   # 获得json中键为label的值，x为图片名，['label']为该图片下label值
train_path = ['../input/mchar_train/mchar_train/' + x for x in train_json.keys()]   # 获得图像的路径，x为图片名

val_json = json.load(open('../input/mchar_val.json'))
val_label = [val_json[x]['label'] for x in val_json.keys()]
val_path = ['../input/mchar_val/mchar_val/' + x for x in val_json.keys()]


def main(
        abc='0123456789',
        seq_proj="7x30",
        backend="resnet18",
        snapshot=None,
        input_size="200x100",   # x为分隔符
        base_lr=1e-3,
        step_size=1000,
        max_iter=15000,
        batch_size=20,
        output_dir='./',
        test_epoch=1,
        test_init=None,
        gpu='0'):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    #cuda = True if gpu is not '' else False
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    input_size = [int(x) for x in input_size.split('x')]    # x是分隔符，将200，100分开，这里获得200，100并转为列表[200,100]
    transform = Compose([
        Rotation(),  # 做旋转
        Translation(),  # 平移
        # Scale(),
        Resize(size=(input_size[0], input_size[1]))
    ])

    data = TextDataset(train_path, train_label, transform=transform)
    data_val = TextDataset(val_path, val_label, transform=transform)

    seq_proj = [int(x) for x in seq_proj.split('x')]    # [7,30]
    net = load_model(data.get_abc(), device, seq_proj, backend, snapshot )     # 下载model
    optimizer = torch.optim.Adam(net.parameters(), lr=base_lr, weight_decay=0.0001)  # 优化器
    lr_scheduler = StepLR(optimizer, step_size=step_size, max_iter=max_iter)    # 等间隔调整学习率
    loss_function = CTCLoss(zero_infinity=True)

    acc_best = 0
    epoch_count = 0
    while epoch_count < 100:
        if (test_epoch is not None and epoch_count != 0 and epoch_count % test_epoch == 0) or (
                test_init and epoch_count == 0):
            print("Test phase")
            data.set_mode("test")
            net = net.eval()
            acc, avg_ed = test(net, data_val, data.get_abc(), device, 50)
            net = net.train()
            data.set_mode("train")
            if acc > acc_best:
                if output_dir is not None:
                    torch.save(net.state_dict(),
                               os.path.join(output_dir, "crnn_" + backend + "_" + str(data.get_abc()) + "_best"))
                acc_best = acc
            print("acc: {}\tacc_best: {}; avg_ed: {}".format(acc, acc_best, avg_ed))

        data_loader = DataLoader(data, batch_size=batch_size, num_workers=1, shuffle=True, collate_fn=text_collate)     # text_collate将图片以及标签合为列表并以字典保存
        loss_mean = []
        iterator = tqdm(data_loader)
        iter_count = 0
        for sample in iterator:
            # for multi-gpu support
            if sample["img"].size(0) % len(gpu.split(',')) != 0:
                continue
            optimizer.zero_grad()
            imgs = Variable(sample["img"])  # 将张量转为变量在gpu中加载
            labels = Variable(sample["seq"]).view(-1)
            label_lens = Variable(sample["seq_len"].int())
            #if device:
            imgs = imgs.to(device)
            preds = net(imgs).cpu()
            pred_lens = Variable(Tensor([preds.size(0)] * batch_size).int())

            # print(preds.shape, labels.shape)
            loss = loss_function(preds, labels, pred_lens, label_lens)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 10.0)
            loss_mean.append(loss.item())
            status = "epoch: {}; iter: {}; lr: {}; loss_mean: {}; loss: {}".format(epoch_count, lr_scheduler.last_iter,
                                                                                   lr_scheduler.get_lr(),
                                                                                   np.mean(loss_mean), loss.item())
            iterator.set_description(status)
            optimizer.step()
            lr_scheduler.step()
            iter_count += 1
        if output_dir is not None:
            torch.save(net.state_dict(),
                       os.path.join(output_dir, "crnn_" + backend + "_" + str(data.get_abc()) + "_last"))
        epoch_count += 1

    #return 1


    test_path = glob.glob('C:/Users/mine/Desktop/pytorch test/jiejingzifushibie/input/mchar_test_a/mchar_test_a/*.png')
    test_label = [[1]] * len(test_path)
    test_path.sort()    # list升序排序

    def predict(net, data, abc, device, visualize, batch_size=50):
        data_loader = DataLoader(data, batch_size=batch_size, num_workers=4, shuffle=False, collate_fn=text_collate)

        count = 0
        tp = 0
        avg_ed = 0
        out = []
        iterator = tqdm_notebook(data_loader)
        for sample in iterator:
            imgs = Variable(sample["img"])
            #if cuda:
            imgs = imgs.to(device)
            out += net(imgs, decode=True)
            # print(out)
            # break
        return out

    model = load_model('0123456789', device, seq_proj=[7, 30], backend='resnet18', snapshot='crnn_resnet18_0123456789_best')

    transform = Compose([
        # Rotation(),
        # Translation(),
        # Scale(),
        Resize(size=(200, 100))
        ])
    test_data = TextDataset(test_path, test_label, transform=transform)
    #print(test_data)

    model.training = False
    test_predict = predict(model, test_data, '0123456789', device, False, batch_size=50)
    #print(test_predict)
    df_submit = pd.read_csv('../input/mchar_sample_submit_A.csv')
    #print(df_submit)
    df_submit['file_code'] = test_predict
    df_submit.to_csv('submit.csv', index=None)


if __name__ == '__main__':
    main()



