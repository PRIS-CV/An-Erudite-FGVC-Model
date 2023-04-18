from __future__ import print_function
import os
import sys

import torch

import argparse
import torchvision
import random
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import learn2learn as l2l
import torchvision.models as models

parser = argparse.ArgumentParser(description='Mix-dataset FGVC (car flower)')

parser.add_argument('--folder_log', default="/home/changdongliang/mix-data-training/log/final-test", type=str, help='log_path')
parser.add_argument('--data_dir', default="/home/changdongliang/data/", type=str, help='data_path')

args = parser.parse_args()


if not os.path.exists(args.folder_log):
    os.makedirs(args.folder_log)
    
print('\n')    
print('TRAIN START!')
print('\n')
print('THE OUTPUT IS SAVED IN A TXT FILE HERE -------------------------------------------> ', args.folder_log)
print('\n')

f = open(args.folder_log + '/out.txt', 'w')
sys.stdout = f

group = ["cars", "flowers"]
nb_epoch = 100


use_cuda = torch.cuda.is_available()

print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomCrop(224, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

transform_test = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])



trainset    = torchvision.datasets.ImageFolder(root=args.data_dir + 'StandCars/train', transform=transform_train)
trainloader_2 = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=True, num_workers=16, drop_last = True)



trainset    = torchvision.datasets.ImageFolder(root=args.data_dir + 'Flowers102/train', transform=transform_train)
trainloader_4 = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=True, num_workers=8, drop_last = True)



testset = torchvision.datasets.ImageFolder(root= args.data_dir + 'StandCars/test', transform=transform_test)
testloader_2 = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True, num_workers=8, drop_last = False)


testset = torchvision.datasets.ImageFolder(root= args.data_dir + 'Flowers102/test', transform=transform_test)
testloader_4 = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True, num_workers=8, drop_last = False)

print('==> Building model..')



net = models.resnet50(pretrained=True)
criterion = nn.CrossEntropyLoss()
criterion_no_average = nn.CrossEntropyLoss(reduce=False)





###


class BasicConv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=True):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class SpatialAttention(nn.Module):
    def __init__(self, feature_size, head_num):
        super(SpatialAttention, self).__init__()
        
        self.feature_size = feature_size
        self.head_num = head_num
        
        self.conv = nn.ModuleList(
            [nn.Conv2d(feature_size // head_num, 1, 1, padding=0) for i in range(head_num)]
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        
        x_list = []
        att_list = []
        for i, x_ in enumerate(torch.chunk(x, self.head_num, 1)):
            att = self.conv[i](x_)
            att = self.sigmoid(att)
            att_list.append(att)
            x_list.append(x_ * att)
            
        att = torch.cat(att_list, 1)
        x = torch.cat(x_list, 1)

        return x, att
    
class FusionGate(nn.Module):

    def __init__(self, feature_size):
        super(FusionGate, self).__init__()
        self.num_group = len(group)
        self.feature_size = feature_size
        self.w_gate=nn.Linear(feature_size * self.num_group,self.num_group * feature_size)
        self.softmax = nn.Softmax(dim = 1)

    def forward(self, xcar, xflower):
       #196 + 102
        gate_weight = self.w_gate(torch.cat([xcar,xflower],dim=1)).view(-1, self.num_group, self.feature_size)
        gate_weight = self.softmax(gate_weight)
        x = torch.stack([xcar,xflower],dim=1)


        x = x*gate_weight
        x = x.mean(1)
        return x

class erudite_model(nn.Module):
    def __init__(self, model, feature_size,classes_num):

        super(erudite_model, self).__init__()

        self.num_ftrs = 2048*1*1

        self.features = nn.Sequential(*list(net.children())[:-2])

        self.max = nn.MaxPool2d(kernel_size=7, stride=7)

        self.spatial_attention = SpatialAttention(feature_size=self.num_ftrs, head_num=8)
        
        #what
        self.car_expert = BasicConv(self.num_ftrs, feature_size, kernel_size=1, stride=1)
        self.flower_expert = BasicConv(self.num_ftrs, feature_size, kernel_size=1, stride=1)
        
        #mix_classifier
        self.mix_gate = FusionGate(feature_size)
   
        self.car_classifier = nn.Sequential(
            nn.BatchNorm1d(feature_size),
            nn.Linear(feature_size, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Linear(feature_size, 196),
        )

        self.flower_classifier = nn.Sequential(
            nn.BatchNorm1d(feature_size),
            nn.Linear(feature_size, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Linear(feature_size, 102),
        )
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(feature_size),
            nn.Linear(feature_size, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Linear(feature_size, classes_num),
        )
    def forward(self, x, targets, car_weight=None, flower_weight=None, origin_target=None,maml=False):

     

        x = self.features(x)

        x, att = self.spatial_attention(x)

        #what
        xcar = self.car_expert(x)
        xflower = self.flower_expert(x)
        

        xcar = self.max(xcar).view(xcar.size(0), -1)
        xflower = self.max(xflower).view(xflower.size(0), -1)
        

        #fusion 
        x = self.mix_gate(xcar, xflower)
        x = self.classifier(x)

        #specific expert loss
        loss_car = 0
        loss_flower = 0
        if self.training and not maml:
            predict_car = self.car_classifier(xcar)
            predict_flower = self.flower_classifier(xflower)
     
            target_car = torch.where(origin_target<196,origin_target,0)
            target_flower = torch.where(origin_target<102,origin_target,0)

            
            loss_car = (criterion_no_average(predict_car, target_car)*car_weight).mean()

            loss_flower = (criterion_no_average(predict_flower, target_flower)*flower_weight).mean()
        
            loss_all = criterion(x, targets)
            diverse_loss = (1 - torch.max(att, 1)[0]).mean() + att.mean()
            loss = loss_all + (loss_car  + loss_flower)/len(group) + diverse_loss * 0.1

        else:
            loss = criterion(x, targets)
            
        return x, loss

#build model
net = erudite_model(net, 512, 196 + 102)

if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net)

     
def train(epoch):
    print('\nEpoch: %d' % epoch)
    print(str(group))
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    idx = 0

    
    iter_train_data_2=trainloader_2.__iter__()  #509
    iter_train_data_4=trainloader_4.__iter__()  #127


    batchNum = max(len(trainloader_2.__iter__()), len(trainloader_4.__iter__()))



    for batch_idx in tqdm(range(0,batchNum)):

        #cars
        try:
            inputs_2, targets_2 = iter_train_data_2.__next__()
        except:
            iter_train_data_2 = trainloader_2.__iter__()
            inputs_2, targets_2 = iter_train_data_2.__next__()
        
        #flowers
        try:
            inputs_4, targets_4 = iter_train_data_4.__next__()
        except:
            iter_train_data_4 = trainloader_4.__iter__()
            inputs_4, targets_4 = iter_train_data_4.__next__()
       

        idx = batch_idx

 
        inputs_2, targets_2 =  Variable(inputs_2.cuda()),  Variable(targets_2.cuda())
        inputs_4, targets_4 =  Variable(inputs_4.cuda()),  Variable(targets_4.cuda())

        origin_target = torch.cat([targets_2, targets_4], 0)
        targets_2 += 0
        targets_4 += 196
        inputs = torch.cat([inputs_2, inputs_4], 0)
        targets = torch.cat([targets_2,  targets_4], 0)

        car_weight = torch.where(targets<196,torch.tensor([1]).float().cuda(),torch.tensor([0]).float().cuda()).detach()
        flower_weight = torch.where(targets>=196,torch.tensor([1]).float().cuda(),torch.tensor([0]).float().cuda()).detach()


        if random.random() > 0.5:
            inputs_evaluation = inputs_2
            targets_evaluation = targets_2
            inputs_adaptation = inputs_4
            targets_adaptation = targets_4
        else:
            inputs_evaluation = inputs_4
            targets_evaluation = targets_4
            inputs_adaptation = inputs_2
            targets_adaptation = targets_2

            


        outputs, loss = net(inputs, targets,  car_weight, flower_weight, origin_target)
        optimizer.zero_grad()
        
        #trival training
        loss = loss.mean()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        #maml training
        optimizer_maml.zero_grad()
        task_model = maml_model.clone()  # torch.clone() for nn.Modules
        outputs_1, adaptation_loss =task_model(inputs_adaptation, targets_adaptation, maml=True)
        task_model.adapt(adaptation_loss)  # computes gradient, update task_model in-place
        outputs_2, evaluation_loss = task_model(inputs_evaluation, targets_evaluation, maml=True)
        evaluation_loss.backward()  # gradients w.r.t. maml.parameters()
        optimizer_maml.step()


        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum().item()


       

    train_acc_all = 100.*correct/total
    train_loss = train_loss/(idx+1)
    
    return train_acc_all, train_loss


def test(epoch, balance = 0, dataset="", dataset_name = "birds", flag="test"):

    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    idx = 0
    for batch_idx, (inputs, targets) in enumerate(dataset):
        with torch.no_grad():
            idx = batch_idx
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs), Variable(targets)


            targets = targets + balance
            outputs, loss =net(inputs, targets)

            loss = loss.mean()

            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum().item()

        
    test_acc = 100.*correct/total
    test_loss = test_loss/(idx+1)
    
    if test_acc > max_val[dataset_name]:
        max_val[dataset_name] = test_acc
        torch.save(net.state_dict(), args.folder_log+'/best.pth')
    if flag == "train":
        print("Epoch: {}".format(epoch) + " | train_acc_"+ dataset_name + " :{} ".format(test_acc))
    elif flag == "test":
        print("Epoch: {}".format(epoch) + " | test_acc_"+ dataset_name + " : {}".format(test_acc))
    else:
        pass

    return test_acc

def cosine_anneal_schedule(t):
    cos_inner = np.pi * (t % (nb_epoch  ))  # t - 1 is used when t has 1-based indexing.
    cos_inner /= (nb_epoch )
    cos_out = np.cos(cos_inner) + 1
    return float( 0.01 * cos_out)



optimizer = optim.SGD([
                        {'params': net.module.classifier.parameters(), 'lr': 0.1},
                        {'params': net.module.car_classifier.parameters(), 'lr': 0.1},
                        {'params': net.module.flower_classifier.parameters(), 'lr': 0.1},
                        {'params': net.module.mix_gate.parameters(), 'lr': 0.1},
                        {'params': net.module.car_expert.parameters(), 'lr': 0.1},
                        {'params': net.module.flower_expert.parameters(), 'lr': 0.1},
                        {'params': net.module.features.parameters(),   'lr': 0.01},

                     ],
                      momentum=0.9, weight_decay=5e-4)
optimizer_maml = optim.SGD([
                        {'params': net.module.spatial_attention.parameters(), 'lr': 0.1},
                     ],
                      momentum=0.9, weight_decay=5e-4)

max_val = {}
max_val['cars'] = 0
max_val['flowers'] = 0


for epoch in range(0, nb_epoch):
    for m in range(6):
        optimizer.param_groups[m]['lr'] = cosine_anneal_schedule(epoch)
    optimizer.param_groups[6]['lr'] = cosine_anneal_schedule(epoch) / 10
    optimizer_maml.param_groups[0]['lr'] = cosine_anneal_schedule(epoch) / 10
    maml_model = l2l.algorithms.MAML(net, lr=cosine_anneal_schedule(epoch) / 10, allow_unused=True,first_order=False)
    
    train(epoch)
    

    #birds  Cars  Air  Flowers102
    #200 196 100 102
    print("testing v24")
 
    _ = test(epoch, balance = 0,   dataset = testloader_2,  dataset_name = "cars",     flag="test")
    _ = test(epoch, balance = 196, dataset = testloader_4,  dataset_name = "flowers",  flag="test")

       

print(max_val)


#pass
