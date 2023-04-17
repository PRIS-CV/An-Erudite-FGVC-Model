from __future__ import print_function
import os
import sys

import torch

import argparse
import torchvision
import random
import numpy as np
import torch.optim as optim
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import learn2learn as l2l
import torchvision.models as models
from model import erudite_model

def cosine_anneal_schedule(t):
    cos_inner = np.pi * (t % (nb_epoch  ))  # t - 1 is used when t has 1-based indexing.
    cos_inner /= (nb_epoch )
    cos_out = np.cos(cos_inner) + 1
    return float( 0.01 * cos_out)  
    
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

parser = argparse.ArgumentParser(description='Mix-dataset FGVC (car flower)')
parser.add_argument('--folder_log', default="/home/changdongliang/mix-data-training/log/final-test1", type=str, help='log_path')
parser.add_argument('--data_dir', default="/home/changdongliang/data/", type=str, help='data_path')
parser.add_argument('--batch_size', default=16, type=int, help='batch_size')


args = parser.parse_args()

##log
print('\n')    
print('TRAIN START!')
print('\n')
print('THE OUTPUT IS SAVED IN A TXT FILE HERE -------------------------------------------> ', args.folder_log)
print('\n')
if not os.path.exists(args.folder_log):
    os.makedirs(args.folder_log)
f = open(args.folder_log + '/out.txt', 'w')
sys.stdout = f


group = ["cars", "flowers"]
nb_epoch = 100
max_val = {}
max_val['cars'] = 0
max_val['flowers'] = 0
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
trainloader_2 = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=16, drop_last = True)


trainset    = torchvision.datasets.ImageFolder(root=args.data_dir + 'Flowers102/train', transform=transform_train)
trainloader_4 = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=8, drop_last = True)


testset = torchvision.datasets.ImageFolder(root= args.data_dir + 'StandCars/test', transform=transform_test)
testloader_2 = torch.utils.data.DataLoader(testset, batch_size= 4*args.batch_size, shuffle=True, num_workers=8, drop_last = False)


testset = torchvision.datasets.ImageFolder(root= args.data_dir + 'Flowers102/test', transform=transform_test)
testloader_4 = torch.utils.data.DataLoader(testset, batch_size= 4*args.batch_size, shuffle=True, num_workers=8, drop_last = False)




#build model
print('==> Building model..')
net = models.resnet50(pretrained=True)
net = erudite_model(net, 512, 196 + 102, group)
if use_cuda:
    
    net.features.cuda()
    net.spatial_attention.cuda()
    net.car_expert.cuda()
    net.flower_expert.cuda()
    net.mix_gate.cuda()
    net.max.cuda()   
    net.car_classifier.cuda()
    net.flower_classifier.cuda()
    net.classifier.cuda()
        

    net.features = torch.nn.DataParallel(net.features)
    net.spatial_attention = torch.nn.DataParallel(net.spatial_attention)
    net.car_expert = torch.nn.DataParallel(net.car_expert)
    net.flower_expert = torch.nn.DataParallel(net.flower_expert)
    net.mix_gate = torch.nn.DataParallel(net.mix_gate)
    net.max = torch.nn.DataParallel(net.max)
    net.car_classifier = torch.nn.DataParallel(net.car_classifier)
    net.flower_classifier = torch.nn.DataParallel(net.flower_classifier)
    net.classifier = torch.nn.DataParallel(net.classifier)


   
#build optimizer
optimizer = optim.SGD([
                        {'params': net.classifier.module.parameters(), 'lr': 0.1},
                        {'params': net.car_classifier.module.parameters(), 'lr': 0.1},
                        {'params': net.flower_classifier.module.parameters(), 'lr': 0.1},
                        {'params': net.mix_gate.module.parameters(), 'lr': 0.1},
                        {'params': net.car_expert.module.parameters(), 'lr': 0.1},
                        {'params': net.flower_expert.module.parameters(), 'lr': 0.1},
                        {'params': net.features.module.parameters(),   'lr': 0.01},

                     ],
                      momentum=0.9, weight_decay=5e-4)
optimizer_maml = optim.SGD([
                        {'params': net.spatial_attention.module.parameters(), 'lr': 0.1},
                     ],
                      momentum=0.9, weight_decay=5e-4)

#train
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


