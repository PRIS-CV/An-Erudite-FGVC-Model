import torch
import torch.nn as nn


criterion = nn.CrossEntropyLoss()
criterion_no_average = nn.CrossEntropyLoss(reduce=False)

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

    def __init__(self, feature_size, group):
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
    def __init__(self, net, feature_size,classes_num, group):

        super(erudite_model, self).__init__()

        self.group = group

        self.num_ftrs = 2048*1*1

        self.features = nn.Sequential(*list(net.children())[:-2])

        self.max = nn.MaxPool2d(kernel_size=7, stride=7)

        self.spatial_attention = SpatialAttention(feature_size=self.num_ftrs, head_num=8)
        
        #what
        self.car_expert = BasicConv(self.num_ftrs, feature_size, kernel_size=1, stride=1)
        self.flower_expert = BasicConv(self.num_ftrs, feature_size, kernel_size=1, stride=1)
        
        #mix_classifier
        self.mix_gate = FusionGate(feature_size, group)
   
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
            
            #eliminate label>class
            target_car = torch.where(origin_target<196,origin_target,0)
            target_flower = torch.where(origin_target<102,origin_target,0)

            
            loss_car = (criterion_no_average(predict_car, target_car)*car_weight).mean()

            loss_flower = (criterion_no_average(predict_flower, target_flower)*flower_weight).mean()
        
            loss_all = criterion(x, targets)
            diverse_loss = (1 - torch.max(att, 1)[0]).mean() + att.mean()
            loss = loss_all + (loss_car  + loss_flower)/len(self.group) + diverse_loss * 0.1

        else:
            loss = criterion(x, targets)
            
        return x, loss
