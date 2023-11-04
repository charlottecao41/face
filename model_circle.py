#create a person dataset and get the model to recognise the person
import torchvision
import torch
import torch.nn as nn
from random import choice
from torch.utils.data import Dataset
from PIL import Image
from collections import defaultdict
from torch.utils.data import DataLoader
from torchvision import transforms
from glob import glob
import pandas as pd
from enum import Enum
from typing import Iterable, Dict
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm
from facenet_pytorch import InceptionResnetV1
from typing import Tuple
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from torchvision.models import resnet18, ResNet18_Weights,vgg19
import os
import numpy as np

#TODO: try different backbone
# backbone = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)
# backbone = torch.nn.Sequential(*(list(backbone.children())[:-1]))

backbone1=InceptionResnetV1(pretrained='vggface2')
backbone2 = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
backbone2 = torch.nn.Sequential(*(list(backbone2.children())[:-2]))
backbone3 = vgg19(pretrained=True)
modify = list(backbone2.children())[:-2]
backbone3 = torch.nn.Sequential(*(modify))

#TODO: put all hp together
DEVICE=torch.device('cuda:0')
LR=1e-5
BATCH_SIZE=64
EPOCH=50
ENCODER_PATH='best_encoder.pth'
SAVE_PATH='best_triplet.pth'
EMBED_SIZE=512
IMG_SIZE= 196 #Depends on the backbone

#TODO: fix paths, i think i forgot some

train_file_path = "/home/charlotte/face/train_relationships.csv"
train_folders_path = "/home/charlotte/face/train/"
val_famillies = "F09"

    
class SamePersonDataset(Dataset):
    def __init__(self, relations, person_to_images_map, transform=None):  
        self.relations = relations
        # for i in relations:
        #     if i[0] not in self.relations:
        #         self.relations.update({i[0]:[i[1]]})

        #     else:
        #         if i[1] not in self.relations[i[0]]:
        #             self.relations[i[0]].append(i[1])
        
        self.transform = transform
        self.person_to_images_map = person_to_images_map
        self.ppl = [i for i in list(person_to_images_map.keys()) if len(self.person_to_images_map[i])>1]

    def __len__(self):
        return len(self.ppl)*2
               
    def __getitem__(self, idx):
        
        if idx >= len(self.ppl):
            new_idx=idx-len(self.ppl)
            anchor=self.ppl[new_idx]
            path1=self.person_to_images_map[anchor][0]
            while True:
                path2=choice(self.person_to_images_map[anchor])
                if path2 != path1:
                    break
            label=1

        else:
            anchor=self.ppl[idx]
            path1=self.person_to_images_map[anchor][0]
            #negative
            while True:
                p1 = choice(self.ppl)
                if p1 != anchor and (anchor,p1) not in self.relations and (p1,anchor) not in self.relations:
                    break

            label=0
        
            path2 = choice(self.person_to_images_map[p1])
        img1, img2 = Image.open(path1), Image.open(path2)
        
        if self.transform:
            img1, img2 = self.transform(img1), self.transform(img2)
        
        return img1, img2, label

class KinDataset(Dataset):
    def __init__(self, relations, person_to_images_map, transform=None):  
        self.relations = relations
        
        # for i in relations:
        #     if i[0] not in self.relations:
        #         self.relations.update({i[0]:[i[1]]})

        #     else:
        #         if i[1] not in self.relations[i[0]]:
        #             self.relations[i[0]].append(i[1])
        
        self.transform = transform
        self.person_to_images_map = person_to_images_map
        self.ppl = list(person_to_images_map.keys())

    def __len__(self):
        return len(self.relations)
               
    def __getitem__(self, idx):
        
        anchor = self.relations[idx][0]

        #negative
        while True:
            p1 = choice(self.ppl)
            if p1 != anchor and (anchor,p1) not in self.relations and (p1,anchor) not in self.relations:
                break 
        
        path1, path2, path3 = choice(self.person_to_images_map[anchor]),choice(self.person_to_images_map[self.relations[idx][1]]), choice(self.person_to_images_map[p1])
        anchor, img1, img2 = Image.open(path1), Image.open(path2), Image.open(path3)
        if self.transform:
            anchor, img1, img2 = self.transform(anchor), self.transform(img1),self.transform(img2)
        
        return anchor, img1, img2 #anchor, +, -
    

class AltKinDataset(Dataset):
    def __init__(self, relations, person_to_images_map, transform=None):  
        self.relations = relations
        self.transform = transform
        self.person_to_images_map = person_to_images_map
        self.ppl = list(person_to_images_map.keys()) 

    def __len__(self):
        return len(self.relations)*2
               
    def __getitem__(self, idx):
        
        if (idx%2==0): #Positive samples
            p1, p2 = self.relations[idx//2]
            label = 1
        else:          #TODO: better way to sample Negative samples
            while True:
                p1 = choice(self.ppl) 
                p2 = choice(self.ppl) 
                if p1 != p2 and (p1, p2) not in self.relations and (p2, p1) not in self.relations:
                    break 
            label = 0
        
        path1, path2 = choice(self.person_to_images_map[p1]), choice(self.person_to_images_map[p2])
        img1, img2 = Image.open(path1), Image.open(path2)
        
        if self.transform:
            img1, img2 = self.transform(img1), self.transform(img2)
        
        return img1, img2, label

all_images = glob(train_folders_path + "*/*/*.jpg")

train_images = [x for x in all_images if val_famillies not in x]
val_images = [x for x in all_images if val_famillies in x]

train_person_to_images_map = defaultdict(list)

ppl = [x.split("/")[-3] + "/" + x.split("/")[-2] for x in all_images]

for x in train_images:
    train_person_to_images_map[x.split("/")[-3] + "/" + x.split("/")[-2]].append(x)

val_person_to_images_map = defaultdict(list)

for x in val_images:
    val_person_to_images_map[x.split("/")[-3] + "/" + x.split("/")[-2]].append(x)
	
relationships = pd.read_csv(train_file_path)
relationships = list(zip(relationships.p1.values, relationships.p2.values))
relationships = [x for x in relationships if x[0] in ppl and x[1] in ppl]

train_relations = [x for x in relationships if val_famillies not in x[0]]

val_relations  = [x for x in relationships if val_famillies in x[0]]

#TODO: fix transform, maybe add some data augmentation. Normalise -> need to check the mean and var

train_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5]) 
])
val_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5]) 
])

sameperson_train=SamePersonDataset(train_relations, train_person_to_images_map, train_transform)
sameperson_val = SamePersonDataset(val_relations, val_person_to_images_map, val_transform)
trainset = KinDataset(train_relations, train_person_to_images_map, train_transform)
trainset_1 = AltKinDataset(train_relations, train_person_to_images_map, train_transform)
valset = KinDataset(val_relations, val_person_to_images_map, val_transform)
valset_1 = AltKinDataset(val_relations, val_person_to_images_map, val_transform)

trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
trainloader_1 = DataLoader(trainset_1, batch_size=BATCH_SIZE, shuffle=True)
valloader = DataLoader(valset, batch_size=BATCH_SIZE, shuffle=False)
valloader_1 = DataLoader(valset_1, batch_size=BATCH_SIZE, shuffle=True) #shuffle so don't be so predictable

sameperson_trainloader = DataLoader(sameperson_train, batch_size=BATCH_SIZE, shuffle=True)
sameperson_valloader = DataLoader(sameperson_val, batch_size=BATCH_SIZE, shuffle=False)

class SiameseDistanceMetric(Enum):
    """
    The metric for the contrastive loss
    """
    EUCLIDEAN = lambda x, y: F.pairwise_distance(x, y, p=2)
    MANHATTAN = lambda x, y: F.pairwise_distance(x, y, p=1)
    COSINE_DISTANCE = lambda x, y: 1-F.cosine_similarity(x, y)


class ContrastiveLoss(nn.Module):

    def __init__(self, distance_metric=SiameseDistanceMetric.EUCLIDEAN, margin: float = 0.5, size_average:bool = True):
        super(ContrastiveLoss, self).__init__()
        self.distance_metric = distance_metric
        self.margin = margin
        self.size_average = size_average

    def get_config_dict(self):
        distance_metric_name = self.distance_metric.__name__
        for name, value in vars(SiameseDistanceMetric).items():
            if value == self.distance_metric:
                distance_metric_name = "SiameseDistanceMetric.{}".format(name)
                break

        return {'distance_metric': distance_metric_name, 'margin': self.margin, 'size_average': self.size_average}

    def forward(self, reps, labels: Tensor):
        assert len(reps) == 2
        rep_anchor, rep_other = reps
        distances = self.distance_metric(rep_anchor, rep_other)
        losses = 0.5 * (labels.float() * distances.pow(2) + (1 - labels).float() * F.relu(self.margin - distances).pow(2))
        return losses.mean() if self.size_average else losses.sum()


class CircleLoss(nn.Module):
    def __init__(self, m: float, gamma: float) -> None:
        super(CircleLoss, self).__init__()
        self.m = m
        self.gamma = gamma
        self.soft_plus = nn.Softplus()

    def forward(self, sp: Tensor, sn: Tensor) -> Tensor:
        ap = torch.clamp_min(- sp.detach() + 1 + self.m, min=0.)
        an = torch.clamp_min(sn.detach() + self.m, min=0.)

        delta_p = 1 - self.m
        delta_n = self.m

        logit_p = - ap * (sp - delta_p) * self.gamma
        logit_n = an * (sn - delta_n) * self.gamma

        loss = self.soft_plus(torch.logsumexp(logit_n, dim=0) + torch.logsumexp(logit_p, dim=0))

        return loss


class Model(nn.Module):
    def __init__(self,backbone,embed_size=EMBED_SIZE,squeeze=False):
        super().__init__()

        self.encoder=backbone
        self.embed_size=embed_size
        self.fc1 = nn.Linear(self.embed_size*2, self.embed_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(self.embed_size,1)
        if squeeze:
            self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.squeeze=squeeze
        
    def forward(self, input1,input2, input3):
        
        anchor = self.encoder(input1)
        positive = self.encoder(input2)
        if self.squeeze:
            anchor=self.avgpool(anchor).squeeze()
            positive=self.avgpool(positive).squeeze()
        if input3 is not None:
            negative = self.encoder(input3)
            if self.squeeze:
                negative=self.avgpool(negative).squeeze()
                
            

            #TODO: explore different combinations
            negative= (anchor @ negative.transpose(1, 0)).view(-1)
        # result= torch.cat((x1,x2),dim=1)
        # result = self.fc1(result)
        # result = self.relu1(result)
        # result = self.fc2(result)

            positive= (anchor @ positive.transpose(1, 0)).view(-1)
        else:
            negative=None

        
        return positive,negative,anchor,positive,negative
    


model1=Model(backbone=backbone1,embed_size=EMBED_SIZE).to(DEVICE)
model2=Model(backbone=backbone2,embed_size=EMBED_SIZE,squeeze=True).to(DEVICE)
model3=Model(backbone=backbone3,embed_size=EMBED_SIZE,squeeze=True).to(DEVICE)
models=[model1,model2,model3]
baseline_name=['inception','resnet','vgg']

for idx, model in enumerate(models):
    name = baseline_name[idx]
    print(name)
    best_loss=10000
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    compute_loss= CircleLoss(m=0.25, gamma=256)
    for e in range(EPOCH):
        total_loss=0
        model.train()
        for i in tqdm(trainloader):
            input1=i[0].to(DEVICE)
            input2=i[1].to(DEVICE)
            input3=i[2].to(DEVICE)

            optimizer.zero_grad()
            p,n,_,_,_=model(input1,input2,input3)

            loss=compute_loss(p,n)
            loss.backward()
            optimizer.step()
            total_loss=total_loss+loss

        print(f"train epoch: {e}, loss: {total_loss/len(trainloader)}")

        val_loss=0
        correct = 0
        model.eval()
        for i in tqdm(valloader):
            input1=i[0].to(DEVICE)
            input2=i[1].to(DEVICE)
            input3=i[2].to(DEVICE)

            p,n,_,_,_=model(input1,input2,input3)

            loss=compute_loss(p,n).cpu().detach().numpy()
            val_loss=val_loss+loss
        print(f"val epoch: {e}, loss: {val_loss/len(valloader)}")

        val_loss=0
        correct = 0
        total=0
        model.eval()
        score=0
        for i in tqdm(valloader_1):
            bs = i[0].shape[0]
            input1=i[0].to(DEVICE)
            input2=i[1].to(DEVICE)
            labels_np=i[2].cpu().numpy()
            labels=i[2].to(DEVICE).type(torch.int64)

            _,_,anchor,target,_=model(input1,input2,None)

            pred_npy = torch.abs(F.cosine_similarity(anchor,target)).detach().cpu().numpy()
            pred=torch.abs(F.cosine_similarity(anchor,target))

            outcome=torch.subtract(pred.type(torch.int64),labels)
            correct=torch.sum(torch.abs(outcome)).cpu().detach().numpy()+correct
            total=total+len(pred)
            score=roc_auc_score(labels_np,pred_npy)*bs+score
        print(f"val epoch: {e}, correct: {correct/total}, score {score/len(valset_1)}")

        if val_loss<best_loss:
            best_loss=val_loss
            torch.save(model.state_dict(), f'{name}_circle.pth')

load_model = []
for idx, model in enumerate(models):
    model.load_state_dict(torch.load(f'{baseline_name[idx]}_circle.pth'))
    model.eval()
    load_model.append(model)

del models

submission = pd.read_csv('sample_submission.csv')
class TestDataset(Dataset):
    def __init__(self, submission=submission, transform=None, test_folder = 'test'):  
        self.submission = submission
        self.transform = transform
        self.test_folder=test_folder

    def __len__(self):
        return len(self.submission)
               
    def __getitem__(self, idx):
        imgs = submission.iloc[idx]['img_pair'].split('-')

        img1, img2 = os.path.join(self.test_folder,imgs[0]),os.path.join(self.test_folder,imgs[1])
        
        img1, img2 = Image.open(img1), Image.open(img2)
        
        if self.transform:
            img1, img2 = self.transform(img1), self.transform(img2)
        
        return img1, img2

testset = TestDataset(transform=val_transform)
testloader = DataLoader(testset,batch_size=BATCH_SIZE,shuffle=False)
with torch.no_grad():
    for idx, model_list in enumerate(load_model):
        name = baseline_name[idx]

        score=[]
        for imgs in tqdm(testloader):
                input1=imgs[0].to(DEVICE)
                input2=imgs[1].to(DEVICE)
                _,_,anchor,target,_=model(input1,input2,None)
                pred=torch.abs(F.cosine_similarity(anchor,target)).cpu().detach().numpy()
                score = np.concatenate((score,pred),0)

        submission['is_related'] = score
        submission.to_csv(f'{name}_circle.csv',index=False)