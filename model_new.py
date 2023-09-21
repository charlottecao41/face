'''
Use contrasive loss for pretrain (same person)

And Circle Loss for train

Highest AUC is about 60%

'''



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
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve

#TODO: try different backbone
# backbone = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)
# backbone = torch.nn.Sequential(*(list(backbone.children())[:-1]))

backbone=InceptionResnetV1(pretrained='vggface2')
# print(backbone)

#TODO: put all hp together
DEVICE=torch.device('cuda:0')
LR1=1e-3
LR2=1e-4
BATCH_SIZE=64
EPOCH1=10 #stage one epoch, contrasive loss for the same person
EPOCH2=10 #stage two epoch, circle loss with cosine distance, you can adjust this value for better results
SAVE_PATH1='best_backbone.pth'
SAVE_PATH2='best_final.pth'
EMBED_SIZE=512
IMG_SIZE= 225 #Depends on the backbone

#TODO: fix paths, i think i forgot some

train_file_path = "/home/charlotte/face/train_relationships.csv"
train_folders_path = "/home/charlotte/face/train/"
val_famillies = "F09"

class SiameseDistanceMetric(Enum):
    """
    The metric for the contrastive loss
    """
    EUCLIDEAN = lambda x, y: F.pairwise_distance(x, y, p=2)
    MANHATTAN = lambda x, y: F.pairwise_distance(x, y, p=1)
    COSINE_DISTANCE = lambda x, y: 1-F.cosine_similarity(x, y)

class PretrainDataset(Dataset):
    def __init__(self, relations, person_to_images_map, transform=None):  
        self.relations = relations
        self.transform = transform
        self.person_to_images_map = person_to_images_map
        self.ppl = list(person_to_images_map.keys()) 

        self.ppl=[x for x in self.ppl if len(self.ppl[x])>1]

    def __len__(self):
        return len(self.ppl)*2
               
    def __getitem__(self, idx):

        ppl=self.ppl[idx//2]

        path1 = choice(self.person_to_images_map[ppl])
        while path1==path2:
            path1, path2 = choice(self.person_to_images_map[ppl]), choice(self.person_to_images_map[ppl])


        if (idx%2==0): #Positive samples
            while True:
                path2 = choice(self.person_to_images_map[ppl])
                if path2 != path1:
                    break

            label = 1
        else:          #Negative samples

            while True:
                ppl2 = choice(self.ppl) 
                if ppl != ppl2:
                    break

            label = 0
            path2 = choice(self.person_to_images_map[ppl2])

        img1, img2 = Image.open(path1), Image.open(path2)
        
        if self.transform:
            img1, img2 = self.transform(img1), self.transform(img2)
        
        return img1, img2, label


class KinDataset(Dataset):
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
    
class CircleLossDataset(Dataset):
    def __init__(self, relations, person_to_images_map, transform=None):  
        self.relations = relations        
        self.transform = transform
        self.person_to_images_map = person_to_images_map
        self.ppl = list(person_to_images_map.keys())

    def __len__(self):
        return len(self.relations)
               
    def __getitem__(self, idx):
        
        anchor = self.relations[idx][0]
        positive = self.relations[idx][1]

        #negative
        while True:
            p1 = choice(self.ppl)
            if p1 != anchor and (anchor,p1) not in self.relations and (p1,anchor) not in self.relations:
                break 
        
        path1, path2, path3 = choice(self.person_to_images_map[anchor]),choice(self.person_to_images_map[positive]), choice(self.person_to_images_map[p1])
        anchor, img1, img2 = Image.open(path1), Image.open(path2), Image.open(path3)
        if self.transform:
            anchor, img1, img2 = self.transform(anchor), self.transform(img1),self.transform(img2)
        
        return anchor, img1, img2 #anchor, +, -

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

pretrainset = KinDataset(train_relations, train_person_to_images_map, train_transform)
prevalset = KinDataset(val_relations, val_person_to_images_map, val_transform)

pretrainloader = DataLoader(pretrainset, batch_size=BATCH_SIZE, shuffle=True)
prevalloader = DataLoader(prevalset, batch_size=BATCH_SIZE, shuffle=False)

# trainset = KinDataset(train_relations, train_person_to_images_map, train_transform)
# valset = KinDataset(val_relations, val_person_to_images_map, val_transform)

trainset = CircleLossDataset(train_relations, train_person_to_images_map, train_transform)
valset = KinDataset(val_relations, val_person_to_images_map, val_transform)

trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
valloader = DataLoader(valset, batch_size=BATCH_SIZE, shuffle=False)


class Model(nn.Module):
    def __init__(self,backbone=backbone):
        super().__init__()

        self.encoder=backbone
        
    def forward(self, input1,input2, input3=None):
        
        emb1 = self.encoder(input1)
        emb2 = self.encoder(input2)
        if input3 is not None:
            emb3 = self.encoder(input3)
            # negative= (emb1 @ emb3.transpose(1, 0)).view(-1)
            # positive= (emb1 @ emb2.transpose(1, 0)).view(-1)

            positive = SiameseDistanceMetric.COSINE_DISTANCE(emb1,emb2)
            negative = SiameseDistanceMetric.COSINE_DISTANCE(emb1,emb3)
            return positive, negative

        else:
            return emb1,emb2

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

model=Model(backbone).to(DEVICE)
#TODO: switch to other loss, distance metric etc
compute_loss=ContrastiveLoss(distance_metric=SiameseDistanceMetric.COSINE_DISTANCE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR1)

best_loss=10000
for e in range(EPOCH1):
    total_loss=0
    model.train()
    for i in tqdm(pretrainloader):
        input1=i[0].to(DEVICE)
        input2=i[1].to(DEVICE)
        labels=i[2].to(DEVICE)

        optimizer.zero_grad()
        emb1,emb2=model(input1,input2)
        loss=compute_loss([emb1,emb2],labels)
        loss.backward()
        optimizer.step()
        total_loss=total_loss+loss

    print(f"train encoder epoch: {e}, loss: {total_loss/len(pretrainloader)}")

    val_loss=0
    score=0
    model.eval()
    for i in tqdm(prevalloader):
        input1=i[0].to(DEVICE)
        input2=i[1].to(DEVICE)
        labels=i[2].to(DEVICE)
        labels_np=labels.cpu().detach().numpy()

        bs=input1.shape[0]

        emb1,emb2 =model(input1,input2)
        pred = (1-SiameseDistanceMetric.COSINE_DISTANCE(emb1,emb2)).cpu().detach().numpy()
        score=roc_auc_score(labels_np,pred)*bs+score
        loss=compute_loss([emb1,emb2],labels).cpu().detach().numpy()
        val_loss=val_loss+loss

    print(f"val encoder epoch: {e}, loss: {val_loss/len(prevalloader)},AUC score {score/len(prevalset)}")

    if val_loss<best_loss:
        best_loss=val_loss
        torch.save(model.state_dict(), SAVE_PATH1)

model = Model(backbone)
model.load_state_dict(torch.load(SAVE_PATH1))
model.to(DEVICE)

#Stage 2

best_loss=10000
best_score=0
optimizer = torch.optim.Adam(model.parameters(), lr=LR2)
compute_loss= CircleLoss(m=0.25, gamma=256)
for e in range(EPOCH2):
    total_loss=0
    model.train()
    for i in tqdm(trainloader):
        input1=i[0].to(DEVICE)
        input2=i[1].to(DEVICE)
        input3=i[2].to(DEVICE)
        optimizer.zero_grad()
        p,n =model(input1,input2,input3)

        loss=compute_loss(p,n)
        loss.backward()
        optimizer.step()
        total_loss=total_loss+loss

    print(f"train epoch: {e}, loss: {total_loss/len(trainloader)}")

    val_loss=0
    score=0
    model.eval()
    for i in tqdm(prevalloader):
        input1=i[0].to(DEVICE)
        input2=i[1].to(DEVICE)
        labels=i[2].to(DEVICE)
        labels_np=labels.cpu().detach().numpy()

        bs=input1.shape[0]

        emb1,emb2 =model(input1,input2)
        pred = (1-SiameseDistanceMetric.COSINE_DISTANCE(emb1,emb2)).cpu().detach().numpy()
        score=roc_auc_score(labels_np,pred)*bs+score
        # loss=compute_loss(p,n)
        # val_loss=val_loss+loss

    print(f"val encoder epoch: {e}, AUC score {score/len(valset)}")

    if score/len(prevalset)>best_score:
        best_score=score/len(prevalset)
        torch.save(model.state_dict(), SAVE_PATH1)