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

#TODO: try different backbone
backbone = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)
backbone = torch.nn.Sequential(*(list(backbone.children())[:-1]))

backbone=InceptionResnetV1(pretrained='vggface2')

print(backbone)

#TODO: put all hp together
DEVICE=torch.device('cuda:0')
LR=1e-3
BATCH_SIZE=64
EPOCH=5
SAVE_PATH='best_model.pth'
EMBED_SIZE=512
IMG_SIZE= 128 #Depends on the backbone

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

trainset = KinDataset(train_relations, train_person_to_images_map, train_transform)
valset = KinDataset(val_relations, val_person_to_images_map, val_transform)

trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
valloader = DataLoader(valset, batch_size=BATCH_SIZE, shuffle=False)


class Model(nn.Module):
    def __init__(self,backbone=backbone,embed_size=EMBED_SIZE):
        super().__init__()

        self.encoder=backbone
        self.embed_size=embed_size
        self.fc1 = nn.Linear(self.embed_size*2, self.embed_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(self.embed_size,1)
        
    def forward(self, input1,input2):
        
        emb1 = self.encoder(input1).squeeze()
        emb2 = self.encoder(input2).squeeze()

        x1=torch.pow(emb1,2)+ torch.pow(emb2,2)
        x2=torch.pow(emb1 - emb2, 2)
        result= torch.cat((x1,x2),dim=1)
        result = self.fc1(result)
        result = self.relu1(result)
        result = self.fc2(result)

        #TODO: do sth with the backbone - two stage training? idk, or use the weird version

        
        return result


model=Model(backbone=backbone,embed_size=EMBED_SIZE).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
entrophy_loss=torch.nn.BCEWithLogitsLoss()

best_loss=10000
for e in range(EPOCH):
    total_loss=0
    model.train()
    for i in tqdm(trainloader):
        input1=i[0].to(DEVICE)
        input2=i[1].to(DEVICE)
        labels=i[2].to(DEVICE).type(torch.float).view(-1,1)

        optimizer.zero_grad()
        pred=model(input1,input2)
        loss=entrophy_loss(pred,labels)
        loss.backward()
        optimizer.step()
        total_loss=total_loss+loss

    print(f"train encoder epoch: {e}, loss: {total_loss/len(trainloader)}")

    val_loss=0
    correct = 0
    model.eval()
    for i in tqdm(valloader):
        input1=i[0].to(DEVICE)
        input2=i[1].to(DEVICE)
        labels=i[2].to(DEVICE).type(torch.float).view(-1,1)

        pred =model(input1,input2)
        loss=entrophy_loss(pred,labels).cpu().detach().numpy()
        val_loss=val_loss+loss
        pred = (pred > 0.5).type(torch.int64)
        correct = torch.sum(torch.eq(pred,labels)).cpu().detach().numpy()+correct

    print(f"val encoder epoch: {e}, loss: {val_loss/len(valloader)}, correct: {correct/len(valset)}")

    if val_loss<best_loss:
        best_loss=val_loss
        torch.save(model.state_dict(), SAVE_PATH)

model = Model(backbone=backbone,embed_size=EMBED_SIZE).to(DEVICE)
model.load_state_dict(torch.load(SAVE_PATH))
model.eval()

correct=0
for i in tqdm(valloader):
        input1=i[0].to(DEVICE)
        input2=i[1].to(DEVICE)
        labels=i[2].to(DEVICE)

        pred=model(input1,input2).squeeze()
        pred = (pred > 0.5).type(torch.int64)
        correct = torch.sum(torch.eq(pred,labels)).cpu().detach().numpy()+correct

print("final correct",correct/len(valset))