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
import csv
import os
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau


#TODO: try different backbone
# backbone = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)
# backbone = torch.nn.Sequential(*(list(backbone.children())[:-1]))

backbone1 = 'vggface2'
backbone2 = 'casia-webface'



# print(backbone)

#TODO: put all hp together
DEVICE=torch.device('cuda:0')
LR=5e-4
CHAIR_LR=1e-3
BATCH_SIZE=128
#NOTE: you can modify this
BASE_EPOCH=100  #EPOCH OF BASELINES
EPOCH=1 #EPOCH OF CHAIR MODEL
SAVE_PATH='best_model.pth'  #SAVE PATH OF CHAIR MODEL
EMBED_SIZE=512
IMG_SIZE= 160 #Depends on the backbone

#TODO: fix paths, i think i forgot some

train_file_path = "train_relationships.csv"
train_folders_path = "train/"
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
        
        if idx%2==0: #Positive samples
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
    def __init__(self,backbone,embed_size=EMBED_SIZE,mode=0):
        super().__init__()

        self.encoder=InceptionResnetV1(pretrained=backbone)
        self.embed_size=embed_size
        self.mode=mode
        self.relu = nn.ReLU()
        if mode==0:
            self.fc1 = nn.Linear(self.embed_size*2, self.embed_size)
            self.batchnorm = nn.BatchNorm1d(self.embed_size, eps=0.001, momentum=0.1, affine=True)
            self.fc2 = nn.Linear(self.embed_size,1)
        elif mode==1:
            self.fc1 = nn.Linear(self.embed_size*4, self.embed_size*2)
            self.batchnorm = nn.BatchNorm1d(self.embed_size*2, eps=0.001, momentum=0.1, affine=True)
            self.fc2 = nn.Linear(self.embed_size*2,1)
        else:
            NotImplementedError

        
    def forward(self, input1,input2,return_embed=False):
        
        emb1 = self.encoder(input1)
        emb2 = self.encoder(input2)

        if self.mode==0:
            x1=torch.pow(emb1,2)+ torch.pow(emb2,2)
            x2=torch.pow(emb1 - emb2, 2)
            result= torch.cat((x1,x2),dim=-1)

            result = self.fc1(result)
            result = self.batchnorm(result)
            result = self.relu(result)
            result = self.fc2(result)

        elif self.mode==1:
            x1=torch.pow(emb1,2)+ torch.pow(emb2,2)
            x2=torch.pow(emb1 - emb2, 2)
            x3=emb1+emb2
            x4=emb1*emb2
            result= torch.cat((x1,x2,x3,x4),dim=-1)
            result = self.fc1(result)
            result = self.batchnorm(result)
            result = self.relu(result)
            result = self.fc2(result)

        result=torch.sigmoid(result)

        #TODO: do sth with the backbone - two stage training? idk, or use the weird version
        if return_embed:
            score=torch.pow(emb1-emb2,2)

            return score, result

        else:
            return result
    
class ChairModel(nn.Module):
    def __init__(self,agent_num=3,embed_size=EMBED_SIZE):
        super().__init__()
        self.embed_size=embed_size
        self.attn1 = nn.Linear(agent_num*embed_size, embed_size)
        self.attn2 = nn.Linear(embed_size, agent_num)
        self.relu = nn.ReLU()
        
    def forward(self, diff, r):
        '''diff: list of torch embedding
            r: list of results
        '''

        diff= torch.cat(diff,dim=-1)
        diff=self.relu(self.attn1(diff))
        diff=self.attn2(diff).softmax(dim=-1)

        result=0
        for idx,score in enumerate(r):
            result=score*diff[:,idx:idx+1]+result



        #TODO: do sth with the backbone - two stage training? idk, or use the weird version

        
        return result


# # baseline=[]
model1=Model(backbone=backbone1,embed_size=EMBED_SIZE).to(DEVICE)
model2=Model(backbone=backbone2,embed_size=EMBED_SIZE).to(DEVICE)
model3=Model(backbone=backbone1,embed_size=EMBED_SIZE,mode=1).to(DEVICE)
baseline=[model1,model2,model3]
baseline_name=['vgg','casia','quad']
# baseline=[model1,model2]
# baseline_name=['vgg','casia']

# baseline=[model3]
# baseline_name=['quad']

entrophy_loss=torch.nn.BCELoss()
load_model=[]

for idx, model in enumerate(baseline):
    print(f"training {baseline_name[idx]} model: ")
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = ReduceLROnPlateau(optimizer, patience=10)
    best_loss=10000
    for e in range(BASE_EPOCH):
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
        score=0
        for i in tqdm(valloader):
            input1=i[0].to(DEVICE)
            input2=i[1].to(DEVICE)
            labels=i[2]
            labels_np=labels.cpu().detach().numpy()
            labels=labels.to(DEVICE).type(torch.float).view(-1,1)
            bs=input1.shape[0]

            pred =model(input1,input2)
            pred_npy=pred.cpu().detach().numpy()
            loss=entrophy_loss(pred,labels).cpu().detach().numpy()
            val_loss=val_loss+loss
            pred = (pred > 0.5).type(torch.int64)
            correct = torch.sum(torch.eq(pred,labels)).cpu().detach().numpy()+correct
            labels_np=labels.cpu().detach().numpy()
            score=roc_auc_score(labels_np,pred_npy)*bs+score

        scheduler.step(val_loss/len(valloader))


        print(f"val encoder epoch: {e}, loss: {val_loss/len(valloader)}, correct: {correct/len(valset)},AUC score {score/len(valset)}")

        if val_loss<best_loss:
            best_loss=val_loss
            torch.save(model.state_dict(), f'{baseline_name[idx]}.pth')

    



print("Loading models...")
load_model=[]
for idx, model in enumerate(baseline):
    model.load_state_dict(torch.load(f'{baseline_name[idx]}.pth'))
    load_model.append(model)

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
    for idx, model in enumerate(load_model):
        name = baseline_name[idx]
        score=[]
        for imgs in tqdm(testloader):
            input1=imgs[0].to(DEVICE)
            input2=imgs[1].to(DEVICE)
            pred = model(input1,input2).squeeze().cpu().detach().numpy()
            score = np.concatenate((score,pred),0)

        submission['is_related'] = score
        submission.to_csv(f'{name}.csv',index=False)
        


        




# model = ChairModel(agent_num=len(load_model)).to(DEVICE)
# optimizer = torch.optim.Adam(model.parameters(), lr=CHAIR_LR)
# entrophy_loss=torch.nn.BCEWithLogitsLoss()

# best_loss=10000


# print("training chair: ")
# for e in range(EPOCH):
#     total_loss=0
#     model.train()
#     for i in tqdm(trainloader):
#         input1=i[0]
#         input2=i[1]
#         # input1=i[0].to(DEVICE)
#         # input2=i[1].to(DEVICE)
#         labels=i[2].to(DEVICE).type(torch.float).view(-1,1)

#         optimizer.zero_grad()
#         r=[]
#         diff=[]

#         for m in load_model:
#             score,result=m(input1,input2,return_embed=True)
#             diff.append(score.to(DEVICE))
#             r.append(result.to(DEVICE))

#         pred=model(r=r,diff=diff)
            
#         loss=entrophy_loss(pred,labels)
#         loss.backward()
#         optimizer.step()
#         total_loss=total_loss+loss

#     print(f"train encoder epoch: {e}, loss: {total_loss/len(trainloader)}")

#     val_loss=0
#     correct = 0
#     model.eval()
#     score=0
#     for i in tqdm(valloader):
#         input1=i[0]
#         input2=i[1]
#         # input1=i[0].to(DEVICE)
#         # input2=i[1].to(DEVICE)
#         labels=i[2]
#         labels_np=labels.cpu().detach().numpy()
#         labels=labels.to(DEVICE).type(torch.float).view(-1,1)
#         bs=input1.shape[0]

#         r=[]
#         diff=[]

#         for m in load_model:
#             score,result=m(input1,input2,return_embed=True)
#             diff.append(score.to(DEVICE))
#             r.append(result.to(DEVICE))

#         pred=model(r=r,diff=diff)

#         pred_npy=pred.cpu().detach().numpy()
#         loss=entrophy_loss(pred,labels).cpu().detach().numpy()
#         val_loss=val_loss+loss
#         pred = (pred > 0.5).type(torch.int64)
#         correct = torch.sum(torch.eq(pred,labels)).cpu().detach().numpy()+correct
#         labels_np=labels.cpu().detach().numpy()
#         score=roc_auc_score(labels_np,pred_npy)*bs+score


#     print(f"val encoder epoch: {e}, loss: {val_loss/len(valloader)}, correct: {correct/len(valset)},AUC score {score/len(valset)}")

#     if val_loss<best_loss:
#         best_loss=val_loss
#         torch.save(model.state_dict(), SAVE_PATH)

# model = ChairModel(base=load_model).to(DEVICE)
# model.load_state_dict(torch.load(SAVE_PATH))
# model.eval()

# correct=0
# score = 0
# for i in tqdm(valloader):
#         input1=i[0].to(DEVICE)
#         input2=i[1].to(DEVICE)
#         labels=i[2].to(DEVICE)

#         bs=input1.shape[0]

#         pred=model(input1,input2).squeeze()
#         pred_npy=pred.cpu().detach().numpy()
#         pred = (pred > 0.5).type(torch.int64)
#         correct = torch.sum(torch.eq(pred,labels)).cpu().detach().numpy()+correct
#         labels_np=labels.cpu().detach().numpy()
#         score=roc_auc_score(labels_np,pred_npy)*bs+score



# print(f"final correct {correct/len(valset)}, AUC score {score/len(valset)}")