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

from torchvision.models import resnet18, ResNet18_Weights,vgg19,squeezenet1_1


#TODO: try different backbone
# backbone = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)
# backbone = torch.nn.Sequential(*(list(backbone.children())[:-1]))

backbone1 = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
backbone1 = torch.nn.Sequential(*(list(backbone1.children())[:-1]))
backbone2 = vgg19(pretrained=True)
modify = list(backbone2.children())[:-2]
modify.append(torch.nn.AdaptiveAvgPool2d(1))
backbone2 = torch.nn.Sequential(*(modify))

backbone3 = squeezenet1_1(pretrained=True)
modify = list(backbone3.children())[:-1]
modify.append(torch.nn.AdaptiveAvgPool2d(1))
backbone3 = torch.nn.Sequential(*(modify))



#TODO: put all hp together
DEVICE=torch.device('cuda:0')
LR=1e-6
CHAIR_LR=1e-3
BATCH_SIZE=64
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
    def __init__(self,backbone):
        super().__init__()

        self.encoder=backbone

        
    def forward(self, input1,input2,return_embed=False):
        emb1 = self.encoder(input1).squeeze()
        emb2 = self.encoder(input2).squeeze()

        
        return emb1,emb2
    



# # baseline=[]
model1=Model(backbone=backbone1).to(DEVICE)
model2=Model(backbone=backbone2).to(DEVICE)
model3=Model(backbone=backbone3).to(DEVICE)
# baseline=[model1,model2,model3]
# baseline_name=['vgg','casia','quad']
# baseline=[model1,model2]
# baseline_name=['vgg','casia']

baseline=[backbone1,backbone2,backbone3]
baseline_name=['resnetemb','vggemb','squeezeemb']

# baseline=[model3]
# baseline_name=['swinemb']

entrophy_loss=ContrastiveLoss(distance_metric=SiameseDistanceMetric.COSINE_DISTANCE)
load_model=[]

for idx, model in enumerate(baseline):
    print(f"training {baseline_name[idx]} model: ")
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    # scheduler = ReduceLROnPlateau(optimizer, patience=10)
    best_loss=10000
    for e in range(BASE_EPOCH):
        total_loss=0
        model.train()
        for i in tqdm(trainloader):
            input1=i[0].to(DEVICE)
            input2=i[1].to(DEVICE)
            labels=i[2].to(DEVICE).type(torch.float).view(-1,1)

            optimizer.zero_grad()
            emb1,emb2=model(input1).squeeze(),model(input2).squeeze()
            loss=entrophy_loss([emb1,emb2],labels)
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

            emb1,emb2=model(input1).squeeze(),model(input2).squeeze()
            loss=entrophy_loss([emb1,emb2],labels).cpu().detach().numpy()
            val_loss=val_loss+loss
            pred = (F.cosine_similarity(emb1,emb2) > 0.5).type(torch.int64).unsqueeze(1)
            correct = torch.sum(torch.eq(pred,labels)).cpu().detach().numpy()+correct
            labels_np=labels.cpu().detach().numpy()
            pred_npy=pred.cpu().detach().numpy()
            score=roc_auc_score(labels_np,pred_npy)*bs+score

        # scheduler.step(val_loss/len(valloader))


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
            emb1,emb2=model(input1).squeeze(),model(input2).squeeze()
            pred = F.cosine_similarity(emb1,emb2).detach().cpu().numpy()
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