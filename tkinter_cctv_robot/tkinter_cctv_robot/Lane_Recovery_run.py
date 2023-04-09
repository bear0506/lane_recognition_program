from os import listdir
from os.path import join
import random
import matplotlib.pyplot as plt
#%matplotlib inline

import os
import time
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_pil_image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.environ['KMP_DUPLICATE_LIB_OK']='True'

#학습 데이터 정렬
class FacadeDataset(Dataset):
    def __init__(self, path2img, direction='b2a', transform=False):
        super().__init__()
        self.direction = direction
        self.path2a = join(path2img, 'a')
        self.path2b = join(path2img, 'b')
        self.img_filenames = [x for x in listdir(self.path2a)]
        self.transform = transform

    def __getitem__(self, index):
        # Windows OS 전용(Linux는 다름)
        a = Image.open(join(self.path2a+'/', self.img_filenames[index])).convert('RGB')
        b = Image.open(join(self.path2b+'/', self.img_filenames[index])).convert('RGB')

        if self.transform:
            a = self.transform(a)
            b = self.transform(b)
        # 학습 방향 결정
        if self.direction == 'b2a':
            return b,a
        else:
            return a,b

    def __len__(self):
        return len(self.img_filenames)

#이미지 변환
transform = transforms.Compose([
                    transforms.Resize((256,256)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])

#Windows OS 전용(Linux는 다름)
#path2img = './Train_Image/'
#train_ds = FacadeDataset(path2img, transform=transform)

#a,b = train_ds[0]
#plt.figure(figsize=(10,10))
#plt.subplot(1,2,1)
#plt.imshow(to_pil_image(0.5*a+0.5))
#plt.axis('off')
#plt.subplot(1,2,2)
#plt.imshow(to_pil_image(0.5*b+0.5))
#plt.axis('off')
#plt.show()

#데이터 읽어 들이기(섞이)
#train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)

#Backbone 네트워크 설정(UNet 사용, 필요시 ResNet으로 변경 가능)
#===============================================
class UNetDown(nn.Module):
    def __init__(self, in_channels, out_channels, normalize=True, dropout=0.0):
        super().__init__()

        layers = [nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1, bias=False)]

        if normalize:
            layers.append(nn.InstanceNorm2d(out_channels)),

        layers.append(nn.LeakyReLU(0.2))

        if dropout:
            layers.append(nn.Dropout(dropout))

        self.down = nn.Sequential(*layers)

    def forward(self, x):
        x = self.down(x)
        return x

# UNet 다운셀플링(중간 CHK)
x = torch.randn(16, 3, 256,256, device=device)
model = UNetDown(3,64).to(device)
down_out = model(x)
print(down_out.shape)

class UNetUp(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()

        layers = [
            nn.ConvTranspose2d(in_channels, out_channels,4,2,1,bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU()
        ]

        if dropout:
            layers.append(nn.Dropout(dropout))

        self.up = nn.Sequential(*layers)

    def forward(self,x,skip):
        x = self.up(x)
        x = torch.cat((x,skip),1)
        return x

# UNet 업셀플링(중간 CHK)
x = torch.randn(16, 128, 64, 64, device=device)
model = UNetUp(128,64).to(device)
out = model(x,down_out)
print(out.shape)

#===============================================

# generator: 복구된 차선 이미지를 생성(UNet기반)
class GeneratorUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()

        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64,128)
        self.down3 = UNetDown(128,256)
        self.down4 = UNetDown(256,512,dropout=0.5)
        self.down5 = UNetDown(512,512,dropout=0.5)
        self.down6 = UNetDown(512,512,dropout=0.5)
        self.down7 = UNetDown(512,512,dropout=0.5)
        self.down8 = UNetDown(512,512,normalize=False,dropout=0.5)

        self.up1 = UNetUp(512,512,dropout=0.5)
        self.up2 = UNetUp(1024,512,dropout=0.5)
        self.up3 = UNetUp(1024,512,dropout=0.5)
        self.up4 = UNetUp(1024,512,dropout=0.5)
        self.up5 = UNetUp(1024,256)
        self.up6 = UNetUp(512,128)
        self.up7 = UNetUp(256,64)
        self.up8 = nn.Sequential(
            nn.ConvTranspose2d(128,3,4,stride=2,padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)

        u1 = self.up1(d8,d7)
        u2 = self.up2(u1,d6)
        u3 = self.up3(u2,d5)
        u4 = self.up4(u3,d4)
        u5 = self.up5(u4,d3)
        u6 = self.up6(u5,d2)
        u7 = self.up7(u6,d1)
        u8 = self.up8(u7)

        return u8

# Generator 구간(중간CHK)
x = torch.randn(16,3,256,256,device=device)
model = GeneratorUNet().to(device)
out = model(x)
#print(out.shape)

# Patch 블럭(16x16의 패치로 분할하여 확인)
class Dis_block(nn.Module):
    def __init__(self, in_channels, out_channels, normalize=True):
        super().__init__()

        layers = [nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        x = self.block(x)
        return x

# Patch 블럭 (중간CHK)
x = torch.randn(16,64,128,128,device=device)
model = Dis_block(64,128).to(device)
out = model(x)
#print(out.shape)

#===============================================

# Discriminator: 복구된 차선 검증
class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()

        self.stage_1 = Dis_block(in_channels*2,64,normalize=False)
        self.stage_2 = Dis_block(64,128)
        self.stage_3 = Dis_block(128,256)
        self.stage_4 = Dis_block(256,512)

        self.patch = nn.Conv2d(512,1,3,padding=1) # 16x16 패치 생성

    def forward(self,a,b):
        x = torch.cat((a,b),1)
        x = self.stage_1(x)
        x = self.stage_2(x)
        x = self.stage_3(x)
        x = self.stage_4(x)
        x = self.patch(x)
        x = torch.sigmoid(x)
        return x

# check
x = torch.randn(16,3,256,256,device=device)
model = Discriminator().to(device)
out = model(x,x)
print(out.shape)

#================================================

# Model 선언(GPU사용)
model_gen = GeneratorUNet().to(device)
model_dis = Discriminator().to(device)

# 학습을 위한 가중치 초기화
def initialize_weights(model):
    class_name = model.__class__.__name__
    if class_name.find('Conv') != -1:
        nn.init.normal_(model.weight.data, 0.0, 0.02)

model_gen.apply(initialize_weights);
model_dis.apply(initialize_weights);

#================================================

# 학습결과 불러오기
# weights = torch.load(rf'D:\workspace\Lane detection\22차 트레이닝(흰색 실선, 점선)\Trained_Model\weights_gen.pt')
weights = torch.load(rf'./Trained_Model/weights_gen.pt', device)
model_gen.load_state_dict(weights)

# # 평가 모드(추가 학습: train(), Weight initialized: False)
# model_gen.eval()
#
# with torch.no_grad():
#     a = Image.open(r'C:\Users\LSM\Desktop\Lane detection\200.jpg').convert('RGB')
#     # a = frameImg.convert('RGB')
#     a = transform(a)
#     a = a.unsqueeze(0)
#     fake_imgs = model_gen(a.to(device)).detach().cpu()
#     fake_imgs = fake_imgs.squeeze()

    # plt.imshow(to_pil_image(0.5*fake_imgs+0.5))
    # plt.show()
