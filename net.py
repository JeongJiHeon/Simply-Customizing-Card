import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision import models

import numpy as np
import glob
import tqdm
import os
import random
from PIL import ImageEnhance
from PIL import Image

import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname('utils.py'))))


import utils
from U_GAT_IT.ugatit import *



img_size = 256

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.Resize((img_size + 30, img_size + 30)),
    transforms.RandomCrop(img_size),
#    ColorTransformations(),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])
test_transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])


class dataset(torch.utils.data.Dataset):
    def __init__(self, dir = 'dataA/*.jpg', transform = train_transform):
        super().__init__()
        self.dir = glob.glob(dir)
        self.transform = transform
        
    def __len__(self):
        return len(self.dir)
        
    def __getitem__(self, idx):
        img = self.dir[idx]
        img = self.transform(Image.open(img))
        
        return img
    
    
    
class VGG_loss(nn.Module):
    def __init__(self, device):
        super(VGG_loss, self).__init__()
        self.device = device

        vgg_pretrained_feaures = models.vgg19(pretrained=True).features.to(self.device)
        
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.slice5 = nn.Sequential()

        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_feaures[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_feaures[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_feaures[x])
        for x in range(12, 21): 
            self.slice4.add_module(str(x), vgg_pretrained_feaures[x])
        for x in range(21, 30): 
            self.slice5.add_module(str(x), vgg_pretrained_feaures[x])

        for param in self.parameters():
            param.requires_grad= False

    def forward(self, real_image, fake_image):
        loss = 0 

        real_h = self.slice1(real_image)
        fake_h = self.slice1(fake_image)
        loss += torch.mean(torch.abs(real_h-fake_h)) * 1/32

        real_h = self.slice2(real_h)
        fake_h = self.slice2(fake_h)
        loss += torch.mean(torch.abs(real_h-fake_h)) * 1/16

        real_h = self.slice3(real_h)
        fake_h = self.slice3(fake_h)
        loss += torch.mean(torch.abs(real_h-fake_h)) * 1/8

        real_h = self.slice4(real_h)
        fake_h = self.slice4(fake_h)
        loss += torch.mean(torch.abs(real_h-fake_h)) * 1/4

        real_h = self.slice5(real_h)
        fake_h = self.slice5(fake_h)
        loss += torch.mean(torch.abs(real_h-fake_h)) * 1

        return loss

    def denorm(self, x):
        return ((x+1)/2) * 255.0
    
    
class TVLoss(nn.Module):
    def __init__(self,TVLoss_weight=1, beta = 1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight
        self.beta = 1

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        tv = torch.sqrt(h_tv + w_tv)
        return self.TVLoss_weight*2*(tv/count_w)/batch_size

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]
    
    
    
class UGATIT(object):
    def __init__(self, device, batch_size = 4, img_size = img_size,
                 path = ('dataA/*.jpg', 'dataB/*.jpg'), lr = 0.0001, 
                 betas = (0, 0.999), light = True, weight = [1,10,10,1500,0,20], weight_decay = 0.0001,
                 ID = 1, total_iteration = 150000
                
                ):
        self.device = device

        
        ### Build Model ###
        self.GeneratorA2B = ResnetGenerator(input_nc = 3, output_nc = 3, n_blocks = 6, img_size = img_size, light = light).to(device = self.device) # A->B
        self.GeneratorB2A = ResnetGenerator(input_nc = 3, output_nc = 3, n_blocks = 6, img_size = img_size, light = light).to(device = self.device) # B->A
        
        self.DiscriminatorLA  = Discriminator(input_nc = 3, n_layers = 5).to(device = self.device)
        self.DiscriminatorGA = Discriminator(input_nc = 3, n_layers = 7).to(device = self.device)
        
        self.DiscriminatorLB  = Discriminator(input_nc = 3, n_layers = 5).to(device = self.device)
        self.DiscriminatorGB = Discriminator(input_nc = 3, n_layers = 7).to(device = self.device)
        
        
        self.optimG = torch.optim.Adam(list(self.GeneratorA2B.parameters())+list(self.GeneratorB2A.parameters()), lr = lr, betas = betas, weight_decay = weight_decay)
        self.optimD = torch.optim.Adam(
            list(self.DiscriminatorLA.parameters())+list(self.DiscriminatorGA.parameters())+
            list(self.DiscriminatorLB.parameters())+list(self.DiscriminatorGB.parameters()),
            lr = lr,
            betas = betas,
            weight_decay = weight_decay
        )
        
        
        ### Build Loader ###
        self.datasetA    = dataset(path[0], transform = train_transform)
        self.datasetB    = dataset(path[1], transform = train_transform)
        self.dataloaderA = torch.utils.data.DataLoader(dataset(path[0]), batch_size = batch_size, shuffle = True, num_workers = 8, drop_last = True)
        self.dataloaderB = torch.utils.data.DataLoader(dataset(path[1]), batch_size = batch_size, shuffle = True, num_workers = 8, drop_last = True)
        
        self._iterA = iter(self.dataloaderA)
        self._iterB = iter(self.dataloaderB)
        
        
        ### Build Utils ###
        self.test_datasetA = dataset(path[0], transform = test_transform)
        self.test_datasetB = dataset(path[1], transform = test_transform)

        self.fixdataA_idx = utils.make_fix_idx(16, len(self.test_datasetA)-1)
        self.fixdataB_idx = utils.make_fix_idx(16, len(self.test_datasetB)-1)
        
        self.lr = lr
        
        self.total_iteration = total_iteration
        self.check_iteration = 1000
        self.times = 0
        self.ID = ID
        
        self.weight = weight # adv, cycle, identity, cam_logit
        self.RhoClipper = RhoClipper(0,1)
        self.output_directory = '{}/model/'.format(self.ID)
        
        self.MSELoss = nn.MSELoss()
        self.BCELoss = nn.BCEWithLogitsLoss()
        self.L1Loss  = nn.L1Loss()
#        self.vggloss = VGG_loss(device)
        self.tvloss  = TVLoss(self.weight[5]) 
        
    def FreezeD(self):
        
        FreezeD = [
            self.DiscriminatorLA.model[7].weight, self.DiscriminatorLA.model[7].bias,
            self.DiscriminatorLB.model[7].weight, self.DiscriminatorLB.model[7].bias,
            self.DiscriminatorGA.model[10].weight, self.DiscriminatorGA.model[10].bias,
            self.DiscriminatorGB.model[10].weight, self.DiscriminatorGB.model[10].bias
        ]
        for F in FreezeD:
            F.requires_grad = False

        
    def _train(self):
        self.GeneratorA2B.train(), self.GeneratorB2A.train(), self.DiscriminatorLA.train(), self.DiscriminatorGA.train(), self.DiscriminatorLB.train(), self.DiscriminatorGB.train()
        
    def _eval(self):
        self.GeneratorA2B.eval() , self.GeneratorB2A.eval() , self.DiscriminatorLA.eval() , self.DiscriminatorGA.eval() , self.DiscriminatorLB.eval() , self.DiscriminatorGB.eval()
    def _next(self):
        try:
            A = self._iterA.next()
        except:
            self._iterA = iter(self.dataloaderA)
            A = self._iterA.next()
            
        try:
            B = self._iterB.next()
        except:
            self._iterB = iter(self.dataloaderB)
            B = self._iterB.next()
            
        return A.to(device = self.device), B.to(device = self.device)
    
    def output(self, image, model = 'A'):
        assert A.__class__ == Image.Image, 'Image is not PIL.Image.Image type'
        
        test_transform = transforms.Compose([
            transforms.Resize((192, 192)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        
        image = test_transform(image)
        
        if model == 'A':
            self.GeneratorB2A.eval()
            return transforms.ToPILImage()((self.GeneratorB2A(image)+1)/2)
        elif model == 'B':
            self.GeneratorA2B.eval()
            return transforms.ToPILImage()((self.GeneratorA2B(image)+1)/2)


    def save(self):
        params = {}
        params['GeneratorA2B'] = self.GeneratorA2B.state_dict()
        params['GeneratorB2A'] = self.GeneratorB2A.state_dict()
        params['DiscriminatorLA'] = self.DiscriminatorLA.state_dict()
        params['DiscriminatorGA'] = self.DiscriminatorGA.state_dict()
        params['DiscriminatorLB'] = self.DiscriminatorLB.state_dict()
        params['DiscriminatorGB'] = self.DiscriminatorGB.state_dict()
        
        params['optimG'] = self.optimG.state_dict()
        params['optimD'] = self.optimD.state_dict()
        
        params['fixdataA_idx'] = self.fixdataA_idx
        params['fixdataB_idx'] = self.fixdataB_idx
        
        params['times'] = self.times

        torch.save(params, self.output_directory + '{:03}_k_model.pt'.format(self.times+1))
        torch.save(params, self.output_directory + 'lastest_model.pt')

    def load(self, model):
        
        
        if model == 'lastest':
            params = torch.load(self.output_directory + 'lastest_model.pt')
            
        elif model == 'transfer':
            params = torch.load(self.output_directory + 'pretrained.pt')
        else:
            params = torch.load(self.output_directory + '{:03}_k_model.pt'.format(model))

        self.GeneratorA2B.load_state_dict(params['GeneratorA2B'])
        self.GeneratorB2A.load_state_dict(params['GeneratorB2A'])
        self.DiscriminatorLA.load_state_dict(params['DiscriminatorLA'])
        self.DiscriminatorGA.load_state_dict(params['DiscriminatorGA'])
        self.DiscriminatorLB.load_state_dict(params['DiscriminatorLB'])
        self.DiscriminatorGB.load_state_dict(params['DiscriminatorGB'])
        self.fixdataA_idx = params['fixdataA_idx']
        self.fixdataB_idx = params['fixdataB_idx']
        self.times = params['times']
        
        print('--------------------------')
        print('      {:03}K iter Load '.format(self.times))
        print('--------------------------')
        
        if model == 'transfer':
            self.times = 0

        
    def test(self, times):
        self.GeneratorA2B.eval()
        self.GeneratorB2A.eval()
        
        with torch.no_grad():
            utils.saveimage(self.GeneratorA2B(self.fixdataA)[0], times, 'A', self.ID)
            utils.saveimage(self.GeneratorB2A(self.fixdataB)[0], times, 'B', self.ID)
        

    def train(self):
        self.fixdataA = utils.make_fix_img(self.fixdataA_idx, self.test_datasetA).to(device = self.device)
        self.fixdataB = utils.make_fix_img(self.fixdataB_idx, self.test_datasetB).to(device = self.device)
        utils.saveimage(self.fixdataA, 0, 'A', self.ID)
        utils.saveimage(self.fixdataB, 0, 'B', self.ID)
#        self.FreezeD()


        
        for times in range(self.total_iteration//self.check_iteration):
            self._train()

            if times >= (self.total_iteration//self.check_iteration)//2:
                self.optimG.param_groups[0]['lr'] -= self.lr/((self.total_iteration//self.check_iteration)//2)
                self.optimD.param_groups[0]['lr'] -= self.lr/((self.total_iteration//self.check_iteration)//2)

            if times < self.times:
                continue
            pbar = tqdm.tqdm(range(self.check_iteration), total = self.check_iteration)

            for step in pbar:
                self.optimD.zero_grad()
                
                realA, realB = self._next()
                fakeB, _ = self.GeneratorA2B(realA)
                fakeA, _ = self.GeneratorB2A(realB)
                
                realLA, realLA_CAM = self.DiscriminatorLA(realA)
                realGA, realGA_CAM = self.DiscriminatorGA(realA)
                
                realLB, realLB_CAM = self.DiscriminatorLB(realB)
                realGB, realGB_CAM = self.DiscriminatorGB(realB)
            
                fakeLA, fakeLA_CAM = self.DiscriminatorLA(fakeA)
                fakeGA, fakeGA_CAM = self.DiscriminatorGA(fakeA)
                
                fakeLB, fakeLB_CAM = self.DiscriminatorLB(fakeB)
                fakeGB, fakeGB_CAM = self.DiscriminatorGB(fakeB)
                
                Adversarial_Loss_A = self.MSELoss(realLA, torch.ones(realLA.shape).to(device = self.device)) + self.MSELoss(realGA, torch.ones(realGA.shape).to(device = self.device)) + self.MSELoss(fakeLA, torch.zeros(fakeLA.shape).to(device = self.device)) + self.MSELoss(fakeGA, torch.zeros(fakeGA.shape).to(device = self.device))
                Adversarial_Loss_B = self.MSELoss(realLB, torch.ones(realLB.shape).to(device = self.device)) + self.MSELoss(realGB, torch.ones(realGB.shape).to(device = self.device)) + self.MSELoss(fakeLB, torch.zeros(fakeLB.shape).to(device = self.device)) + self.MSELoss(fakeGB, torch.zeros(fakeGB.shape).to(device = self.device))
                
                Ad_CAM_Loss_A      = self.MSELoss(realLA_CAM, torch.ones(realLA_CAM.shape).to(device = self.device)) + self.MSELoss(realGA_CAM, torch.ones(realGA_CAM.shape).to(device = self.device)) + self.MSELoss(fakeLA_CAM, torch.zeros(fakeLA_CAM.shape).to(device = self.device)) + self.MSELoss(fakeGA_CAM, torch.zeros(fakeGA_CAM.shape).to(device = self.device))
                Ad_CAM_Loss_B      = self.MSELoss(realLB_CAM, torch.ones(realLB_CAM.shape).to(device = self.device)) + self.MSELoss(realGB_CAM, torch.ones(realGB_CAM.shape).to(device = self.device)) + self.MSELoss(realLB_CAM, torch.zeros(realLB_CAM.shape).to(device = self.device)) + self.MSELoss(fakeGB_CAM, torch.zeros(fakeGB_CAM.shape).to(device = self.device))
                            
                Discriminator_Loss_A = Adversarial_Loss_A + Ad_CAM_Loss_A
                Discriminator_Loss_B = Adversarial_Loss_B + Ad_CAM_Loss_B
                
                Discriminator_Loss = self.weight[0] * (Discriminator_Loss_A + Discriminator_Loss_B)
                Discriminator_Loss.backward()
                self.optimD.step()
                
                del(fakeB, fakeA, realLA, realLA_CAM, realGA, realGA_CAM, realLB, realLB_CAM, realGB, realGB_CAM, fakeLA, fakeLA_CAM, fakeGA, fakeGA_CAM, fakeLB, fakeLB_CAM, fakeGB, fakeGB_CAM)
                
                self.optimG.zero_grad()
                
                fakeB, fakeB_CAM_gen = self.GeneratorA2B(realA)
                fakeA, fakeA_CAM_gen = self.GeneratorB2A(realB)
                
                reconA, _ = self.GeneratorB2A(fakeB)
                reconB, _ = self.GeneratorA2B(fakeA)
                
                fakeB2B, fakeB2B_CAM_gen = self.GeneratorA2B(realB)
                fakeA2A, fakeA2A_CAM_gen = self.GeneratorB2A(realA)
                
            
                fakeLA, fakeLA_CAM = self.DiscriminatorLA(fakeA)
                fakeGA, fakeGA_CAM = self.DiscriminatorGA(fakeA)
                
                fakeLB, fakeLB_CAM = self.DiscriminatorLB(fakeB)
                fakeGB, fakeGB_CAM = self.DiscriminatorGB(fakeB)
                
                Adversarial_Loss_A = self.MSELoss(fakeLA, torch.ones(fakeLA.shape).to(device = self.device)) + self.MSELoss(fakeGA, torch.ones(fakeGA.shape).to(device = self.device))
                Adversarial_Loss_B = self.MSELoss(fakeLB, torch.ones(fakeLB.shape).to(device = self.device)) + self.MSELoss(fakeGB, torch.ones(fakeGB.shape).to(device = self.device))
                
                Ad_CAM_Loss_A = self.MSELoss(fakeLA_CAM, torch.ones(fakeLA_CAM.shape).to(device = self.device)) + self.MSELoss(fakeGA_CAM, torch.ones(fakeGA_CAM.shape).to(device = self.device))
                Ad_CAM_Loss_B = self.MSELoss(fakeLB_CAM, torch.ones(fakeLB_CAM.shape).to(device = self.device)) + self.MSELoss(fakeGB_CAM, torch.ones(fakeGB_CAM.shape).to(device = self.device))
                
                
                Cycle_Loss_A = self.L1Loss(reconA, realA)
                Cycle_Loss_B = self.L1Loss(reconB, realB)
                Identity_Loss_A = self.L1Loss(fakeA2A, realA)
                Identity_Loss_B = self.L1Loss(fakeB2B, realB)
                
                G_CAM_Loss_A = self.BCELoss(fakeB_CAM_gen, torch.ones(fakeB_CAM_gen.shape).to(device = self.device)) + self.BCELoss(fakeB2B_CAM_gen, torch.zeros(fakeB2B_CAM_gen.shape).to(device = self.device))
                G_CAM_Loss_B = self.BCELoss(fakeA_CAM_gen, torch.ones(fakeA_CAM_gen.shape).to(device = self.device)) + self.BCELoss(fakeA2A_CAM_gen, torch.zeros(fakeA2A_CAM_gen.shape).to(device = self.device))
                
                Generator_Loss_A = self.weight[0] * (Adversarial_Loss_A + Ad_CAM_Loss_A) + self.weight[1] * Cycle_Loss_A + self.weight[2] * Identity_Loss_A + self.weight[3] * G_CAM_Loss_A
                Generator_Loss_B = self.weight[0] * (Adversarial_Loss_B + Ad_CAM_Loss_B) + self.weight[1] * Cycle_Loss_B + self.weight[2] * Identity_Loss_B + self.weight[3] * G_CAM_Loss_B
#                 Generator_vgg_Loss_A = self.vggloss(realA, fakeA)
#                 Generator_vgg_Loss_B = self.vggloss(realB, fakeB)
#                 Generator_Loss_A += self.weight[4] * Generator_vgg_Loss_A
#                 Generator_Loss_B += self.weight[4] * Generator_vgg_Loss_B
    
    
                Generator_TV_Loss_A = self.tvloss(fakeA)
                Generator_TV_Loss_B = self.tvloss(fakeB)             
                Generator_Loss_A += Generator_TV_Loss_A
                Generator_Loss_B += Generator_TV_Loss_B 
                
                
                Generator_Loss = Generator_Loss_A + Generator_Loss_B


                Generator_Loss.backward()
                self.optimG.step()
                del(fakeB, fakeB_CAM_gen, fakeA, fakeA_CAM_gen, reconA, reconB, fakeB2B, fakeB2B_CAM_gen, fakeA2A, fakeA2A_CAM_gen)
                
                self.GeneratorA2B.apply(self.RhoClipper)
                self.GeneratorB2A.apply(self.RhoClipper)
                
#                msg = '[{:03}/{:03}] [Generator A : {:.3f} | B : {:.3f}] [Discriminator A : {:.3f} | B : {:.3f}]'.format(times, self.total_iteration//self.check_iteration, Generator_Loss_A.item(), Generator_Loss_B.item(), Discriminator_Loss_A.item(), Discriminator_Loss_B.item())
                msg = '[{:03}/{:03}] [ModelA G : {:.3f} | D : {:.3f}] [ModelB G : {:.3f} | D : {:.3f}]'.format(times, self.total_iteration//self.check_iteration, Generator_Loss_A.item(), Discriminator_Loss_A.item(), Generator_Loss_B.item(), Discriminator_Loss_B.item())

                pbar.set_description_str(msg)
                
            self.times = times + 1
            self.test(times = self.times)
            self.save()


                
                
                



