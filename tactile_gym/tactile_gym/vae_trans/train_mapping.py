import torch
from torch import optim
from torch.nn import functional as F
import torch.nn 
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image

import numpy as np
import argparse
import os
from utils import *
from model import VAE,Discriminator,FeatureMapping

parser = argparse.ArgumentParser()
parser.add_argument('--test-dir', default='data/tactile_test/', type=str, help='path to the data')
parser.add_argument('--data-dir', default='data/tactile_pair/', type=str, help='path to the data')
parser.add_argument('--data-tag', default='tactile', type=str, help='all domains containing keywords')
parser.add_argument('--batch-size', default=64, type=int)
parser.add_argument('--num-epochs', default=1200, type=int)
parser.add_argument('--num-workers', default=4, type=int)
parser.add_argument('--learning-rate', default=0.0001, type=float)
parser.add_argument('--content-latent-size', default=32, type=int)

args = parser.parse_args()

Model_R = VAE
Model_S = VAE
Model_D = Discriminator
Model_M = FeatureMapping

def match_loss(x,y, model_R ,model_M,model_S,model_D,device):
    mu, logsigma = model_R.encoder(x)
    contentcode_x = reparameterize(mu, logsigma)

    mu_y, logsigma_y = model_S.encoder(y)
    contentcode_y_gt = reparameterize(mu_y, logsigma_y)

    contentcode_y = model_M( contentcode_x)

    recon_y = model_S.decoder(contentcode_y)
    fake_out = model_D(recon_y).squeeze()
    loss = torch.nn.BCELoss()(fake_out, torch.ones_like(fake_out, device=device))

    losst = F.l1_loss(contentcode_y_gt,contentcode_y)
    #print(losst)
    return loss + losst * 60

def mixup_loss(x,y,model_R ,model_M,model_S,device):
    mu_x, logsigma_x = model_R.encoder(x)
    mu_y, logsigma_y = model_S.encoder(y)
    x_mu1,x_mu2 = torch.chunk(mu_x,2,0)
    y_mu1,y_mu2 = torch.chunk(mu_y,2,0)

    lamda = torch.rand(x_mu1.size(0),1,device=device)

    x_new = lamda * x_mu1 + (1-lamda) * x_mu2
    y_new = lamda * y_mu1 + (1-lamda) * y_mu2

    y_pred = model_M(x_new)

    loss = F.mse_loss(y_pred, y_new, reduction='mean')
    #print(losst)
    return loss

def Discriminator_loss(x,y, model_R ,model_M,model_S,model_D,device):
    mu, logsigma = model_R.encoder(x)
    contentcode_x = reparameterize(mu, logsigma)
    contentcode_y = model_M( contentcode_x)
    recon_y = model_S.decoder(contentcode_y).detach()
    real_out = model_D(y).squeeze()
    fake_out = model_D(recon_y).squeeze()
    real_loss = torch.nn.BCELoss()(real_out, torch.ones_like(real_out, device=device))
    fake_loss = torch.nn.BCELoss()(fake_out, torch.zeros_like(real_out, device=device))
    return real_loss + fake_loss    
    
def main():
    # create direc
    if not os.path.exists("checkpoints"):
        os.makedirs("checkpoints")

    if not os.path.exists('checkimages'):
        os.makedirs("checkimages")

    # create dataset and loader
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = TactilePairDataset(args.data_dir,args.data_tag, transform)
    testset = TactilePairDataset(args.test_dir,args.data_tag, transform)

    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last = True)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last = True)
    # create model

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_R = Model_R().eval()
    model_S = Model_S().eval()
    
    model_R.load_state_dict(torch.load("checkpoints/1_model.pt"))   #qqqq exchange
    model_S.load_state_dict(torch.load("checkpoints/0_model.pt"))

    model_R = model_R.to(device)
    model_S = model_S.to(device)

    model_D = Model_D()
    model_M = Model_M()
    #print(model_M)
    
    model_D = model_D.to(device)
    model_M = model_M.to(device)
    optimizer_D = optim.Adam(model_D.parameters(), lr=args.learning_rate * 0.1)
    optimizer_M = optim.Adam(model_M.parameters(), lr=args.learning_rate)
    # do the training
    writer = SummaryWriter()
    batch_count = 0
    for i_epoch in range(args.num_epochs):
        model_M.train()
        for i_batch, imgs in enumerate(loader):
                batch_count += 1
                imgs = imgs.permute(1,0,2,3,4).to(device, non_blocking=True).type(torch.cuda.FloatTensor)  
                # imgs[0] = random_augmentaction(imgs[0])
                imgs[0] = salt_and_pepper(imgs[0],0.1)

                optimizer_M.zero_grad()
                matchloss = match_loss(imgs[0],imgs[1], model_R ,model_M,model_S,model_D,device)
                mix_loss = mixup_loss(imgs[0],imgs[1],model_R ,model_M,model_S,device)
                mloss = matchloss *0.1 + mix_loss * 10 #TODO
                mloss.backward()
                optimizer_M.step()               
                
                optimizer_D.zero_grad()
                dloss = Discriminator_loss(imgs[0],imgs[1], model_R ,model_M,model_S,model_D,device)
                if dloss > 0.35:
                    dloss.backward()
                else:
                    pass
                optimizer_D.step()

                # write log
                #print("mloss: ",mloss.item())
                #print("dloss: ",dloss.item())
                writer.add_scalar('mloss', mloss.item(), batch_count)
                writer.add_scalar('dloss', dloss.item(), batch_count)

        if i_epoch % 200 == 0:
            model_M.eval()
            for i_batch, imgs in enumerate(testloader):
                imgs = imgs.permute(1,0,2,3,4).to(device, non_blocking=True).type(torch.cuda.FloatTensor)  

            test_img = imgs[0][0:8]
            # test_img = random_augmentaction(test_img)
            test_img = salt_and_pepper(test_img,0.1)

            mu,_ = model_R.encoder(test_img)
            gt_img = imgs[1][0:8]
            mu_sim = model_M(mu)
            recon_imgs = model_S.decoder(mu_sim)
            saved_imgs = torch.cat([test_img, recon_imgs, gt_img], dim=0)

            all_test_img = imgs[0]
            all_gt_img = imgs[1]
            all_mu,_ = model_R.encoder(all_test_img)
            all_mu_sim = model_M(all_mu)
            all_recon_imgs = model_S.decoder(all_mu_sim)
            ssim_score = ssim(all_gt_img, all_recon_imgs).item()
            MAE_score = F.mse_loss(all_gt_img, all_recon_imgs, reduction='mean')
            torch.save(model_M.state_dict(), "./checkpoints/mapping_model.pt" )
            torch.save(model_D.state_dict(), "./checkpoints/discriminator_model.pt" )
            save_image(saved_imgs, "./checkimages/final/%d.png" % (i_epoch), nrow=8)
            print("epoch: ", i_epoch, " SSIM: ",'%.6f' % ssim_score," MAE: ",'%.6f' % MAE_score.item())
            #print("loss: ",loss)


        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last = True)
    writer.close()

if __name__ == '__main__':
    main()
