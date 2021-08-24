from __future__ import print_function
import argparse
import os, sys
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import time
import cv2
from losses.multiscaleloss import EPE

from networks.FADNet import FADNet
from networks.stackhourglass import PSMNet
from networks.DispNetC import DispNetC
from networks.ESNet import ESNet
from networks.ESNet_M import ESNet_M

from dataloader.HabitatLoader import HabitatDataset

parser = argparse.ArgumentParser(description='Evaluate')
parser.add_argument('--vallist', default=None,
                    help='path to validation split file')
parser.add_argument('--loadmodel', default=None,
                    help='loading model')
parser.add_argument('--savepath', default='results/',
                    help='path to save the results.')
parser.add_argument('--model', default='esnet',
                    help='select model')
parser.add_argument('--maxdisp', type=int, default=192,
                    help='maxium disparity')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--devices', type=str, help='indicates CUDA devices, e.g. 0,1,2', default='0')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()

if not os.path.exists(args.savepath):
    os.makedirs(args.savepath)

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

devices = [int(item) for item in args.devices.split(',')]
ngpus = len(devices)

if args.model == 'esnet':
    model = ESNet(batchNorm=False, lastRelu=True, maxdisp=-1)
elif args.model == 'esnet_m':
    model = ESNet_M(batchNorm=False, lastRelu=True, maxdisp=-1)
elif args.model == 'psmnet':
    model = PSMNet(args.maxdisp)
elif args.model == 'fadnet':
    model = FADNet(False, True)
elif args.model == 'dispnetc':
    model = DispNetC(batchNorm=False, lastRelu=True, maxdisp=-1)
else:
    print('no model')
    sys.exit(-1)

model = nn.DataParallel(model, device_ids=devices)
model.cuda()

if args.loadmodel is not None:
    state_dict = torch.load(args.loadmodel)
    model.load_state_dict(state_dict['state_dict'])
    print('Loaded checkpoint {}'.format(args.loadmodel))

print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

def test(imgL,imgR):
        model.eval()

        if args.cuda:
           imgL = torch.FloatTensor(imgL).cuda()
           imgR = torch.FloatTensor(imgR).cuda()     

        imgL, imgR= Variable(imgL), Variable(imgR)

        with torch.no_grad():
            if args.model == "fadnet":
                output_net1, output_net2 = model(torch.cat((imgL, imgR), 1))
                output = torch.squeeze(output_net2)
            elif args.model == "psmnet":
                output = model(torch.cat((imgL, imgR), 1))
                output = torch.squeeze(output)
            elif args.model in ['esnet', 'esnet_m', 'dispnetc', 'dispnets']:
                output = model(torch.cat((imgL, imgR), 1))
                output = torch.squeeze(output[0])

        pred_disp = output.data.cpu().numpy()

        #print('larger than 192: %s' % pred_disp[pred_disp>0.75].shape)
        # print('min: %f, max: %f, mean: %f' % (np.min(pred_disp), np.max(pred_disp), np.mean(pred_disp)))

        return pred_disp

def main():
    eval_dataset = HabitatDataset(split_path=args.vallist, training=False)  
    for inputs in eval_dataset:

        imgL, imgR, dispL = inputs['img_left'],  inputs['img_right'], inputs['gt_disp']
        imgL = imgL.reshape((1, *imgL.shape))
        imgR = imgR.reshape((1, *imgR.shape))
        pred_disp = test(imgL,imgR)
        pred_disp = cv2.resize(pred_disp, eval_dataset.get_img_size()[::-1], interpolation = cv2.INTER_CUBIC)
        pred_disp *= eval_dataset.img_size[1]/eval_dataset.scale_size[1]

        pred_depth = np.nan_to_num((eval_dataset.baseline * eval_dataset.focal_length)/pred_disp, 10, 10)
        pred_img = (pred_depth * 65535/10).astype(np.uint16)
        
        depth_path = inputs['img_names'][2]
        scene = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(depth_path))))
        filename = os.path.basename(depth_path)

        dispL = dispL.squeeze().detach().numpy()
        mask = (dispL < 192) & (dispL > 0)
        epe = np.mean(np.abs(dispL[mask] - pred_disp[mask]))
        print('{} {} epe:{}'.format(scene, filename, str(epe)))
       
        outdir = os.path.join(args.savepath, scene)
        os.makedirs(outdir, exist_ok=True)
        cv2.imwrite(os.path.join(outdir, filename), pred_img)

if __name__ == '__main__':
   main()






