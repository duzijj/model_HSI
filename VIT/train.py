# -*- coding: utf-8 -*-
"""
Created on Sun Jul  4 17:36:37 2021

@author: Administrator
"""

import argparse
import collections
import time
import pandas as pd
import numpy as np

import torch
import torch.optim as optim
import torchvision
from sklearn import metrics, preprocessing

import record
import torch_optimizer as optim2

from EarlyStopping import EarlyStopping
from model import generate_model, load_dataset
import geniter
# # Setting Params
# torch.autograd.set_detect_anomaly(True)
parser = argparse.ArgumentParser(description='Training for HSI')
model_info=''




#transformer
parser.add_argument('--depth', default=14, type=int,
                    help='depth of transformer')
parser.add_argument('--mlp_dim', default=200, type=int,
                    help='mlp dim')

parser.add_argument('--dataset', dest='dataset', default='KSC',
                    help="Name of dataset.")
parser.add_argument('--dropout', dest='dropout', default=0.1, type=float,
                    help="dropout")
parser.add_argument('--emb_dropout', dest='emb_dropout', default=0.1, type=float,
                    help="emb_dropout")
# parser.add_argument('--ratio', dest='ratio', default='[1/8,8/16,16/24,24/32,32/40]', type=str,
#                     help="Attenuation(decay) coefficient 衰减系数")
parser.add_argument('--ratio', dest='ratio', default='[0,0,0,0,0]', type=str,
                    help="Attenuation(decay) coefficient 衰减系数")
parser.add_argument('--ratio_trainable', dest='ratio_trainable', default=0, type=int,
                    help="Attenuation(decay) coefficient trainable? 衰减系数是否可训练")



# parser.add_argument('--model_name', dest='model_name', default='cw_vit', type=str)
# parser.add_argument('--model_name', dest='model_name', default='hs_cw_vit', type=str)

# parser.add_argument('--model_name', dest='model_name', default='decay_vit', type=str)
# parser.add_argument('--model_name', dest='model_name', default='vit', type=str)

# parser.add_argument('--model_name', dest='model_name', default='A2S2K')
# parser.add_argument('--model_name', dest='model_name', default='SSRN', type=str)
# parser.add_argument('--model_name', dest='model_name', default='ResNet', type=str)
parser.add_argument('--model_name', dest='model_name', default='PyResNet', type=str)
# parser.add_argument('--model_name', dest='model_name', default='ContextualNet', type=str)

parser.add_argument('--set_embed_linear', dest='set_embed_linear', default=0, type=int,
                    help='''whether set a linear before input into transformer 
                    在transformer前面是否要设置一个linear''')
parser.add_argument('--has_map_ratio', dest='has_map_ratio', default=0, type=int,
                    help="全局可训练参数 for decay_vit")
parser.add_argument('--pos_embed',dest='pos_embed',default='sin',type=str,
                    help="position embedding of [sin,trainable_sin,random]")
parser.add_argument('--patch', type=int, dest='patch', default=4, help="Length of = patch*2+1")


#args for decay_vit/vit
parser.add_argument('--has_pe', type=int, dest='has_pe',
                    default=0, help="need pos embedding")
parser.add_argument('--cw_pe', type=int, dest='cw_pe',
                    default=0, help="clockwise_pos_embedding")



parser.add_argument('--transpose', default=0, type=int, help='transpose the image plot')
parser.add_argument('--pool', default='cls', type=str, help='pool of transformer in [mean,cls]')
parser.add_argument('--test_rotation', default=0, type=int, help='Test set rotation by 90°,180°,270°')
parser.add_argument('--scheduler', default='CosineAnnealingLR', type=str, help='''learning rate scheduler
                     in [StepLR, CosineAnnealingLR]''')
parser.add_argument('--dim', default=200, type=int, help='dim')
parser.add_argument('--heads', default=3, type=int, help='heads')
parser.add_argument('--optimizer',dest='optimizer',default='adam', type=str,
                    help="Name of optimizer.")
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--epoch', type=int, dest='epoch', default=100, help="")
parser.add_argument('--iter', type=int, dest='iter', default=10, help="")


parser.add_argument('--kernel',type=int,dest='kernel',default=24,help="Length of kernel")
parser.add_argument('--valid_split',type=float,dest='valid_split',default=0.9,
                    help="Percentage of validation split.")
parser.add_argument('--load',type=int,dest='load',default=0, help="读取已经训练好的模型")
parser.add_argument('--cuda', type=str, dest='cuda', default='cuda:0', help="手动控制使用哪个gpu")
parser.add_argument('--pretreat', type=str, dest='pretreat', default='PCA', help="预处理:默认pca")
parser.add_argument('--earlystop', type=int, dest='earlystop', default=1, help="earlystopping")
parser.add_argument('--patch_size', type=int, dest='patch_size', default=1, help="patch_size for vit")
args = parser.parse_args()




def train(net,train_iter,valida_iter,loss,optimizer,device,epochs=args.epoch):
    loss_list = [100]
    global X,y,TOTAL_TOTAL_TIME,TOTAL_TOTAL_SAMPLE
    print("training on ", device)
    start = time.time()
    if args.scheduler=='StepLR':
        lr_adjust = torch.optim.lr_scheduler.StepLR(optimizer,
                                        15, gamma=0.5, last_epoch=-1)
    elif args.scheduler=='CosineAnnealingLR':
        lr_adjust = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 15, eta_min=0.0, last_epoch=-1)
        
    PATH = "../save/" + model_info  + '.pt'
    early_stopping = EarlyStopping(patience=20, path=PATH)
    train_loss_list = [100]
    valid_loss_list = [100]
    
    for epoch in range(epochs):
        train_acc_sum, n = 0.0, 0
        time_epoch = time.time()
        for X, y in train_iter:
            
            batch_count, train_l_sum = 0, 0
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y.long())

            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
            TOTAL_TOTAL_SAMPLE += X.shape[0]
        if args.scheduler:
            lr_adjust.step()
        valida_acc, valida_loss = record.evaluate_accuracy(
            valida_iter, net, loss, device)
        loss_list.append(valida_loss)
        
        train_loss_list.append(train_l_sum/batch_count)
        valid_loss_list.append(valida_loss)

        # if loss_list[-1] > 10*loss_list[-2]:
        #     net.load_state_dict(torch.load(PATH))
        #     loss_list[-1] = loss_list[-2]
        #     print('backtrack')
        backtrack = 0
        if train_loss_list[-1] > 10*train_loss_list[-2] and\
            valid_loss_list[-1] > 10*valid_loss_list[-2]:
            net.load_state_dict(torch.load(PATH))
            train_loss_list.pop()
            valid_loss_list.pop()
            backtrack = 1
        else:
            early_stopping(valida_loss, net)
            if early_stopping.early_stop:
                print("Early stopping")
                break
        
        time_cost = time.time() - time_epoch
        TOTAL_TOTAL_TIME += time_cost
        print(
            'epoch %d, train loss %.6f, train acc %.4f, valida loss %.6f, valida acc %.4f, time %.8f sec, lr %.5f, %d'
            % (epoch + 1, train_l_sum / batch_count, train_acc_sum*100 / n,
               valida_loss, valida_acc*100, time_cost ,
               optimizer.param_groups[0]['lr'], backtrack))
        # if early_stopping and loss_list[-1] > 10*loss_list[-2]:
        #     if early_epoch == 0:
        #         torch.save(net.state_dict(), PATH)
        #     early_epoch += 1
        #     loss_list[-1] = loss_list[-2]
        #     if early_epoch == early_num:
        #         net.load_state_dict(torch.load(PATH))
        #         break
        # else:
        #     early_epoch = 0

    net.load_state_dict(torch.load(PATH))
    print('epoch %d, loss %.4f, train acc %.3f, time %.1f sec'
          % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n,
             time.time() - start))



def rotation(X, INPUT_3D, test_rotation):
    if test_rotation == 12345:
        X = X[:,:,[2,1,3,4,6,5,7,8,0],:]
        return X
    if INPUT_3D:
        shape1,shape2,shape3,shape4,shape5 = X.shape
        X = X.reshape(shape1,shape3,shape4,shape5).transpose(1,3)
        X = torchvision.transforms.functional.rotate(X,test_rotation)
        X = X.transpose(1,3).reshape(shape1,shape2,shape3,shape4,shape5)
    else:
        X = torchvision.transforms.functional.rotate(X,test_rotation)
    return X



def pred_test(test_rotation, draw, rotation_list):
    global best_OA,best_pred_image
    draw_next = draw
    test_output = []
    with torch.no_grad():
        for X, y in test_iter:
            if test_rotation!=0:
                X = rotation(X, INPUT_3D, test_rotation)
            X = X.to(device)
            net.eval()
            test_output.extend(np.array(net(X).cpu().argmax(axis=1)))
    collections.Counter(test_output)
    gt_test = gt[test_indices] - 1

    overall_acc = metrics.accuracy_score(gt_test[:-VAL_SIZE],test_output)*100
    confusion_matrix = metrics.confusion_matrix(gt_test[:-VAL_SIZE],test_output)
    
    each_acc, average_acc = record.aa_and_each_accuracy(confusion_matrix)
    kappa = metrics.cohen_kappa_score(test_output, gt_test[:-VAL_SIZE])*100
    # print(confusion_matrix)
    each_acc*=100
    average_acc*=100
    
    
    if test_rotation==0:
        title = 'test     : '
        torch.save(net.state_dict(), "../save/" + model_info  + '.pt')
                   # '_' + str(round(overall_acc, 3))
    else:
        title = 'test %3d°: '%test_rotation
        
    if (overall_acc>best_OA and test_rotation==rotation_list[0]) or draw==True:
        if test_rotation==rotation_list[0]:
            best_OA = overall_acc
            draw_next = True
            best_pred_image = record.generate_png(title, all_iter, net, gt_hsi,
                          args.dataset, device, total_indices, BASIC_SIZE, ground_truth=False,
                          transpose=args.transpose)
        if test_rotation==rotation_list[-1]:
            best_pred_image += record.generate_png(title, all_iter, net, gt_hsi,
                          args.dataset, device, total_indices, BASIC_SIZE, ground_truth=True,
                          transpose=args.transpose)
        else:
            best_pred_image += record.generate_png(title, all_iter, net, gt_hsi,
                          args.dataset, device, total_indices, BASIC_SIZE, ground_truth=False,
                          transpose=args.transpose)
        best_pred_image += '\n\n'
    if test_rotation==rotation_list[-1]:
        draw_next = False
    output_confuse = pd.DataFrame(confusion_matrix, index = labels, columns=labels)
    
    
    
    ELEMENT_ACC[str(test_rotation)][index_iter, :] = each_acc
    KAPPA[str(test_rotation)].append(kappa)
    OA[str(test_rotation)].append(overall_acc)
    AA[str(test_rotation)].append(average_acc)
    if type(confuse[str(test_rotation)])==list:
        confuse[str(test_rotation)] = output_confuse
    else:
        confuse[str(test_rotation)] += output_confuse

    print(title, '%.4f  |  %.4f  |  %.4f  '%(overall_acc,average_acc,kappa))
    return draw_next
    

    
    
args.dataset = args.dataset.upper()
PARAM_OPTIM = args.optimizer
args.ratio = eval(args.ratio)
INPUT_3D = True if args.model_name in ['A2S2K'] else False
# # Data Loading
device = args.cuda if torch.cuda.is_available() else 'cpu'
args.device = device
# for Monte Carlo runs
# seeds = [1331, 1332, 1333, 1334, 1335, 1336, 1337, 1338, 1339, 1340, 1341]
seeds = np.arange(0,10000,100)
ensemble = 1
# # Training
# # Pytorch Data Loader Creation
data_hsi, gt_hsi, BASIC_SIZE, labels = load_dataset(args.dataset, args.pretreat)
print(data_hsi.shape)
image_x, image_y, args.band = data_hsi.shape
data = data_hsi.reshape(
    np.prod(data_hsi.shape[:2]), np.prod(data_hsi.shape[2:]))
gt = gt_hsi.reshape(np.prod(gt_hsi.shape[:2]), )
args.CLASSES_NUM = max(gt)
print('The class numbers of the HSI data is:', args.CLASSES_NUM)
print('-----Importing Setting Parameters-----')
ITER = args.iter
PATCH_LENGTH = args.patch
lr, num_epochs, batch_size = 0.001, 200, 32
loss = torch.nn.CrossEntropyLoss()
img_rows = 2 * PATCH_LENGTH + 1
img_cols = 2 * PATCH_LENGTH + 1
INPUT_DIMENSION = data_hsi.shape[2]
KAPPA = {'0':[],'90':[],'180':[],'270':[],'12345':[]}
OA = {'0':[],'90':[],'180':[],'270':[],'12345':[]}
AA = {'0':[],'90':[],'180':[],'270':[],'12345':[]}
ELEMENT_ACC = {'0':np.zeros((ITER, args.CLASSES_NUM)),
               '90':np.zeros((ITER, args.CLASSES_NUM)),
               '180':np.zeros((ITER, args.CLASSES_NUM)),
               '270':np.zeros((ITER, args.CLASSES_NUM)),
               '12345':np.zeros((ITER, args.CLASSES_NUM))}
confuse = {'0':[],'90':[],'180':[],'270':[],'12345':[]}
data = preprocessing.scale(data)
data_ = data.reshape(data_hsi.shape[0], data_hsi.shape[1], data_hsi.shape[2])
whole_data = data_
padded_data = np.lib.pad(
    whole_data, ((PATCH_LENGTH, PATCH_LENGTH), (PATCH_LENGTH, PATCH_LENGTH),
                 (0, 0)),
    'constant',
    constant_values=0)
best_OA = 0
TOTAL_TOTAL_SAMPLE = 0
TOTAL_TOTAL_TIME = 0


for index_iter in range(ITER):
    print('iter:', index_iter)
    #define the model
    net, generate_iter, model_info = generate_model(img_rows, data_hsi, args)
            
            
    
    if PARAM_OPTIM == 'diffgrad':
        optimizer = optim2.DiffGrad(
            net.parameters(),
            lr=lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0)  # weight_decay=0.0001)
    if PARAM_OPTIM == 'adam':
        optimizer = optim.Adam(
            net.parameters(),
            lr=1e-3,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0)
    time_1 = int(time.time())
    np.random.seed(seeds[index_iter])
    # train_indices, test_indices = select(gt)
    
    
    TOTAL_SIZE = (gt_hsi!=0).sum().sum()
    VALIDATION_SPLIT = args.valid_split
    train_indices, test_indices = geniter.sampling(VALIDATION_SPLIT, gt)
    _, total_indices = geniter.sampling(1, gt)

    TRAIN_SIZE = len(train_indices)
    print('Train size: ', TRAIN_SIZE)
    TEST_SIZE = TOTAL_SIZE - TRAIN_SIZE
    print('Test size: ', TEST_SIZE)
    VAL_SIZE = int(TRAIN_SIZE)
    print('Validation size: ', VAL_SIZE)

    print('-----Selecting Small Pieces from the Original Cube Data-----')
    train_iter, valida_iter, test_iter, all_iter = generate_iter(
        TRAIN_SIZE, train_indices, TEST_SIZE, test_indices, TOTAL_SIZE,
        total_indices, VAL_SIZE, whole_data, PATCH_LENGTH, padded_data,
        INPUT_DIMENSION, args.batch_size, gt)  #batchsize in 1
    if args.load:
        load_path = '../save/t2t_vit_model_%s.pt'%args.dataset
        save_path = '../extract_data/t2t_vit_model_%s.mat'%args.dataset
        geniter.extract_data(load_path, all_iter, net, gt_hsi, args.dataset, device, total_indices, save_path)
        break
    tic1 = time.time()
    net = net.to(device)
    train(
        net,
        train_iter,
        valida_iter,
        loss,
        optimizer,
        device,
        epochs=args.epoch)
    toc1 = time.time()
    
    print('               OA    |     AA    |  kappa  ')
    if args.test_rotation:
        draw=False
        for test_rotation in [0,90,180,270]:
            draw = pred_test(test_rotation, draw, [0,90,180,270])
    else:
        draw = pred_test(0, False, [0])
        
    print(TOTAL_TOTAL_TIME/TOTAL_TOTAL_SAMPLE)
        
        
    


print("--------" + " Training Finished-----------")


import record
best_OA = 0
record_str_ = ''
for test_rotation in [0,90,180,270]: 
    if test_rotation==0:
        title = 'test:'
        output_confuse = confuse[str(test_rotation)]/ITER
        output_confuse1 = (output_confuse.T*100/output_confuse.sum(axis=1)).round(2).T
        each_acc_sr = pd.Series(ELEMENT_ACC[str(test_rotation)].mean(0),
                                index=labels, name='mean acc').round(2)
        output_confuse = pd.concat([output_confuse.round(2),each_acc_sr], axis=1)
        output_confuse1 = pd.concat([output_confuse1,each_acc_sr], axis=1)
    else:
        title = 'test %d°:'%test_rotation
    
    
    record_str_ += record.record_str(title, OA[str(test_rotation)],
                    AA[str(test_rotation)], KAPPA[str(test_rotation)]
                    , ELEMENT_ACC[str(test_rotation)]) 



record.save_html([output_confuse,output_confuse1], record_str_,
    '../report/'  + model_info + '.html', img=best_pred_image)
    


