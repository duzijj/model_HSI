# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 08:59:12 2021

@author: Administrator
"""
import geniter,geniter3D
from models import cw_vit, vit, A2S2KResNet, SSRN, ResNet, PyResNet, ContextualNet
import scipy.io as sio
from sklearn.decomposition import PCA
import numpy as np

def generate_model(img_rows, data_hsi, args):
    model_name = args.model_name
    def set_model_info(report_attr):
        report_attr_str = '_' + '_%s__'.join(report_attr) + '_%s'
        report_attr_val = [args.__getattribute__(attr) for attr in report_attr]
        report_attr_str = report_attr_str%(tuple(report_attr_val))
        model_info = '%s_'%args.model_name + args.dataset + report_attr_str
        return model_info
    
    if model_name=='cw_vit':
        net = cw_vit.cw_vit(img_size=img_rows, in_chans=data_hsi.shape[2], args=args)
        generate_iter = geniter.generate_iter
        model_info = set_model_info(['patch','has_pe', 'ratio', 'pool','valid_split'])
        
        
    elif args.model_name=='hs_cw_vit':
        net = cw_vit.hs_cw_vit(img_size=img_rows, in_chans=data_hsi.shape[2], args=args)
        generate_iter = geniter.generate_iter
        model_info = set_model_info(['patch','has_pe', 'cw_pe', 'ratio', 'pool'])
        
    elif args.model_name=='A2S2K':
        net = A2S2KResNet.S3KAIResNet(args.band, args.CLASSES_NUM, 2, 
                                      PARAM_KERNEL_SIZE = args.kernel)
        model_info = set_model_info(['patch', 'batch_size', 'epoch', 'scheduler'])
        generate_iter = geniter3D.generate_iter

    
    elif args.model_name=='decay_vit':
        net = vit.decay_vit(img_size=img_rows, in_chans=data_hsi.shape[2], args=args)
        generate_iter = geniter.generate_iter
        model_info = set_model_info(['patch', 'has_pe','cw_pe','valid_split','ratio', 'patch_size'])
        
    
        
    elif args.model_name=='vit':
        net = vit.ViT(image_size=img_rows, channels=data_hsi.shape[2], patch_size=args.patch_size,
                             num_classes=args.CLASSES_NUM, depth=args.depth, mlp_dim=args.mlp_dim,
                             heads=args.heads, dim=args.dim, dropout = args.dropout,
                             emb_dropout = args.emb_dropout,
                             set_embed_linear = args.set_embed_linear)
        generate_iter = geniter.generate_iter
        model_info = set_model_info(['patch', 'pretreat', 'valid_split', 'patch_size'])
    
    elif args.model_name=='SSRN':
        net = SSRN.SSRN_network(args.band, args.CLASSES_NUM)
        model_info = set_model_info(['patch', 'batch_size', 'epoch', 'scheduler'])
        generate_iter = geniter3D.generate_iter
        
    elif args.model_name=='ResNet':
        net = ResNet.ResNet34(in_shape=(args.band, img_rows, img_rows), num_classes=args.CLASSES_NUM)
        model_info = set_model_info(['patch', 'batch_size', 'epoch', 'scheduler'])
        generate_iter = geniter.generate_iter
    
    
    elif args.model_name=='PyResNet':
        net = PyResNet.PyResNet34(in_shape=(args.band, img_rows, img_rows), num_classes=args.CLASSES_NUM)
        model_info = set_model_info(['patch', 'batch_size', 'epoch', 'scheduler'])
        generate_iter = geniter.generate_iter
        
    elif args.model_name=='ContextualNet':
        net = ContextualNet.LeeEtAl(args.band, args.CLASSES_NUM)
        model_info = set_model_info(['patch', 'batch_size', 'epoch', 'scheduler'])
        generate_iter = geniter3D.generate_iter

    return net, generate_iter, model_info
        


def load_dataset(Dataset, pretreat):
    data_path = '../datasets/'
    if Dataset == 'IN':
        mat_data = sio.loadmat(data_path + 'Indian_pines_corrected.mat')
        mat_gt = sio.loadmat(data_path + 'Indian_pines_gt.mat')
        data_hsi = mat_data['indian_pines_corrected']
        gt_hsi = mat_gt['indian_pines_gt']
        labels = ['Alfalfa','Corn-notill','Corn-mintill','Corn','Grass-pasture','Grass-trees','Grass-pasture-mowed',
                 'Hay-windrowed','Oats','Soybean-notill','Soybean-mintill','Soybean-clean','Wheat','Woods',
                 'Buildings-Grass-Trees-Drives','Stone-Steel-Towers']
        BASIC_SIZE = 4
        
    elif Dataset == 'UP':
        uPavia = sio.loadmat(data_path + 'PaviaU.mat')
        gt_uPavia = sio.loadmat(data_path + 'PaviaU_gt.mat')
        data_hsi = uPavia['paviaU']
        gt_hsi = gt_uPavia['paviaU_gt']
        labels = ['Asphalt','Meadows','Gravel','Trees','Painted metal sheets',
                  'Bare Soil','Bitumen','Self-Blocking Bricks','Shadows']
        BASIC_SIZE = 12

    elif Dataset == 'SV':
        SV = sio.loadmat(data_path + 'Salinas_corrected.mat')
        gt_SV = sio.loadmat(data_path + 'Salinas_gt.mat')
        data_hsi = SV['salinas_corrected']
        gt_hsi = gt_SV['salinas_gt']
        labels = ['Brocoli_green_weeds_1','Brocoli_green_weeds_2','Fallow','Fallow_rough_plow','Fallow_smooth','Stubble',
         'Celery','Grapes_untrained','Soil_vinyard_develop','Corn_senesced_green_weeds','Lettuce_romaine_4wk',
         'Lettuce_romaine_5wk','Lettuce_romaine_6wk','Lettuce_romaine_7wk','Vinyard_untrained','Vinyard_vertical_trellis']       
        BASIC_SIZE = 12

    elif Dataset == 'KSC':
        KSC = sio.loadmat(data_path + 'KSC.mat')
        gt_KSC = sio.loadmat(data_path + 'KSC_gt.mat')
        data_hsi = KSC['KSC']
        gt_hsi = gt_KSC['KSC_gt']
        labels = ['Scrub','Willow swamp','CP hammock','CP/Oak','Slash pine','Oak/Broadleaf','Hardwood swamp',
                  'Graminoid marsh','Spartina marsh','Catiail marsh','Salt marsh','Mud flats','Water']
        BASIC_SIZE = 12
        
    elif Dataset == 'PAVIA':
        Pavia = sio.loadmat(data_path + 'Pavia.mat')
        gt_Pavia = sio.loadmat(data_path + 'Pavia_gt.mat')
        data_hsi = Pavia['pavia']
        gt_hsi = gt_Pavia['pavia_gt']
        labels = ['Water','Trees','Asphalt','Self-Blocking Bricks','Bitumen','Tiles','Shadows','Meadows','Bare Soil']
        BASIC_SIZE = 12

    elif Dataset == 'BOT':
        BOT = sio.loadmat(data_path + 'Botswana.mat')
        gt_BOT = sio.loadmat(data_path + 'Botswana_gt.mat')
        data_hsi = BOT['Botswana']
        gt_hsi = gt_BOT['Botswana_gt']
        labels = ['Water','Hippo grass','Floodplain grasses 1','Floodplain grasses 2','Reeds','Riparian',
                'Firescar','Island interior','Acacia woodlands','Acacia shrublands','Acacia grasslands',
                'Short mopane','Mixed mopane','Exposed soils']
        BASIC_SIZE = 30
        
        
    K = data_hsi.shape[2]
    shapeor = data_hsi.shape
    data_hsi = data_hsi.reshape(-1, data_hsi.shape[-1])
    shapeor = np.array(shapeor)
    shapeor[-1] = K
    if(pretreat=='PCA'):
        data_hsi = PCA(n_components=K).fit_transform(data_hsi)
        print(pretreat)
    elif(pretreat=='1'):
          
        data_hsi_mean = np.mean(data_hsi, axis=0)       
        ######
        # data_scaled = (data_hsi-data_hsi_mean)/data_hsi_mean
        # np.abs(data_scaled).mean()
        ######
        # data_scaled = data_hsi-data_hsi_mean
        # data_positive = (data_scaled>0).astype(int)
        # data_positive[data_positive==0] = -1
        # data_scaled = np.abs(data_scaled)
        # data_scaled_min = np.min(data_scaled, axis=0)
        # data_scaled_max = np.max(data_scaled, axis=0)
        # data_hsi = (data_scaled-np.mean(data_scaled, axis=0))/(data_scaled_max-data_scaled_min)*data_positive
        
        data_hsi = data_hsi-data_hsi_mean
        print(pretreat)
        # np.abs(data_hsi).mean()
    elif(pretreat=='1_PCA'):
        data_hsi_mean = np.mean(data_hsi, axis=0)
        data_hsi = data_hsi-data_hsi_mean
        data_hsi = PCA(n_components=K).fit_transform(data_hsi)
        print(pretreat)
    elif(pretreat=='k7'):
        output = []
        for i in range(K-6):
            pca_1 = PCA(n_components=1).fit_transform(data_hsi[:,i:i+7])
            output = np.append(output, pca_1, axis=1) if len(output)>0 else pca_1
        shapeor[2]-=6
        data_hsi = output
        print(pretreat)
        
    data_hsi = data_hsi.reshape(shapeor)

    return data_hsi, gt_hsi, BASIC_SIZE, labels