import torch
import numpy as np
import torch.utils.data as Data

import scipy.io as sio



def index_assignment(index, row, col, pad_length):
    new_assign = {}
    for counter, value in enumerate(index):
        assign_0 = value // col + pad_length
        assign_1 = value % col + pad_length
        new_assign[counter] = [assign_0, assign_1]
    return new_assign

def select_patch(matrix, pos_row, pos_col, ex_len):
    selected_rows = matrix[range(pos_row-ex_len, pos_row+ex_len+1)]
    selected_patch = selected_rows[:, range(pos_col-ex_len, pos_col+ex_len+1)]
    return selected_patch


def select_small_cubic(data_size, data_indices, whole_data, patch_length, padded_data, dimension):
    small_cubic_data = np.zeros((data_size, 2 * patch_length + 1, 2 * patch_length + 1, dimension))
    data_assign = index_assignment(data_indices, whole_data.shape[0], whole_data.shape[1], patch_length)
    for i in range(len(data_assign)):
        small_cubic_data[i] = select_patch(padded_data, data_assign[i][0], data_assign[i][1], patch_length)
    return small_cubic_data


def generate_iter(TRAIN_SIZE, train_indices, TEST_SIZE, test_indices, TOTAL_SIZE, total_indices, VAL_SIZE,
                  whole_data, PATCH_LENGTH, padded_data, INPUT_DIMENSION, batch_size, gt):
    gt_all = gt[total_indices] - 1
    y_train = gt[train_indices] - 1
    y_test = gt[test_indices] - 1

    all_data =  select_small_cubic(TOTAL_SIZE, total_indices, whole_data,
                                                      PATCH_LENGTH, padded_data, INPUT_DIMENSION)

    train_data = select_small_cubic(TRAIN_SIZE, train_indices, whole_data,
                                                        PATCH_LENGTH, padded_data, INPUT_DIMENSION)
    print(train_data.shape)
    test_data =  select_small_cubic(TEST_SIZE, test_indices, whole_data,
                                                       PATCH_LENGTH, padded_data, INPUT_DIMENSION)
    x_train = train_data.reshape(train_data.shape[0], train_data.shape[1], train_data.shape[2], INPUT_DIMENSION)
    x_test_all = test_data.reshape(test_data.shape[0], test_data.shape[1], test_data.shape[2], INPUT_DIMENSION)

    x_val = x_test_all[-VAL_SIZE:]
    y_val = y_test[-VAL_SIZE:]

    x_test = x_test_all[:-VAL_SIZE]
    y_test = y_test[:-VAL_SIZE]
    
    x1_tensor_train = torch.from_numpy(x_train).type(torch.FloatTensor).transpose(1,3)
    y1_tensor_train = torch.from_numpy(y_train).type(torch.FloatTensor)
    torch_dataset_train = Data.TensorDataset(x1_tensor_train, y1_tensor_train)

    x1_tensor_valida = torch.from_numpy(x_val).type(torch.FloatTensor).transpose(1,3)
    y1_tensor_valida = torch.from_numpy(y_val).type(torch.FloatTensor)
    torch_dataset_valida = Data.TensorDataset(x1_tensor_valida, y1_tensor_valida)

    x1_tensor_test = torch.from_numpy(x_test).type(torch.FloatTensor).transpose(1,3)
    y1_tensor_test = torch.from_numpy(y_test).type(torch.FloatTensor)
    torch_dataset_test = Data.TensorDataset(x1_tensor_test,y1_tensor_test)

    all_data.reshape(all_data.shape[0], all_data.shape[1], all_data.shape[2], INPUT_DIMENSION)
    all_tensor_data = torch.from_numpy(all_data).type(torch.FloatTensor).transpose(1,3)
    all_tensor_data_label = torch.from_numpy(gt_all).type(torch.FloatTensor)
    torch_dataset_all = Data.TensorDataset(all_tensor_data, all_tensor_data_label)


    train_iter = Data.DataLoader(
        dataset=torch_dataset_train,  # torch TensorDataset format
        batch_size=batch_size,  # mini batch size
        shuffle=True,  
        num_workers=0, 
    )
    valiada_iter = Data.DataLoader(
        dataset=torch_dataset_valida,  # torch TensorDataset format
        batch_size=batch_size,  # mini batch size
        shuffle=True,  
        num_workers=0, 
    )
    test_iter = Data.DataLoader(
        dataset=torch_dataset_test,  # torch TensorDataset format
        batch_size=batch_size,  # mini batch size
        shuffle=False, 
        num_workers=0, 
    )
    all_iter = Data.DataLoader(
        dataset=torch_dataset_all,  # torch TensorDataset format
        batch_size=batch_size,  # mini batch size
        shuffle=False, 
        num_workers=0, 
    )
    return train_iter, valiada_iter, test_iter, all_iter #, y_test


def extract_data(load_path, all_iter, net, gt_hsi, Dataset, device, total_indices, save_path):
    
    state_dict = torch.load(load_path)
    net.load_state_dict(state_dict)
    net.to(device)
    extracted = []
    # ori_data = []
    label = []
    pred_test = []
    for X, y in all_iter:
        #X = X.permute(0, 3, 1, 2)
        X = X.to(device)
        net.eval()
        pred_test.extend(net(X).cpu().argmax(axis=1).detach().numpy())
        # ori_data.extend(X.cpu().detach().numpy())
        label.extend(y.detach().numpy())
        extracted.extend(net.extract_parameter().cpu().detach().numpy())
    gt = gt_hsi.flatten()
    sio.savemat(save_path,{'data':extracted,
                            # 'cube_data':ori_data,
                            'indices':total_indices,
                            'label':label,
                            'model_output':pred_test,
                            'gt':gt,
                            'gt_as_matrix':gt_hsi
                                  })
    
def sampling(proportion, ground_truth):
    train = {}
    test = {}
    labels_loc = {}
    m = max(ground_truth)
    for i in range(m):
        indexes = [
            j for j, x in enumerate(ground_truth.ravel().tolist())
            if x == i + 1
        ]
        np.random.shuffle(indexes)
        labels_loc[i] = indexes
        if proportion != 1:
            nb_val = max(int((1 - proportion) * len(indexes)), 3)
        else:
            nb_val = 0
        train[i] = indexes[:nb_val]
        test[i] = indexes[nb_val:]
    train_indexes = []
    test_indexes = []
    for i in range(m):
        train_indexes += train[i]
        test_indexes += test[i]
    np.random.shuffle(train_indexes)
    np.random.shuffle(test_indexes)
    return train_indexes, test_indexes

def select(groundTruth, Dataset):  #divide dataset into train and test datasets
    labels_loc = {}
    train = {}
    test = {}
    m = max(groundTruth)
    #amount = [3, 41, 29, 7, 14, 20, 2, 15, 3, 36, 64, 22, 4, 28, 10, 2]
    #amount = [43, 1387, 801, 230, 469, 710, 26, 463, 17, 936, 2391, 571, 201, 1237, 376, 91]
    if Dataset == 'IN':
        amount = [
            35, 1011, 581, 167, 344, 515, 19, 327, 12, 683, 1700, 418, 138,
            876, 274, 69
        ]  #IP 20%
    #amount = [6, 144, 84, 24, 50, 75, 3, 49, 2, 97, 247, 62, 22, 130, 38, 10]   #IP 20%
    if Dataset == 'UP':
        amount = [5297, 14974, 1648, 2424, 1076, 4026, 1046, 2950, 755]  #UP
    if Dataset == 'KSC':
        amount = [
            530, 165, 176, 170, 110, 161, 80, 299, 377, 283, 296, 341, 654
        ]  #KSC
    for i in range(m):
        indices = [
            j for j, x in enumerate(groundTruth.ravel().tolist()) if x == i + 1
        ]
        np.random.shuffle(indices)
        labels_loc[i] = indices
        nb_val = int(amount[i])
        train[i] = indices[:-nb_val]
        test[i] = indices[-nb_val:]
#    whole_indices = []
    train_indices = []
    test_indices = []
    for i in range(m):
        #        whole_indices += labels_loc[i]
        train_indices += train[i]
        test_indices += test[i]
    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)
    return train_indices, test_indices