import numpy as np
import torch
from operator import truediv
import re
import matplotlib.pyplot as plt
from io import BytesIO
import base64


def evaluate_accuracy(data_iter, net, loss, device):
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            test_l_sum, test_num = 0, 0
            #X = X.permute(0, 3, 1, 2)
            X = X.to(device)
            y = y.to(device)
            net.eval() 
            y_hat = net(X)
            l = loss(y_hat, y.long())
            acc_sum += (y_hat.argmax(dim=1) == y.to(device)).float().sum().cpu().item()
            test_l_sum += l
            test_num += 1
            net.train() 
            n += y.shape[0]
    return [acc_sum / n, test_l_sum] # / test_num]


def aa_and_each_accuracy(confusion_matrix):
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc



def record_output(oa_ae, aa_ae, kappa_ae, element_acc_ae, training_time_ae, testing_time_ae, path):
    f = open(path, 'a')
    sentence0 = 'OAs for each iteration are:' + str(oa_ae) + '\n'
    f.write(sentence0)
    sentence1 = 'AAs for each iteration are:' + str(aa_ae) + '\n'
    f.write(sentence1)
    sentence2 = 'KAPPAs for each iteration are:' + str(kappa_ae) + '\n' + '\n'
    f.write(sentence2)
    sentence3 = 'mean_OA ± std_OA is: ' + str(np.mean(oa_ae)) + ' ± ' + str(np.std(oa_ae)) + '\n'
    f.write(sentence3)
    sentence4 = 'mean_AA ± std_AA is: ' + str(np.mean(aa_ae)) + ' ± ' + str(np.std(aa_ae)) + '\n'
    f.write(sentence4)
    sentence5 = 'mean_KAPPA ± std_KAPPA is: ' + str(np.mean(kappa_ae)) + ' ± ' + str(np.std(kappa_ae)) + '\n' + '\n'
    f.write(sentence5)
    sentence6 = 'Total average Training time is: ' + str(np.sum(training_time_ae)) + '\n'
    f.write(sentence6)
    sentence7 = 'Total average Testing time is: ' + str(np.sum(testing_time_ae)) + '\n' + '\n'
    f.write(sentence7)
    element_mean = np.mean(element_acc_ae, axis=0)
    element_std = np.std(element_acc_ae, axis=0)
    sentence8 = "Mean of all elements in confusion matrix: " + str(element_mean) + '\n'
    f.write(sentence8)
    sentence9 = "Standard deviation of all elements in confusion matrix: " + str(element_std) + '\n'
    f.write(sentence9)
    f.close()





def table_data1(columns):
    #没有index只有column的表格
    TABLE_HEAD = '''<tr id='header_row' class="text-center success" style="font-weight: bold;font-size: 14px;">\n'''
    TABLE_DATA = '''<tr class='failClass warning'>\n'''
    TABLE_POS = '''<col align='left' />'''
    for i in columns:
        TABLE_HEAD += '''<th>%s</th>\n'''%i
        TABLE_DATA += '<td>%(' + str(i) + ')s</td>\n'
        if i != columns[-1]:
            TABLE_POS += '''\n<col align='left' />'''
    TABLE_HEAD += '''</tr>'''
    TABLE_DATA += '''</tr>'''
    return TABLE_POS,TABLE_HEAD,TABLE_DATA
    
def table_data(columns):
    #带有index的表格
    TABLE_HEAD = '''<tr id='header_row' class="text-center success" style="font-weight: bold;font-size: 14px;">\n
                    <th width="30" height="30">index</th>\n'''
    TABLE_DATA = '''<tr class='failClass warning'>\n<td width="30" height="30"><strong>%(index)s</strong></td>\n'''
    TABLE_POS = '''<col align='left'/>'''
    for i in columns:
        TABLE_HEAD += '''<th width="30" height="30">%s</th>\n'''%i
        TABLE_DATA += '<td width="30" height="30">%(' + str(i) + ')s</td>\n'
        TABLE_POS += '''\n<col align='right' />'''
    TABLE_HEAD += '''</tr>'''
    TABLE_DATA += '''</tr>'''
    return TABLE_POS,TABLE_HEAD,TABLE_DATA

HTML_TMPL = r"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>测试报告</title>
        <link href="http://libs.baidu.com/bootstrap/3.0.3/css/bootstrap.min.css" rel="stylesheet">
        <h1 style="font-family: Microsoft YaHei">测试报告</h1>
        <p class='attribute'><strong>测试结果 : <br></strong> %(report_result)s</p>
        <style type="text/css" media="screen">
    body  { font-family: Microsoft YaHei,Tahoma,arial,helvetica,sans-serif;padding: 20px;}
    </style>
    </head>
    <body>
        %(body_data)s
    </body>
    </html>"""
    
HTML_TMPL_img = r"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>测试报告</title>
        <link href="http://libs.baidu.com/bootstrap/3.0.3/css/bootstrap.min.css" rel="stylesheet">
        <h1 style="font-family: Microsoft YaHei">测试报告</h1>
        <p class='attribute'><strong>测试结果 : <br></strong> %(report_result)s</p>
        <style type="text/css" media="screen">
    body  { font-family: Microsoft YaHei,Tahoma,arial,helvetica,sans-serif;padding: 20px;}
    </style>
    </head>
    <body>
        %(body_data)s
    </body>
    %(img)s
    </html>"""
        
body = '''<table id='result_table' class="table table-condensed table-bordered table-hover">
        <colgroup>
            %(col_pos)s
        </colgroup>
        %(table_head)s
        %(table_tr)s
        </table>
        '''
def record_str(title, oa_ae, aa_ae, kappa_ae, element_acc_ae):
    title = '<br>' + title + '<br>'
    sentence3 = '&ensp;|&ensp;' + 'OA&emsp;&ensp;:&ensp;' + '%.4f'%np.mean(oa_ae) + ' ± ' + '%.4f'%np.std(oa_ae)
    sentence4 = '&ensp;|&ensp;' + 'AA&emsp;&ensp;:&ensp;' + '%.4f'%np.mean(aa_ae) + ' ± ' + '%.4f'%np.std(aa_ae)
    sentence5 = '&ensp;|&ensp;' + 'KAPPA:&ensp;' + '%.4f'%np.mean(kappa_ae) + ' ± ' + '%.4f'%np.std(kappa_ae)
    
    # np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
    oa_ae = np.array(oa_ae).round(2)
    aa_ae = np.array(aa_ae).round(2)
    kappa_ae = np.array(kappa_ae).round(2)
    sentence0 = 'OAs&emsp;&emsp;:&ensp;' + str(oa_ae) + sentence3 + '<br>'
    sentence1 = 'AAs&emsp;&emsp;:&ensp;' + str(aa_ae) + sentence4 + '<br>'
    sentence2 = 'KAPPAs:&ensp;' + str(kappa_ae) + sentence5 + '<br>'
    
    
    # sentence6 = 'Total average Training time is: ' + str(np.sum(training_time_ae)) + '<br>'
    # sentence7 = 'Total average Testing time is: ' + str(np.sum(testing_time_ae)) + '<br>' + '<br>'
    element_mean = np.mean(element_acc_ae, axis=0).round(3)
    element_std = np.std(element_acc_ae, axis=0).round(3)
    sentence8 = "Mean of all elements in confusion matrix: " + str(element_mean) + '<br>'
    sentence9 = "Standard deviation of all elements in confusion matrix: " + str(element_std) + '<br>'
    sentence = title + sentence0 + sentence1 + sentence2 + sentence8 + sentence9
        
    return sentence
    


def df_to_html(data, record, img=None):
    body_data = ''
    if type(data)!=list:
        data = [data]
    for d in data:
        TABLE_POS,TABLE_HEAD,TABLE_DATA = table_data(d.columns)
        table_tr = ''
        report_result = record
        for i in range(len(d)):
            dict1 = d.iloc[i].to_dict()
            dict1['index'] = d.index[i]
            table_tr += TABLE_DATA % dict1
        body_data += body % dict(col_pos=TABLE_POS,table_head=TABLE_HEAD,table_tr=table_tr)
        
    if img:
        if type(img)==dict:
            img = '\n<br>'.join(img)
        output_dict=dict(report_result=report_result,
                         body_data = body_data,
                         img = img
                         )
        return HTML_TMPL_img % output_dict
    else:
        output_dict=dict(report_result=report_result,
                         body_data = body_data
                         )
        
        return HTML_TMPL % output_dict

def save_html(data, record, path, img=None):
    output = df_to_html(data, record, img)
    with open(path, 'wb') as f:
        f.write(output.encode('utf8'))


def classification_map(title, map, basic_size, save_path=None, save_as_html=True):
    fig = plt.figure(frameon=False)
    fig.set_size_inches(basic_size, map.shape[1]/map.shape[0]*basic_size)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    fig.add_axes(ax)
    ax.imshow(map)
    plt.title(title, fontsize = 15, color = 'black')

    if save_as_html:
        figfile = BytesIO()
        plt.savefig(figfile, format='png')
        figfile.seek(0)
        figdata_png = base64.b64encode(figfile.getvalue()) # 将图片转为base64
        figdata_str = str(figdata_png, "utf-8") # 提取base64的字符串，不然是b'xxx'
        # 保存为.html
        img = '<img src=\"data:image/png;base64,{}\"/>'.format(figdata_str)
        return img
    else:
        fig.savefig(save_path, dpi=100)
        return 0

color_map = [np.array([255, 0, 0]) / 255.,
            np.array([0, 255, 0]) / 255.,
            np.array([0, 0, 255]) / 255.,
            np.array([255, 255, 0]) / 255.,
            np.array([0, 255, 255]) / 255.,
            np.array([255, 0, 255]) / 255.,
            np.array([192, 192, 192]) / 255.,
            np.array([128, 128, 128]) / 255.,
            np.array([128, 0, 0]) / 255.,
            np.array([128, 128, 0]) / 255.,
            np.array([0, 128, 0]) / 255.,
            np.array([128, 0, 128]) / 255.,
            np.array([0, 128, 128]) / 255.,
            np.array([0, 0, 128]) / 255.,
            np.array([255, 165, 0]) / 255.,
            np.array([255, 215, 0]) / 255.,
            np.array([0, 0, 0]) / 255.,
            np.array([215, 255, 0]) / 255.,
            np.array([0, 255, 215]) / 255.,
            np.array([0, 0, 0]) / 255.]




def list_to_colormap(x_list):
    y = np.zeros((x_list.shape[0], 3))
    for index, item in enumerate(x_list):
        y[index] = color_map[int(item)]
    return y





def generate_png(title, all_iter, net, gt_hsi, Dataset, device, total_indices,
                 basic_size, path=None, ground_truth=False,transpose=False):
    pred_test = []
    for X, y in all_iter:
        #X = X.permute(0, 3, 1, 2)
        X = X.to(device)
        net.eval()
        pred_test.extend(net(X).cpu().argmax(axis=1).detach().numpy())
    gt = gt_hsi.flatten()
    x_label = np.zeros(gt.shape)
    for i in range(len(gt)):
        if gt[i] == 0:
            gt[i] = 17
            x_label[i] = 16
    gt = gt[:] - 1
    x_label[total_indices] = pred_test
    x = np.ravel(x_label)
    y_list = list_to_colormap(x)
    y_gt = list_to_colormap(gt)
    y_re = np.reshape(y_list, (gt_hsi.shape[0], gt_hsi.shape[1], 3))
    gt_re = np.reshape(y_gt, (gt_hsi.shape[0], gt_hsi.shape[1], 3))
    if transpose:
        y_re = y_re.transpose(1,0,2)
        gt_re = gt_re.transpose(1,0,2)
    if ground_truth:
        img = classification_map(title, y_re, basic_size)+\
            classification_map(title, gt_re, basic_size)
    else:
        img = classification_map(title, y_re, basic_size)
    return img
    
    
def generate_html(all_iter, net, gt_hsi, Dataset, device, total_indices, path):
    pred_test = []
    for X, y in all_iter:
        #X = X.permute(0, 3, 1, 2)
        X = X.to(device)
        net.eval()
        pred_test.extend(net(X).cpu().argmax(axis=1).detach().numpy())
    gt = gt_hsi.flatten()
    x_label = np.zeros(gt.shape)
    for i in range(len(gt)):
        if gt[i] == 0:
            gt[i] = 17
            x_label[i] = 16
    gt = gt[:] - 1
    x_label[total_indices] = pred_test
    x = np.ravel(x_label)
    x_re = np.reshape(x, (gt_hsi.shape[0], gt_hsi.shape[1]))
    gt_re = np.reshape(gt, (gt_hsi.shape[0], gt_hsi.shape[1]))
    hot_point = []
    for i in range(gt_hsi.shape[0]):
        for j in range(gt_hsi.shape[1]):
            hot_point += [[j,gt_hsi.shape[0]-1-i,x_re[i,j]]]
