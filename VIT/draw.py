# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 22:40:51 2021

@author: Administrator
"""
import webbrowser
from pyecharts import options as opts
from pyecharts.charts import HeatMap,Grid
from pyecharts.faker import Faker
from pyecharts.options.global_options import ThemeType
import random
import numpy as np
color_binary = ['#FF0000','#00FF00','#0000FF','#FFFF00','#00FFFF','#FF00FF',
    '#C0C0C0','#808080','#800000','#808000','#008000','#800080',
    '#008080','#000080','#FFA500','#FFD700','#000000','#D7FF00',
    '#00FFD7','#000000']

color_binary = ['#FF0000','#00FF00','#0000FF','#FFFF00','#00FFFF','#FF00FF',
    '#C0C0C0','#808080','#800000','#808000','#008000','#800080',
    '#008080','#111180','#FFA500','#FFD700','#000000','#D7FF00',
    '#00FFD7','#000000']

grid=Grid()
hm = HeatMap(
    init_opts=opts.InitOpts(bg_color='black')
    )
# 使用列表表达式创建7*24的二维列表
for xx in range(x.astype(int).max()-1):
    data = np.array(hot_point)[x==xx]
# np.array([[i,j,random.randint(10,200)] for i in range(222) for j in range(222)])
# hm.add_xaxis(np.arange(10))
    hm.add_yaxis(str(xx)+'_'+labels[int(xx)],None,data.tolist())
hm.set_global_opts(
    
    # title_opts=opts.TitleOpts(title="热力图基本示例"),
    # tooltip_opts=opts.TooltipOpts(axis_pointer_type="cross"),# 指示器类型 横向纵向指示
    # visualmap_opts=opts.VisualMapOpts(min_=0,max_=200,
    #                                    range_color=color_binary,
    #                                   is_piecewise=True,is_show=False)
    # graphic_opts = opts.GraphicItem(width=100,height=100)
    legend_opts = opts.LegendOpts(pos_right='0%',orient = 'vertical')
)

# 仅使用pos_top修改相对顶部的位置

grid.add(hm,grid_opts=opts.GridOpts(pos_right="30%"))
grid.render("render.html")
webbrowser.open('render.html')




dir(opts)

graphic_opts
itemstyle_opts

opts.ItemStyleOpts()
title_opts: types.Title = opts.TitleOpts(),
legend_opts: types.Legend = opts.LegendOpts(),
tooltip_opts: types.Tooltip = None,
toolbox_opts: types.Toolbox = None,
brush_opts: types.Brush = None,
xaxis_opts: types.Axis = None,
yaxis_opts: types.Axis = None,
visualmap_opts: types.VisualMap = None,
datazoom_opts: types.DataZoom = None,
graphic_opts: types.Graphic = None,
axispointer_opts: types.AxisPointer = None,



import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64


x=np.arange(0,10,0.01)
y=np.sin(x)
plt.figure()
plt.plot(x,y)
# plt.show()

# 转base64
figfile = BytesIO()
plt.savefig(figfile, format='png')
figfile.seek(0)
figdata_png = base64.b64encode(figfile.getvalue()) # 将图片转为base64
figdata_str = str(figdata_png, "utf-8") # 提取base64的字符串，不然是b'xxx'

# 保存为.html
html = '<img src=\"data:image/png;base64,{}\"/>'.format(figdata_str)
filename='png.html'
with open(filename,'w') as f:
    f.write(html)







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

import matplotlib.pyplot as plt

data = [5, 20, 15, 25, 10]
fig = plt.figure(figsize=(20,20))
fig.set_dpi(1000)
for i in range(len(labels)):
    exec('''rects%d = plt.bar(range(len(data)), data,color=color_map[%d],label='%s')'''%(i,i, labels[i]))
plt.legend()
fig.get_children()[1].get_children()[-2].legendPatch.set_facecolor(np.array([255, 255, 255]) / 255.)
plt.show()






import pandas as pd
desktop = r'C:/Users/Administrator/Desktop'



data = pd.read_excel(desktop + '\\output.xlsx')






# def get_22(x):
#     x = x.replace(' ','').split('±')
#     x = ['%.2f'%float(i) for i in x]
#     return '±'.join(x)
# data['OA'] = data['OA'].apply(get_22)
# data['AA'] = data['AA'].apply(get_22)
# data['kappa*100'] = data['kappa*100'].apply(get_22)
# data.to_excel(desktop + '\\output.xlsx', index=False)



# def get_2(a):
#     a = a.replace('[','').replace(']','').split()
#     a = ['%.2f'%float(i) for i in a]
#     return a
# data['mean1'] = data['mean'].apply(get_2)
# data['std1'] = data['std'].apply(get_2)
# def add_2(x):
#     output = []
#     print(x['mean1'])
#     for i in range(len(x['mean1'])):
#         output += [x['mean1'][i]+'±'+x['std1'][i]]
#     return ','.join(output)
# data['output'] = data[['mean1','std1']].apply(add_2, axis=1)

# data[['dataset','patch','sin位置编码','加法位置编码','OA','AA','kappa*100','mean','std','output']].to_excel(desktop + '\\output.xlsx', index=False)




head = ['OA','AA','$\kappa \\times 100$']
dataset = 'SV'


label = {
'IN':head + ['Alfalfa','Corn-notill','Corn-mintill','Corn','Grass-pasture','Grass-trees','Grass-pasture-mowed',
                 'Hay-windrowed','Oats','Soybean-notill','Soybean-mintill','Soybean-clean','Wheat','Woods',
                 'Buildings-Grass-Trees-Drives','Stone-Steel-Towers'],
'UP':head + ['Asphalt','Meadows','Gravel','Trees','Painted metal sheets',
                  'Bare Soil','Bitumen','Self-Blocking Bricks','Shadows'],
'SV':head + ['Brocoli_green_weeds_1','Brocoli_green_weeds_2','Fallow','Fallow_rough_plow','Fallow_smooth','Stubble',
         'Celery','Grapes_untrained','Soil_vinyard_develop','Corn_senesced_green_weeds','Lettuce_romaine_4wk',
         'Lettuce_romaine_5wk','Lettuce_romaine_6wk','Lettuce_romaine_7wk','Vinyard_untrained','Vinyard_vertical_trellis']

}

for i in range(len(label['SV'])):
    label['SV'][i] = label['SV'][i].replace('_','\_')



output_IN = data[data.dataset=='%s数据集'%dataset].iloc[[4,0,1,2,3],:]


def get_list(x):
    y = x[['OA','AA','kappa*100']].tolist() + x.output.split(',')
    return y

y = output_IN.apply(get_list, axis=1)
z = pd.DataFrame(y.values.tolist()).T
for i in range(len(z)):
    latex_line = [label[dataset][i]] + z.iloc[i].values.tolist()
    print(' & '.join(latex_line), '\\\\')













net.eval()
x = X
y = X[:,:,[1,0,2,3,4,5,6,7,8],:]

a = net(x)
b = net(y)



a
b





x = X.clone()
x,all_index,_ = self.clockwise_input(x, edge=self.img_size)
if self.has_map_ratio:
    x = x*self.map_ratio
x = x.transpose(1,2)
if self.set_embed_linear:
    x = self.embed_linear(x)
b, n, d = x.shape


cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
x = torch.cat((cls_tokens, x), dim=1)
    
x = self.emb_dropout(x)
# a = self.transformer(x)
a = self.mlp_head(self.transformer(x)[:,0])
# a = self.mlp_head(self.transformer(x).mean(dim=1))


y = X.clone()
cc = y[:,:,0,0].clone()
y[:,:,0,0] = y[:,:,0,8]
y[:,:,0,8] = cc
y,all_index,_ = self.clockwise_input(y, edge=self.img_size)
if self.has_map_ratio:
    y = y*self.map_ratio
y = y.transpose(1,2)
if self.set_embed_linear:
    y = self.embed_linear(y)
b, n, d = y.shape



cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
y = torch.cat((cls_tokens, y), dim=1)
    
y = self.emb_dropout(y)
# b = self.transformer(y)
b = self.mlp_head(self.transformer(y)[:,0])
# b = self.mlp_head(self.transformer(x).mean(dim=1))


x==y
a
b

x[:,:,0,8]
y[:,:,0,0]
y[:,:,0,8]
x[:,:,0,0]


x[:,8,:]
y[:,0,:]
x.shape


x = self.emb_dropout(x)
a = self.transformer(x)
# a = self.mlp_head(self.transformer(x)[:,0])
# a = self.mlp_head(self.transformer(x).mean(dim=1))

cc = x[:,0,:].clone()
x[:,0,:] = x[:,1,:]
x[:,1,:] = cc
b = self.transformer(x)
# b = self.mlp_head(self.transformer(x)[:,0])
# b = self.mlp_head(self.transformer(x).mean(dim=1))


x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

# x = self.to_latent(x)
return self.mlp_head(x)



