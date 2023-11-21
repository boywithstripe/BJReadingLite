# -*- coding: utf-8 -*-
# read
import json
import os
import shutil
import random

def get_json(json_path):
    with open(json_path,'r') as f:
        data = json.load(f)
    item = dict()
    item['img'] = data['imagePath']
    item['points'] = []
    item['rectangles'] = []
    temp = dict()
    temp['center'] = []
    temp['num'] = []
    temp['index'] = []
    box = {}
    box['r_num'] = []
    box['r_fnum'] = []
    for i in data['shapes']:
        if(i['shape_type']=='point'):
            if i['label'] == 'center':
                temp['center'].extend(i['points'])
            elif i['label'] == 'num':
                temp['num'].extend(i['points'])
            elif i['label'] == 'index':
                temp['index'].extend(i['points'])
        if(i['shape_type']=='rectangle'):
            if i['label'].startswith('f_'):
                box['r_fnum'].append(i['points'])
            else:
                box['r_num'].append(i['points'])
    item['points'].append(temp)
    item['rectangles'].append(box)
    return item


filepath = './jsons'
filelist = os.listdir(filepath)
jsonlist = []
data=dict()
count=0

for filename in filelist:
    if filename.endswith('.json'):
        jsonlist.append(filename)

data_size = len(jsonlist)
data_test_size = int(0.2*data_size)
test_array = random.sample(range(data_size),k=data_test_size)

data_train = dict()
data_test = dict()
data_train['data'] = []
data_test['data'] = []

ind = 0

for jsonname in jsonlist:
    if ind in test_array:
        item = get_json(os.path.join(filepath,jsonname))
        data_test['data'].append(item)
    else:
        item = get_json(os.path.join(filepath, jsonname))
        data_train['data'].append(item)
    ind += 1

with open('train.json','w') as f:
    json.dump(data_train,f,indent=2)
with open('test.json','w') as f:
    json.dump(data_test,f,indent=2)

