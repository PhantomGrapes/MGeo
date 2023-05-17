import json
import os
import random

from torch.utils.data import Dataset, IterableDataset
import torch.distributed as dist
import torch
from tqdm import tqdm

from dataset.utils import pre_caption

import dill

class GisUtt:
    def __init__(self, pad_token_id, cls_token_id, device):
        self.pad_token_id = pad_token_id
        self.cls_token_id = cls_token_id
        self.input_ids = None
        self.attention_mask = None
        self.token_type_ids = None
        self.rel_type_ids = None
        self.absolute_position_ids = None
        self.relative_position_ids = None
        self.device = device
        self.max_length = 32

    def update(self, gis_input_ids, gis_token_type_ids, gis_rel_type_ids, gis_absolute_position_ids, gis_relative_position_ids):
        gis_input_ids = [[self.cls_token_id] + json.loads(f) for f in gis_input_ids]
        gis_token_type_ids = [[self.pad_token_id] + json.loads(f) for f in gis_token_type_ids]
        gis_rel_type_ids = [[self.pad_token_id] + json.loads(f) for f in gis_rel_type_ids]
        gis_absolute_position_ids = [[[self.pad_token_id] * 4] + json.loads(f) for f in gis_absolute_position_ids]
        gis_relative_position_ids = [[[self.pad_token_id] * 4] + json.loads(f) for f in gis_relative_position_ids]

        gis_input_ids = [f[:self.max_length] for f in gis_input_ids]
        gis_token_type_ids = [f[:self.max_length] for f in gis_token_type_ids]
        gis_rel_type_ids = [f[:self.max_length] for f in gis_rel_type_ids]
        gis_absolute_position_ids = [f[:self.max_length] for f in gis_absolute_position_ids]
        gis_relative_position_ids = [f[:self.max_length] for f in gis_relative_position_ids]

        max_length = max([len(item) for item in gis_input_ids])
        self.input_ids = torch.tensor(
                    [f + [self.pad_token_id] * (max_length - len(f)) for f in gis_input_ids], dtype=torch.long
                            ).to(self.device)
        self.attention_mask = torch.tensor(
                    [[1] * len(f) + [0] * (max_length - len(f)) for f in gis_input_ids], dtype=torch.long
                            ).to(self.device)
        self.token_type_ids = torch.tensor(
                    [f + [self.pad_token_id] * (max_length - len(f)) for f in gis_token_type_ids], dtype=torch.long
                            ).to(self.device)
        self.rel_type_ids = torch.tensor(
                    [f + [self.pad_token_id] * (max_length - len(f)) for f in gis_rel_type_ids], dtype=torch.long
                            ).to(self.device)

        self.absolute_position_ids = torch.tensor(
                    [f + [[self.pad_token_id] * 4] * (max_length - len(f)) for f in gis_absolute_position_ids], dtype=torch.long
                            ).to(self.device)
        self.relative_position_ids = torch.tensor(
                    [f + [[self.pad_token_id] * 4] * (max_length - len(f)) for f in gis_relative_position_ids], dtype=torch.long
                            ).to(self.device)

class rerank_train_dataset(Dataset):
    def __init__(self, ann_file,  max_words=64, use_query_gis=False):        
        self.max_words = max_words 
        self.use_query_gis = use_query_gis
        self.ann = []
        
        for line in open(ann_file):
            data = json.loads(line)
            self.ann.append(data)
        self.ann = self.ann[:1000]
                                    
    def __len__(self):
        return len(self.ann)
    
    def __getitem__(self, index):    
        info = self.ann[index]
        if self.use_query_gis:
            datas = {'query': info['query'][:64], 'docs': [],'gold_max': len(info['positive_passages']), 'gis_input_ids': [], 'gis_token_type_ids': [], 'gis_rel_type_ids': [], 'gis_absolute_position_ids': [], 'gis_relative_position_ids': [], 'query_gis_input_ids': [], 'query_gis_token_type_ids': [], 'query_gis_rel_type_ids': [], 'query_gis_absolute_position_ids': [], 'query_gis_relative_position_ids': []}
        else:
            datas = {'query': info['query'][:64], 'docs': [], 'gold_max': len(info['positive_passages']), 'gis_input_ids': [], 'gis_token_type_ids': [], 'gis_rel_type_ids': [], 'gis_absolute_position_ids': [], 'gis_relative_position_ids': []}
        if self.use_query_gis:
            geom_id, geom_type, rel_type, absolute_position, relative_position, _, _ = info['query_gis']
            datas['query_gis_input_ids'].append(json.dumps(geom_id))
            datas['query_gis_token_type_ids'].append(json.dumps(geom_type))
            datas['query_gis_rel_type_ids'].append(json.dumps(rel_type))
            datas['query_gis_absolute_position_ids'].append(json.dumps(absolute_position))
            datas['query_gis_relative_position_ids'].append(json.dumps(relative_position))

        for item in info['positive_passages'] + info['negative_passages']:
            text = item['text']
            geom_id, geom_type, rel_type, absolute_position, relative_position, _, _ = item['gis']

            datas['gis_input_ids'].append(json.dumps(geom_id))
            datas['gis_token_type_ids'].append(json.dumps(geom_type))
            datas['gis_rel_type_ids'].append(json.dumps(rel_type))
            datas['gis_absolute_position_ids'].append(json.dumps(absolute_position))
            datas['gis_relative_position_ids'].append(json.dumps(relative_position))

            datas['docs'].append(text[:self.max_words])   
        return datas

class pretrain_gis_dataset(IterableDataset):
    def __init__(self, train_file, max_words=64):        
        self.train_file = train_file
        self.max_words = max_words
        self.length = 0
        for _ in open(self.train_file):
            self.length += 1
        
    def __len__(self):
        return self.length // torch.cuda.device_count() 

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id
        total_workers = worker_info.num_workers
        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            rank = dist.get_rank()
        else:
            world_size = 1
            rank = 0
        total_workers *= world_size
        global_worker_id = worker_id * world_size + rank
        index = -1
        for i, line in enumerate(open(f'{self.train_file}')):
            index += 1
            if i % total_workers == global_worker_id:
                 dt = json.loads(line)
                 geom_id, geom_type, rel_type, absolute_position, relative_position, text, inst_id, lxly = dt
                 text = text[:self.max_words]
                 
                 yield json.dumps(geom_id), json.dumps(geom_type), json.dumps(rel_type), json.dumps(absolute_position), json.dumps(relative_position), text, inst_id, lxly

