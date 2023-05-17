import torch
from torch.utils.data import DataLoader

from dataset.mgeo_dataset import  pretrain_gis_dataset, rerank_train_dataset


def create_dataset(dataset, config):
    if dataset=='pretrain_gis' or dataset == 'pretrain_mm':
        dataset = pretrain_gis_dataset(config['train_file'])                  
        return dataset      
    elif dataset=='rerank' or dataset == 'retrieval':
        train_dataset = rerank_train_dataset(config['train_file'], use_query_gis=config.get('use_query_gis', False))                  
        val_dataset = rerank_train_dataset(config['val_file'], use_query_gis=config.get('use_query_gis', False))                  
        test_dataset = rerank_train_dataset(config['test_file'], use_query_gis=config.get('use_query_gis', False))                  
        return train_dataset, val_dataset, test_dataset   

def create_sampler(datasets, shuffles, num_tasks, global_rank):
    samplers = []
    for dataset,shuffle in zip(datasets,shuffles):
        sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank, shuffle=shuffle)
        samplers.append(sampler)
    return samplers     


def create_loader(datasets, samplers, batch_size, num_workers, is_trains, collate_fns, shuffle=None):
    loaders = []
    for dataset,sampler,bs,n_worker,is_train,collate_fn in zip(datasets,samplers,batch_size,num_workers,is_trains,collate_fns):
        if is_train:
            if shuffle is None:
                shuffle = (sampler is None)
            drop_last = True
        else:
            shuffle = False
            drop_last = False
        loader = DataLoader(
            dataset,
            batch_size=bs,
            num_workers=n_worker,
            pin_memory=True,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last,
        )              
        loaders.append(loader)
    return loaders    
