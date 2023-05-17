import argparse
import os
try:
    import ruamel_yaml as yaml
except ModuleNotFoundError:
    import ruamel.yaml as yaml
import numpy as np
from sklearn.metrics import ndcg_score
import random
import time
import datetime
import json
from pathlib import Path
import logging
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader

from models.model_rerank import MGeo 
from models.tokenization_bert import BertTokenizer

import utils
from dataset import create_dataset, create_sampler, create_loader
from scheduler import create_scheduler
from dataset.mgeo_dataset import GisUtt
from optim import create_optimizer

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

def pytorch_cos_sim(a: torch.Tensor, b: torch.Tensor):
    """
    Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
    This function can be used as a faster replacement for 1-scipy.spatial.distance.cdist(a,b)
    :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    a_norm = a / a.norm(dim=1)[:, None]
    b_norm = b / b.norm(dim=1)[:, None]
    return torch.mm(a_norm, b_norm.transpose(0, 1))


def train(model, data_loader, optimizer, tokenizer, epoch, warmup_steps, device, scheduler, config):
    # train
    model.train()  
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50
    step_size = 100
    warmup_iterations = warmup_steps*step_size  
    use_query_gis = config.get('use_query_gis', False)
    
    for i, datas in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        text = []
        gis_input_ids, gis_token_type_ids, gis_rel_type_ids, gis_absolute_position_ids, gis_relative_position_ids = ([], [], [], [], [])
        query_gis_input_ids, query_gis_token_type_ids, query_gis_rel_type_ids, query_gis_absolute_position_ids, query_gis_relative_position_ids = ([], [], [], [], [])
        for bs in range(len(datas['query'])):
            for did in range(len(datas['docs'])):
                text.append(datas['query'][bs] + '[SEP]' + datas['docs'][did][bs] + '[SEP]')
                gis_input_ids.append(datas['gis_input_ids'][did][bs])
                gis_token_type_ids.append(datas['gis_token_type_ids'][did][bs])
                gis_rel_type_ids.append(datas['gis_rel_type_ids'][did][bs])
                gis_absolute_position_ids.append(datas['gis_absolute_position_ids'][did][bs])
                gis_relative_position_ids.append(datas['gis_relative_position_ids'][did][bs])
                if use_query_gis:
                    query_gis_input_ids.append(datas['query_gis_input_ids'][0][bs])
                    query_gis_token_type_ids.append(datas['query_gis_token_type_ids'][0][bs])
                    query_gis_rel_type_ids.append(datas['query_gis_rel_type_ids'][0][bs])
                    query_gis_absolute_position_ids.append(datas['query_gis_absolute_position_ids'][0][bs])
                    query_gis_relative_position_ids.append(datas['query_gis_relative_position_ids'][0][bs])

        gis = GisUtt(0, 1, device)
        gis.update(gis_input_ids, gis_token_type_ids, gis_rel_type_ids, gis_absolute_position_ids, gis_relative_position_ids)

        if use_query_gis:
            query_gis = GisUtt(0, 1, device)
            query_gis.update(query_gis_input_ids, query_gis_token_type_ids, query_gis_rel_type_ids, query_gis_absolute_position_ids, query_gis_relative_position_ids)

        text_input = tokenizer(text, padding='longest', return_tensors="pt").to(device)
        if use_query_gis:
            loss = model(text_input, gis, query_gis, len(datas['query']), config['pnum'] + config['train_nnum'])
        else:
            loss = model(text_input, gis, None, len(datas['query']), config['pnum'] + config['train_nnum'])
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()    
        
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        if epoch==0 and i%step_size==0 and i<=warmup_iterations: 
            scheduler.step(i//step_size)         
        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}  

@torch.no_grad()
def evaluation(model, data_loader, tokenizer, device, config, output_dir=None, output_name=''):
    # test
    model.eval() 
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Evaluation:'    
    if output_dir is not None:
        out = open(f'{output_dir}/{output_name}_evaluation_detail.txt', 'w')

    print('Computing features for evaluation...')
    start_time = time.time()  
    top1acc = 0
    top3acc = 0
    top5acc = 0
    total = 0
    mrr1 = 0
    mrr3 = 0
    mrr5 = 0
    text, query, doc, gis_input_ids, gis_token_type_ids, gis_rel_type_ids, gis_absolute_position_ids, gis_relative_position_ids = ([], [], [], [], [], [], [], [])
    query_gis_input_ids, query_gis_token_type_ids, query_gis_rel_type_ids, query_gis_absolute_position_ids, query_gis_relative_position_ids = ([], [], [], [], [])
    group_ids = []
    gold_max = []
    true_relevance1 = []
    pred_relevance1 = []
    true_relevance3 = []
    pred_relevance3 = []
    true_relevance5 = []
    pred_relevance5 = []

    batch_size = 512
    use_query_gis = config.get('use_query_gis', False)

    for datas in tqdm(data_loader):
        for did in range(len(datas['docs'])):
            text.append(datas['query'] + '[SEP]' + datas['docs'][did] + '[SEP]')
            query.append(datas['query'])
            doc.append(datas['docs'][did])
            gis_input_ids.append(datas['gis_input_ids'][did])
            gis_token_type_ids.append(datas['gis_token_type_ids'][did])
            gis_rel_type_ids.append(datas['gis_rel_type_ids'][did])
            gis_absolute_position_ids.append(datas['gis_absolute_position_ids'][did])
            gis_relative_position_ids.append(datas['gis_relative_position_ids'][did])

            if use_query_gis:
                query_gis_input_ids.append(datas['query_gis_input_ids'][0])
                query_gis_token_type_ids.append(datas['query_gis_token_type_ids'][0])
                query_gis_rel_type_ids.append(datas['query_gis_rel_type_ids'][0])
                query_gis_absolute_position_ids.append(datas['query_gis_absolute_position_ids'][0])
                query_gis_relative_position_ids.append(datas['query_gis_relative_position_ids'][0])

        group_ids.append(len(text))
        gold_max.append(datas['gold_max'])
        total += 1

        if len(text) >= batch_size:
            gis = GisUtt(0, 1, device)
            gis.update(gis_input_ids, gis_token_type_ids, gis_rel_type_ids, gis_absolute_position_ids, gis_relative_position_ids)

            if use_query_gis:
                query_gis = GisUtt(0, 1, device)
                query_gis.update(query_gis_input_ids, query_gis_token_type_ids, query_gis_rel_type_ids, query_gis_absolute_position_ids, query_gis_relative_position_ids)

            gis_output = model.gis_encoder(input_ids = gis.input_ids,
                                           attention_mask = gis.attention_mask,
                                           token_type_ids = gis.token_type_ids,
                                           rel_type_ids = gis.rel_type_ids,
                                           absolute_position_ids = gis.absolute_position_ids,
                                           relative_position_ids = gis.relative_position_ids,
                                           return_dict = True,
                                           mode='text',

                                          )                           
            if use_query_gis:
                gis_embeds = gis_output.last_hidden_state + model.gis_type(torch.LongTensor([0]).to(gis.input_ids.device))
            else:
                gis_embeds = gis_output.last_hidden_state
            gis_atts = gis.attention_mask

            if use_query_gis:
                query_gis_output = model.gis_encoder(input_ids = query_gis.input_ids,
                                               attention_mask = query_gis.attention_mask,
                                               token_type_ids = query_gis.token_type_ids,
                                               rel_type_ids = query_gis.rel_type_ids,
                                               absolute_position_ids = query_gis.absolute_position_ids,
                                               relative_position_ids = query_gis.relative_position_ids,
                                               return_dict = True,
                                               mode='text',
                                              )                           
                query_gis_embeds = query_gis_output.last_hidden_state + model.gis_type(torch.LongTensor([1]).to(gis.input_ids.device))
                query_gis_atts = query_gis.attention_mask

            text_input = tokenizer(text, padding='longest', return_tensors="pt").to(device)
            embedding_output = model.text_encoder.embeddings(
                input_ids=text_input.input_ids
            )

            if use_query_gis:
                merge_emb = torch.cat([embedding_output, model.gis2text(query_gis_embeds), model.gis2text(gis_embeds)], dim=1)
                merge_attention = torch.cat([text_input.attention_mask, query_gis.attention_mask, gis.attention_mask], dim=-1)
            else:
                merge_emb = torch.cat([embedding_output, model.gis2text(gis_embeds)], dim=1)
                merge_attention = torch.cat([text_input.attention_mask, gis.attention_mask], dim=-1)


            text_output = model.text_encoder(attention_mask = merge_attention, encoder_embeds = merge_emb,
                                            return_dict = True, mode = 'text')            
            pooled_output = text_output[1]

            logits = model.myclassifier(pooled_output).view(-1)

            prev = 0
            for gid, gmax in zip(group_ids, gold_max):
                hit_top1 = False
                cur_logits = logits[prev: gid]
                _, topk_ids = cur_logits.topk(10)
                topk_ids = topk_ids.tolist()
                for true_relevance, num in zip([true_relevance1, true_relevance3, true_relevance5], [1, 3, 5]):
                    true_relevance.append([1 if topkid < gmax else 0 for topkid in topk_ids])
                for pred_relevance, num in zip([pred_relevance1, pred_relevance3, pred_relevance5], [1, 3, 5]):
                    pred_relevance.append([float(logits[prev + topkid]) for topkid in topk_ids])
                if topk_ids[0] < gmax:
                    top1acc += 1
                    mrr1 += 1
                    hit_top1 = True
                pos = 0
                for idx in topk_ids[:3]:
                    if idx < gmax:
                        top3acc += 1
                        mrr3 += 1 / (pos + 1)
                        break
                    pos += 1
                pos = 0
                for idx in topk_ids[:5]:
                    if idx < gmax:
                        top5acc += 1
                        mrr5 += 1 / (pos + 1)
                        break
                    pos += 1

                if output_dir is not None:
                    myquery = ''
                    doc = []
                    dpos = 0
                    for printid in topk_ids:
                        q, d = text[printid + prev].split('[SEP]')[:2]
                        myquery = q
                        doc.append('<<' + str(dpos) + '>>' + d + ':{:.2f}'.format(logits[printid + prev] * 100))
                        dpos += 1
                    my_gold = text[prev].split('[SEP]')[1] + ':{:.2f}'.format(logits[prev] * 100)
                    out.write(str(hit_top1) + '\t' + myquery + '\t**' + my_gold +  '**\t' + '||'.join(doc) + '\n')
                prev = gid
            text = []
            group_ids = []
            gold_max = []
            text, query, doc, gis_input_ids, gis_token_type_ids, gis_rel_type_ids, gis_absolute_position_ids, gis_relative_position_ids = ([], [], [], [], [], [], [], [])
            query_gis_input_ids, query_gis_token_type_ids, query_gis_rel_type_ids, query_gis_absolute_position_ids, query_gis_relative_position_ids = ([], [], [], [], [])
    if len(text) > 0:
        gis = GisUtt(0, 1, device)
        gis.update(gis_input_ids, gis_token_type_ids, gis_rel_type_ids, gis_absolute_position_ids, gis_relative_position_ids)
        if use_query_gis:
            query_gis = GisUtt(0, 1, device)
            query_gis.update(query_gis_input_ids, query_gis_token_type_ids, query_gis_rel_type_ids, query_gis_absolute_position_ids, query_gis_relative_position_ids)

        gis_output = model.gis_encoder(input_ids = gis.input_ids,
                                       attention_mask = gis.attention_mask,
                                       token_type_ids = gis.token_type_ids,
                                       rel_type_ids = gis.rel_type_ids,
                                       absolute_position_ids = gis.absolute_position_ids,
                                       relative_position_ids = gis.relative_position_ids,
                                       return_dict = True,
                                       mode='text',
                                      )                           
        if use_query_gis:
            gis_embeds = gis_output.last_hidden_state + model.gis_type(torch.LongTensor([0]).to(gis.input_ids.device))
        else:
            gis_embeds = gis_output.last_hidden_state
        gis_atts = gis.attention_mask

        if use_query_gis:
            query_gis_output = model.gis_encoder(input_ids = query_gis.input_ids,
                                           attention_mask = query_gis.attention_mask,
                                           token_type_ids = query_gis.token_type_ids,
                                           rel_type_ids = query_gis.rel_type_ids,
                                           absolute_position_ids = query_gis.absolute_position_ids,
                                           relative_position_ids = query_gis.relative_position_ids,
                                           return_dict = True,
                                           mode='text',
                                          )                           
            query_gis_embeds = query_gis_output.last_hidden_state + model.gis_type(torch.LongTensor([1]).to(gis.input_ids.device))
            query_gis_atts = query_gis.attention_mask

        text_input = tokenizer(text, padding='longest', return_tensors="pt").to(device)
        embedding_output = model.text_encoder.embeddings(
            input_ids=text_input.input_ids
        )

        if use_query_gis:
            merge_emb = torch.cat([embedding_output, model.gis2text(query_gis_embeds), model.gis2text(gis_embeds)], dim=1)
            merge_attention = torch.cat([text_input.attention_mask, query_gis.attention_mask, gis.attention_mask], dim=-1)
        else:
            merge_emb = torch.cat([embedding_output, model.gis2text(gis_embeds)], dim=1)
            merge_attention = torch.cat([text_input.attention_mask, gis.attention_mask], dim=-1)

        text_output = model.text_encoder(attention_mask = merge_attention, encoder_embeds = merge_emb,
                                        return_dict = True, mode = 'text')            

        pooled_output = text_output[1]

        logits = model.myclassifier(pooled_output).view(-1)


        prev = 0
        for gid, gmax in zip(group_ids, gold_max):
            hit_top1 = False
            cur_logits = logits[prev: gid]
            _, topk_ids = cur_logits.topk(10)
            topk_ids = topk_ids.tolist()
            for true_relevance, num in zip([true_relevance1, true_relevance3, true_relevance5], [1, 3, 5]):
                true_relevance.append([1 if topkid < gmax else 0 for topkid in topk_ids])
            for pred_relevance, num in zip([pred_relevance1, pred_relevance3, pred_relevance5], [1, 3, 5]):
                pred_relevance.append([float(logits[prev + topkid]) for topkid in topk_ids])

            if topk_ids[0] < gmax:
                top1acc += 1
                mrr1 += 1
                hit_top1 = True
            pos = 0
            for idx in topk_ids[:3]:
                if idx < gmax:
                    top3acc += 1
                    mrr3 += 1 / (pos + 1)
                    break
                pos += 1
            pos = 0
            for idx in topk_ids[:5]:
                if idx < gmax:
                    top5acc += 1
                    mrr5 += 1 / (pos + 1)
                    break
                pos += 1
            if output_dir is not None:
                myquery = ''
                doc = []
                dpos = 0
                for printid in topk_ids:
                    q, d = text[printid + prev].split('[SEP]')[:2]
                    myquery = q
                    doc.append('<<' + str(dpos) + '>>' + d + ':{:.2f}'.format(logits[printid + prev] * 100))
                    dpos += 1
                my_gold = text[prev].split('[SEP]')[1] + ':{:.2f}'.format(logits[prev] * 100)
                out.write(str(hit_top1) + '\t' + myquery + '\t**' + my_gold +  '**\t' + '||'.join(doc) + '\n')
            prev = gid

        text = []
        query = []
        doc = []
        group_ids = []
        gold_max = []
        gis_input_ids, gis_token_type_ids, gis_rel_type_ids, gis_absolute_position_ids, gis_relative_position_ids = ([], [], [], [], [])
        query_gis_input_ids, query_gis_token_type_ids, query_gis_rel_type_ids, query_gis_absolute_position_ids, query_gis_relative_position_ids = ([], [], [], [], [])

    true_relevance1 = np.asarray(true_relevance1)
    true_relevance3 = np.asarray(true_relevance3)
    true_relevance5 = np.asarray(true_relevance5)
    pred_relevance1 = np.asarray(pred_relevance1)
    pred_relevance3 = np.asarray(pred_relevance3)
    pred_relevance5 = np.asarray(pred_relevance5)
    others = {'recall@5': top5acc / total, 'ndcg@1': float(ndcg_score(true_relevance1, pred_relevance1, k=1)), 'ndcg@3': float(ndcg_score(true_relevance3, pred_relevance3, k=3)), \
    'ndcg@5': float(ndcg_score(true_relevance5, pred_relevance5, k=5)), 'mrr@1': mrr1 / total, 'mrr@3': mrr3 / total, 'mrr@5': mrr5 / total}
    return top1acc / total, top3acc / total, others

def main(args, config):
    utils.init_distributed_mode(args)    
    
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    #### Dataset #### 
    print("Creating rerank dataset")
    #import pudb; pudb.set_trace()
    train_dataset, val_dataset, test_dataset = create_dataset('rerank', config)  

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()            
        samplers = create_sampler([train_dataset], [True], num_tasks, global_rank)
    else:
        samplers = [None]
    
    train_loader = create_loader([train_dataset],samplers,
                                                          batch_size=[config['batch_size_train']],
                                                          num_workers=[4],
                                                          is_trains=[True],
                                                          collate_fns=[None,])[0]
    val_loader = val_dataset
    test_loader = test_dataset
       
    tokenizer = BertTokenizer.from_pretrained(args.text_encoder)

    #### Model #### 
    print("Creating model")
    model = MGeo(config=config, text_encoder=args.text_encoder, tokenizer=tokenizer)
    print(get_parameter_number(model))
    
    if args.checkpoint:    
        checkpoint = torch.load(args.checkpoint, map_location='cpu') 
        state_dict = checkpoint['model']
        
        for key in list(state_dict.keys()):
            if 'bert' in key:
                encoder_key = key.replace('bert.','')         
                state_dict[encoder_key] = state_dict[key] 
                del state_dict[key]                
            if 'doc_proj' in key:
                del state_dict[key]
        msg = model.load_state_dict(state_dict,strict=False)  
        
        print('load checkpoint from %s'%args.checkpoint)
        print(msg)  
        
    
    model = model.to(device)   
    
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module   
    
    arg_opt = utils.AttrDict(config['optimizer'])
    optimizer = create_optimizer(arg_opt, model)
    arg_sche = utils.AttrDict(config['schedular'])
    lr_scheduler, _ = create_scheduler(arg_sche, optimizer)  
    
    max_epoch = config['schedular']['epochs']
    warmup_steps = config['schedular']['warmup_epochs']
    best = 0
    best_epoch = 0

    print("Start training")
    start_time = time.time()    
    for epoch in range(0, max_epoch):
        if not args.evaluate:
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)
            train_stats = train(model, train_loader, optimizer, tokenizer, epoch, warmup_steps, device, lr_scheduler, config)  
            
        val_top1acc, val_top3acc, val_others = evaluation(model_without_ddp, val_loader, tokenizer, device, config, output_dir=args.output_dir, output_name=f'dev-{epoch}')
        test_top1acc, test_top3acc, test_others = evaluation(model_without_ddp, test_loader, tokenizer, device, config, output_dir=args.output_dir, output_name=f'test-{epoch}')
   
        if utils.is_main_process():  

            if args.evaluate:                
                log_stats = {'val_top1acc': val_top1acc,                  
                             'val_top3acc': val_top3acc,
                             'test_top1acc': test_top1acc,
                             'test_top3acc': test_top3acc,
                             'epoch': epoch,
                            }
                for key in test_others:
                    log_stats['test_' + key] = test_others[key]
                for key in val_others:
                    log_stats['val_' + key] = val_others[key]

                with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
                    f.write(json.dumps(log_stats) + "\n")     
            else:
                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                             'val_top1acc': val_top1acc,                  
                             'val_top3acc': val_top3acc,
                             'test_top1acc': test_top1acc,
                             'test_top3acc': test_top3acc,
                             'epoch': epoch,
                            }
                for key in test_others:
                    log_stats['test_' + key] = test_others[key]
                for key in val_others:
                    log_stats['val_' + key] = val_others[key]

                with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
                    f.write(json.dumps(log_stats) + "\n")   
                    
                if val_top1acc > best:
                    save_obj = {
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'config': config,
                        'epoch': epoch,
                    }
                    torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_best.pth'))  
                    best = val_top1acc 
                    best_epoch = epoch
                    
            print(log_stats)
        if args.evaluate: 
            break
           
        lr_scheduler.step(epoch+warmup_steps+1)  
        dist.barrier()     
        torch.cuda.empty_cache()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str)) 

    if utils.is_main_process():   
        with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
            f.write("best epoch: %d"%best_epoch)               

            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()     
    parser.add_argument('--config', default='./configs/Rerank.yaml')
    parser.add_argument('--output_dir', default='output/Rerank')        
    parser.add_argument('--checkpoint', default='')   
    parser.add_argument('--text_encoder', default='bert-base-chinese')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    
    
    main(args, config)
