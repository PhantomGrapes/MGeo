'''
 * Copyright (c) 2021, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
'''

from functools import partial
from models.xbert import BertConfig, BertForMaskedLM, BertForGisMaskedLM

import torch
import torch.nn.functional as F
from torch import nn

import numpy as np
import random


class MGeo(nn.Module):
    def __init__(self,                 
                 text_encoder = None,
                 tokenizer = None,
                 config = None,    
                 ):
        super().__init__()
        
        self.tokenizer = tokenizer 
        self.mlm_probability = config['mlm_probability']
        self.gis_mlm_probability = config['mlm_probability']
        embed_dim = config['embed_dim']
     
        gis_config = BertConfig.from_json_file(config['gis_bert_config'])
        self.gis_encoder = BertForGisMaskedLM(gis_config)
        gis_width = gis_config.hidden_size

        self.kl_loss = nn.KLDivLoss(reduction="batchmean")

    def forward(self, gis, text, lxlys):
        gis_input_ids = gis.input_ids.clone()
        gis_labels = gis_input_ids.clone()

        gis_token_type_ids = gis.token_type_ids.clone()
        gis_token_type_ids_label = gis_token_type_ids.clone()
        gis_rel_type_ids = gis.rel_type_ids.clone()
        gis_rel_type_ids_label = gis_rel_type_ids.clone()
        gis_absolute_position_ids = gis.absolute_position_ids.clone()
        gis_absolute_position_ids_label = gis_absolute_position_ids.clone()
        gis_relative_position_ids = gis.relative_position_ids.clone()
        gis_relative_position_ids_label = gis_relative_position_ids.clone()

        gis_probability_matrix = torch.full(gis_labels.shape, self.gis_mlm_probability)                    
        gis_input_ids, gis_labels = self.mask_gis(gis_input_ids, self.gis_encoder.config.vocab_size, gis.device, targets=gis_labels,
                                      probability_matrix = gis_probability_matrix) 
        gis_token_type_ids, gis_token_type_ids_label = self.mask_gis(gis_token_type_ids, self.gis_encoder.config.type_vocab_size, gis.device, targets=gis_token_type_ids_label,
                                      probability_matrix = gis_probability_matrix) 
        gis_rel_type_ids, gis_rel_type_ids_label = self.mask_gis(gis_rel_type_ids, self.gis_encoder.config.rel_type_vocab_size, gis.device, targets=gis_rel_type_ids_label,
                                      probability_matrix = gis_probability_matrix) 
        gis_absolute_position_ids[:,:,0], gis_absolute_position_ids_label[:,:,0] = self.mask_gis(gis_absolute_position_ids[:,:,0], self.gis_encoder.config.absolute_x_vocab_size, gis.device, targets=gis_absolute_position_ids_label[:,:,0],
                                      probability_matrix = gis_probability_matrix) 
        gis_absolute_position_ids[:,:,2], gis_absolute_position_ids_label[:,:,2] = self.mask_gis(gis_absolute_position_ids[:,:,2], self.gis_encoder.config.absolute_x_vocab_size, gis.device, targets=gis_absolute_position_ids_label[:,:,2],
                                      probability_matrix = gis_probability_matrix) 
        gis_absolute_position_ids[:,:,1], gis_absolute_position_ids_label[:,:,1] = self.mask_gis(gis_absolute_position_ids[:,:,1], self.gis_encoder.config.absolute_y_vocab_size, gis.device, targets=gis_absolute_position_ids_label[:,:,1],
                                      probability_matrix = gis_probability_matrix) 
        gis_absolute_position_ids[:,:,3], gis_absolute_position_ids_label[:,:,3] = self.mask_gis(gis_absolute_position_ids[:,:,3], self.gis_encoder.config.absolute_y_vocab_size, gis.device, targets=gis_absolute_position_ids_label[:,:,3],
                                      probability_matrix = gis_probability_matrix) 
        gis_relative_position_ids[:,:,0], gis_relative_position_ids_label[:,:,0] = self.mask_gis(gis_relative_position_ids[:,:,0], self.gis_encoder.config.relative_x_vocab_size, gis.device, targets=gis_relative_position_ids_label[:,:,0],
                                      probability_matrix = gis_probability_matrix) 
        gis_relative_position_ids[:,:,2], gis_relative_position_ids_label[:,:,2] = self.mask_gis(gis_relative_position_ids[:,:,2], self.gis_encoder.config.relative_x_vocab_size, gis.device, targets=gis_relative_position_ids_label[:,:,2],
                                      probability_matrix = gis_probability_matrix) 
        gis_relative_position_ids[:,:,1], gis_relative_position_ids_label[:,:,1] = self.mask_gis(gis_relative_position_ids[:,:,1], self.gis_encoder.config.relative_y_vocab_size, gis.device, targets=gis_relative_position_ids_label[:,:,1],
                                      probability_matrix = gis_probability_matrix) 
        gis_relative_position_ids[:,:,3], gis_relative_position_ids_label[:,:,3] = self.mask_gis(gis_relative_position_ids[:,:,3], self.gis_encoder.config.relative_y_vocab_size, gis.device, targets=gis_relative_position_ids_label[:,:,3],
                                      probability_matrix = gis_probability_matrix) 

        gis_return_dic = self.gis_encoder(input_ids = gis_input_ids,
                                       attention_mask = gis.attention_mask,
                                       token_type_ids = gis_token_type_ids,
                                       rel_type_ids = gis_rel_type_ids,
                                       absolute_position_ids = gis_absolute_position_ids,
                                       relative_position_ids = gis_relative_position_ids,
                                       return_dict = True,
                                       labels = gis_labels,   
                                       mode='text',
                                       token_type_ids_label=gis_token_type_ids_label, rel_type_ids_label=gis_rel_type_ids_label, absolute_position_ids_label=gis_absolute_position_ids_label, relative_position_ids_label=gis_relative_position_ids_label,
                                       )                           

        batch_size = len(text.input_ids)
        gis_mlm = gis_return_dic.loss
        gis_embeds = gis_return_dic.hidden_states[-1]
        gis_atts = gis.attention_mask

        gis_feat = F.normalize(gis_embeds[:,0,:], dim=1)

        n = len(gis_feat)
        raw_sim_matrix = torch.matmul(gis_feat, gis_feat.t()).masked_select(~torch.eye(n, dtype=bool).to(gis.device)).view(n, n - 1)
        sim_matrix = F.log_softmax(raw_sim_matrix, dim=1)
        raw_dist_matrix = torch.cdist(lxlys, lxlys, p=2).masked_select(~torch.eye(n, dtype=bool).to(gis.device)).view(n, n - 1)
        dist_mean = torch.mean(raw_dist_matrix, dim=1).view(n, 1).expand(n, n-1)
        dist_std = torch.std(raw_dist_matrix, dim=1).view(n, 1).expand(n, n-1)
        dist_matrix = (raw_dist_matrix - dist_mean) / dist_std
        dist_matrix = F.softmax(F.sigmoid(-dist_matrix), dim=1)
        ctl = self.kl_loss(sim_matrix, dist_matrix) * 1000
        return gis_mlm, ctl

    def mask_gis(self, input_ids, vocab_size, device, targets=None, masked_indices=None, probability_matrix=None):
        if masked_indices is None:                                       
            masked_indices = torch.bernoulli(probability_matrix).bool()
                                               
        masked_indices[input_ids == 0] = False
        masked_indices[input_ids == 1] = False
        
        if targets is not None:
            targets[~masked_indices] = -100 # We only compute loss on masked tokens            

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(input_ids.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = 2 

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(input_ids.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(vocab_size, input_ids.shape, dtype=torch.long).to(device)
        input_ids[indices_random] = random_words[indices_random]                     
        # The rest of the time (10% of the time) we keep the masked input tokens unchanged   
        
        if targets is not None:
            return input_ids, targets
        else:
            return input_ids
        

