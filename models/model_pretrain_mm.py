from functools import partial
from models.xbert import BertConfig, BertModel, BertForMaskedLM, BertOnlyMLMHead
import random

import torch
from torch import nn
import torch.nn.functional as F

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

class MGeo(nn.Module):
    def __init__(self,                 
                 text_encoder = None,
                 tokenizer = None,
                 config = None,     
                 ):
        super().__init__()
        
        self.tokenizer = tokenizer 

        bert_config = BertConfig.from_pretrained(text_encoder)
        bert_config.gis_embedding = 0
        self.text_encoder = BertModel.from_pretrained(text_encoder, config=bert_config, add_pooling_layer=False)      
        self.text_cls = BertOnlyMLMHead(bert_config)
        self.text_vocab_size = bert_config.vocab_size

        gis_config = BertConfig.from_json_file(config['gis_bert_config'])
        self.gis_encoder = BertModel(gis_config, add_pooling_layer=False)
        for param in self.gis_encoder.parameters():
            param.requires_grad = False
        gis_width = gis_config.hidden_size
        gis_config.hidden_size = bert_config.hidden_size
        self.gis_cls = BertOnlyMLMHead(gis_config)
        self.gis_vocab_size = gis_config.vocab_size

        text_width = self.text_encoder.config.hidden_size
        self.gis2text = nn.Linear(gis_width, text_width)

        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(self, text, gis, gh_labels=None):
        if random.random() < 0.5:
            return self.text_forward(text, gis)
        else:
            return self.tg_forward(text, gis)

    def text_forward(self, text, gis):
        input_ids = text.input_ids.clone()
        labels = input_ids.clone()

        probability_matrix = torch.full(labels.shape, 0.15)                    
        input_ids, labels = self.mask(input_ids, self.text_encoder.config.vocab_size, gis.device, targets=labels,
                                      probability_matrix = probability_matrix) 
        gis_input_ids = gis.input_ids
        gis_labels = gis_input_ids

        embedding_output = self.text_encoder.embeddings(
            input_ids=input_ids
        )
        text_output = self.text_encoder(attention_mask = text.attention_mask, encoder_embeds = embedding_output,
                                        return_dict = True, mode = 'text')            

        txt_emb = text_output.last_hidden_state
        txt_score = self.text_cls(txt_emb)
        loss_fct = nn.CrossEntropyLoss()  # -100 index = padding token
        loss = loss_fct(txt_score.view(-1, self.text_vocab_size), labels.view(-1))
        return loss


    def tg_forward(self, text, gis):
        if random.random() < 0.5:
            gis_input_ids = gis.input_ids.clone()
            gis_labels = gis_input_ids.clone()
            gis_probability_matrix = torch.full(gis_labels.shape, 0.1)                    
            gis_input_ids, gis_labels = self.mask_gis(gis_input_ids, self.gis_encoder.config.vocab_size, gis.device, targets=gis_labels,
                                          probability_matrix = gis_probability_matrix) 
            input_ids = text.input_ids
            labels = input_ids
        else:
            input_ids = text.input_ids.clone()
            labels = input_ids.clone()

            probability_matrix = torch.full(labels.shape, 0.15)                    
            input_ids, labels = self.mask(input_ids, self.text_encoder.config.vocab_size, gis.device, targets=labels,
                                          probability_matrix = probability_matrix) 
            gis_input_ids = gis.input_ids
            gis_labels = gis_input_ids

        gis_output = self.gis_encoder(input_ids = gis_input_ids,
                                       attention_mask = gis.attention_mask,
                                       token_type_ids = gis.token_type_ids,
                                       rel_type_ids = gis.rel_type_ids,
                                       absolute_position_ids = gis.absolute_position_ids,
                                       relative_position_ids = gis.relative_position_ids,
                                       return_dict = True,
                                       mode='text',
                                      )                           
        gis_embeds = gis_output.last_hidden_state
        gis_atts = gis.attention_mask

        embedding_output = self.text_encoder.embeddings(
            input_ids=input_ids
        )
        merge_emb = torch.cat([embedding_output, self.gis2text(gis_embeds)], dim=1)
        merge_attention = torch.cat([text.attention_mask, gis.attention_mask], dim=-1)

        text_output = self.text_encoder(attention_mask = merge_attention, encoder_embeds = merge_emb,
                                        return_dict = True, mode = 'text')            

        text_embeds = text_output.last_hidden_state
        b = text_embeds.size(0)
        tl = embedding_output.size(1)
        il = gis_embeds.size(1)

        txt_emb = text_embeds[:, :tl, :]
        gis_emb = text_embeds[:, tl:tl+il, :]
        txt_score = self.text_cls(txt_emb)
        gis_score = self.gis_cls(gis_emb)
        loss_fct = nn.CrossEntropyLoss()  # -100 index = padding token
        loss = loss_fct(txt_score.view(-1, self.text_vocab_size), labels.view(-1)) + loss_fct(gis_score.view(-1, self.gis_vocab_size), gis_labels.view(-1))

        return loss

    def mask(self, input_ids, vocab_size, device, targets=None, masked_indices=None, probability_matrix=None):
        if masked_indices is None:                                       
            masked_indices = torch.bernoulli(probability_matrix).bool()
                                               
        masked_indices[input_ids == self.tokenizer.pad_token_id] = False
        masked_indices[input_ids == self.tokenizer.cls_token_id] = False
        
        if targets is not None:
            targets[~masked_indices] = -100 # We only compute loss on masked tokens            

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(input_ids.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.tokenizer.mask_token_id

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(input_ids.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(vocab_size, input_ids.shape, dtype=torch.long).to(device)
        input_ids[indices_random] = random_words[indices_random]                     
        # The rest of the time (10% of the time) we keep the masked input tokens unchanged   
        
        if targets is not None:
            return input_ids, targets
        else:
            return input_ids


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


