from functools import partial
from models.xbert import BertConfig, BertModel

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

        gis_config = BertConfig.from_json_file(config['gis_bert_config'])
        self.gis_encoder = BertModel(gis_config, add_pooling_layer=False)
        for param in self.gis_encoder.parameters():
            param.requires_grad = False
        gis_width = gis_config.hidden_size

        bert_config = BertConfig.from_pretrained(text_encoder)
        bert_config.gis_embedding = 0
        self.text_encoder = BertModel.from_pretrained(text_encoder, config=bert_config, add_pooling_layer=False)      

        text_width = self.text_encoder.config.hidden_size
        self.gis2text = nn.Linear(gis_width, text_width)
        self.embed_dim = config['embed_dim']
        self.query_proj = nn.Linear(text_width, config['embed_dim'])
        self.doc_proj = nn.Linear(text_width, config['embed_dim'])   

        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(self, query, doc, gis, query_gis, bs, docnum):
        if query_gis is not None:
            query_gis_output = self.gis_encoder(input_ids = query_gis.input_ids,
                                           attention_mask = query_gis.attention_mask,
                                           token_type_ids = query_gis.token_type_ids,
                                           rel_type_ids = query_gis.rel_type_ids,
                                           absolute_position_ids = query_gis.absolute_position_ids,
                                           relative_position_ids = query_gis.relative_position_ids,
                                           return_dict = True,
                                           mode='text',
                                          )                           
            query_gis_embeds = query_gis_output.last_hidden_state
            query_gis_atts = query_gis.attention_mask

            query_embedding_output = self.text_encoder.embeddings(
                input_ids=query.input_ids
            )
            query_merge_emb = torch.cat([query_embedding_output, self.gis2text(query_gis_embeds)], dim=1)
            query_merge_attention = torch.cat([query.attention_mask, query_gis.attention_mask], dim=-1)

            query_output = self.text_encoder(attention_mask = query_merge_attention, encoder_embeds = query_merge_emb,
                                        return_dict = True, mode = 'text')            
        else:
            query_output = self.text_encoder(query.input_ids, attention_mask = query.attention_mask,                      
                                        return_dict = True, mode = 'query')            
        query_embeds = query_output.last_hidden_state
        query_feat = self.query_proj(query_embeds[:,0,:])

        gis_output = self.gis_encoder(input_ids = gis.input_ids,
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
            input_ids=doc.input_ids
        )
        merge_emb = torch.cat([embedding_output, self.gis2text(gis_embeds)], dim=1)
        merge_attention = torch.cat([doc.attention_mask, gis.attention_mask], dim=-1)

        text_output = self.text_encoder(attention_mask = merge_attention, encoder_embeds = merge_emb,
                                        return_dict = True, mode = 'text')            
        text_embeds = text_output.last_hidden_state

        doc_feat = self.doc_proj(text_embeds[:,0,:])
        
        scores = query_feat.view(bs, 1, self.embed_dim).matmul(doc_feat.view(bs, docnum, self.embed_dim).transpose(1, 2))


        labels = torch.zeros(bs, dtype=torch.long, device=doc_feat.device)
        loss = self.cross_entropy_loss(scores.view(bs, docnum), labels)
        return loss
