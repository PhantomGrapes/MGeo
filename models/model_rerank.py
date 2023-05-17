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
        if config.get('use_query_gis', False):
            self.gis_type = nn.Embedding(2, gis_width)

        bert_config = BertConfig.from_pretrained(text_encoder)
        bert_config.gis_embedding = 0
        self.text_encoder = BertModel.from_pretrained(text_encoder, config=bert_config, add_pooling_layer=True)      

        text_width = self.text_encoder.config.hidden_size
        self.gis2text = nn.Linear(gis_width, text_width)
        self.dropout = nn.Dropout(0.1)
        self.myclassifier = nn.Linear(text_width, 1)   
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(self, text, gis, query_gis, bs, docnum):
        gis_output = self.gis_encoder(input_ids = gis.input_ids,
                                       attention_mask = gis.attention_mask,
                                       token_type_ids = gis.token_type_ids,
                                       rel_type_ids = gis.rel_type_ids,
                                       absolute_position_ids = gis.absolute_position_ids,
                                       relative_position_ids = gis.relative_position_ids,
                                       return_dict = True,
                                      )                           
        if query_gis is not None:
            gis_embeds = gis_output.last_hidden_state + self.gis_type(torch.LongTensor([0]).to(gis.input_ids.device))
        else:
            gis_embeds = gis_output.last_hidden_state
        gis_atts = gis.attention_mask

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
            query_gis_embeds = query_gis_output.last_hidden_state + self.gis_type(torch.LongTensor([1]).to(gis.input_ids.device))
            query_gis_atts = query_gis.attention_mask

        embedding_output = self.text_encoder.embeddings(
            input_ids=text.input_ids
        )

        if query_gis is not None:
            merge_emb = torch.cat([embedding_output,  self.gis2text(query_gis_embeds), self.gis2text(gis_embeds)], dim=1)
            merge_attention = torch.cat([text.attention_mask, query_gis.attention_mask,  gis.attention_mask], dim=-1)
        else:
            merge_emb = torch.cat([embedding_output,  self.gis2text(gis_embeds)], dim=1)
            merge_attention = torch.cat([text.attention_mask, gis.attention_mask], dim=-1)

        text_output = self.text_encoder(attention_mask = merge_attention, encoder_embeds = merge_emb,
                                        return_dict = True, mode = 'text')            

        pooled_output = text_output[1]
        doc_feat = self.myclassifier(pooled_output).reshape(bs, docnum)

        labels = torch.zeros(bs, dtype=torch.long, device=doc_feat.device)
        loss = self.cross_entropy_loss(doc_feat, labels)
        return loss


