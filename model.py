import torch.nn as nn
from transformers import AutoModel


class EffectEncModel(nn.Module):

    def __init__(self, hidden_drop_rate):
        super(EffectEncModel, self).__init__()

        self.base_model = AutoModel.from_pretrained(
            'princeton-nlp/sup-simcse-bert-base-uncased')
        self.dropout = nn.Dropout(hidden_drop_rate)
        self.effect_head = nn.Linear(768, 768)
        self.object_head = nn.Linear(768, 768)

    def forward(self, input, return_hidden=False):
        hidden_state = self.base_model(**input).pooler_output
        hidden_state = self.dropout(hidden_state)
        effect_embed = self.effect_head(hidden_state)
        obj_embed = self.object_head(hidden_state)
        if return_hidden:
            return effect_embed, obj_embed, hidden_state
        else:
            return effect_embed, obj_embed
