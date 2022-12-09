import torch
import torch.nn.functional as F


class SimCSELoss:

    def __init__(self, temperature=0.05, device='cpu'):
        self.temperature = temperature
        self.device = device

    def __call__(self, embedding_1, embedding_2):
        # obj_embedding: [batch_size, 768]
        # action_embedding: [batch_size, 768]
        # same_action: [batch_size]
        # same_object_feature: [batch_size]
        batch_size = embedding_1.shape[0]
        target = torch.arange(batch_size, device=self.device)
        similarity_matrix = F.cosine_similarity(embedding_1.unsqueeze(1),
                                                embedding_2.unsqueeze(0),
                                                dim=-1)
        similarity_matrix /= self.temperature

        return F.cross_entropy(similarity_matrix, target)