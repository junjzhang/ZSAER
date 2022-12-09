import pandas as pd
import math

from sentence_transformers import SentenceTransformer, InputExample
from sentence_transformers import models, losses
from torch.utils.data import DataLoader
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from pathlib import Path

data_dir = Path('data')
model_name = 'sentence-transformers/all-distilroberta-v1'
out_dir = Path('tune_results') / (model_name + '1')
batch_size = 64
num_epochs = 50

# Define your sentence transformer model using CLS pooling
word_embedding_model = models.Transformer(model_name, max_seq_length=64)
pooling_model = models.Pooling(
    word_embedding_model.get_word_embedding_dimension())
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
model._modules["1"].pooling_mode_mean_tokens = False
model._modules["1"].pooling_mode_cls_token = True

# Convert train sentences to sentence pairs
train_sentence_pairs = pd.read_pickle(
    data_dir / 'train_ft_emb.pkl')[['effect_sentence_1',
                                    'effect_sentence_2']].values.tolist()
train_data = [
    InputExample(texts=[str(sentences[0]), str(sentences[1])])
    for sentences in train_sentence_pairs
]
val_sentence_pairs = pd.read_pickle(data_dir / 'val_ft_emb.pkl')[[
    'effect_sentence_1', 'effect_sentence_2', 'score'
]].values.tolist()
val_data = [
    InputExample(texts=[str(sentences[0]), str(sentences[1])],
                 label=float(sentences[2])) for sentences in val_sentence_pairs
]
test_sentence_pairs = pd.read_pickle(data_dir / 'test_ft_emb.pkl')[[
    'effect_sentence_1', 'effect_sentence_2', 'score'
]].values.tolist()
test_data = [
    InputExample(texts=[str(sentences[0]), str(sentences[1])],
                 label=float(sentences[2])) for sentences in test_sentence_pairs
]

dev_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(val_data,
                                                                 batch_size=8,
                                                                 name='dev')

# DataLoader to batch your data
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

# Use the denoising auto-encoder loss
train_loss = losses.MultipleNegativesRankingLoss(model)

warmup_steps = math.ceil(len(train_dataloader) * num_epochs *
                         0.1)  #10% of train data for warm-up
print("Warmup-steps: {}".format(warmup_steps))

# Train the model
model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=dev_evaluator,
          epochs=num_epochs,
          evaluation_steps=len(train_dataloader) // 2,
          warmup_steps=warmup_steps,
          output_path=str(out_dir),
          save_best_model=True)
