from sentence_transformers import SentenceTransformer, InputExample
from sentence_transformers import models, losses
from torch.utils.data import DataLoader
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator

# Define your sentence transformer model using CLS pooling
model_name = 'distilroberta-base'
word_embedding_model = models.Transformer(model_name, max_seq_length=32)
pooling_model = models.Pooling(
    word_embedding_model.get_word_embedding_dimension())
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

# Define a list with sentences (1k - 100k sentences)
train_sentences = [
    "Your set of sentences", "Model will automatically add the noise",
    "And re-construct it", "You should provide at least 1k sentences",
    "Your set of sentences", "Model will automatically add the noise",
    "And re-construct it", "You should provide at least 1k sentences",
    "Your set of sentences", "Model will automatically add the noise",
    "And re-construct it", "You should provide at least 1k sentences",
    "Your set of sentences", "Model will automatically add the noise",
    "And re-construct it", "You should provide at least 1k sentences"
]

# Convert train sentences to sentence pairs
train_data = [InputExample(texts=[s, s]) for s in train_sentences]
val_data = [InputExample(texts=[s, s], label=1.0) for s in train_sentences]

dev_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(val_data,
                                                                 batch_size=8,
                                                                 name='dev')

# DataLoader to batch your data
train_dataloader = DataLoader(train_data, batch_size=1, shuffle=True)

# Use the denoising auto-encoder loss
train_loss = losses.MultipleNegativesRankingLoss(model)

# Call the fit method
model.fit(train_objectives=[(train_dataloader, train_loss)],
          epochs=1,
          evaluator=dev_evaluator,
          evaluation_steps=2,
          show_progress_bar=True)

model.save('output/simcse-model')