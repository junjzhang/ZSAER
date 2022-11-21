from transformers import AutoTokenizer
from transformers import TrainingArguments
from model import EffectEncModel

if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained(
        'princeton-nlp/sup-simcse-bert-base-uncased')
    training_args = TrainingArguments(output_dir="test_trainer")
    model = EffectEncModel(0.1)
