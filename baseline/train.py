import torch
import random
import os

from datasets import load_dataset
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForMultipleChoice, TrainingArguments, Trainer
from pathlib2 import Path
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from typing import Optional, Union
import numpy as np


def set_all_random_seed(seed, rank=0):
    """Set random seed.
    Args:
        seed (int): Nonnegative integer.
        rank (int): Process rank in the distributed training. Defaults to 0.
    """
    assert seed >= 0, f"Got invalid seed value {seed}."
    seed += rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_metrics(eval_predictions):
    predictions, label_ids = eval_predictions
    preds = np.argmax(predictions, axis=1)
    return {"accuracy": (preds == label_ids).astype(np.float32).mean().item()}


def preprocess_function(examples, tokenizer):
    ending_names = [
        'effect_sentence_1', 'effect_sentence_2', 'effect_sentence_3',
        'effect_sentence_4'
    ]
    # Repeat each first sentence four times to go with the four possibilities of second sentences.
    first_sentences = [[action + '<SEP>' + obj] * 4 for obj, action in zip(
        *[examples[name] for name in ('object', 'action')])]
    second_sentences = [
        list(effects)
        for effects in zip(*[examples[name] for name in ending_names])
    ]

    # Flatten everything
    first_sentences = sum(first_sentences, [])
    second_sentences = sum(second_sentences, [])

    # Tokenize
    tokenized_examples = tokenizer(first_sentences,
                                   second_sentences,
                                   truncation=False)
    # Un-flatten
    return {
        k: [v[i:i + 4] for i in range(0, len(v), 4)]
        for k, v in tokenized_examples.items()
    }


@dataclass
class DataCollatorForMultipleChoice:
    """
    Data collator that will dynamically pad the inputs for multiple choice received.
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])
        flattened_features = [[{k: v[i]
                                for k, v in feature.items()}
                               for i in range(num_choices)]
                              for feature in features]
        flattened_features = sum(flattened_features, [])

        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        # Un-flatten
        batch = {
            k: v.view(batch_size, num_choices, -1)
            for k, v in batch.items()
        }
        # Add back labels
        batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        return batch


if __name__ == '__main__':
    model_checkpoint = "roberta-base"
    batch_size = 12
    seed = 42
    epoch = 50
    set_all_random_seed(seed)

    # prepare the model
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
    model = AutoModelForMultipleChoice.from_pretrained(model_checkpoint)

    # load the data set
    data_path = Path('/home/junjayz/eecs595/ZSAER/data/multi_choice')
    cache_dir = data_path / 'cache'
    data_files = {
        'train': str(data_path / 'train.csv'),
        'val': str(data_path / 'val.csv'),
        'test': str(data_path / 'test.csv')
    }
    datasets = load_dataset('../', data_files=data_files, cache_dir=cache_dir)
    encoded_datasets = datasets.map(lambda x: preprocess_function(x, tokenizer),
                                    batched=True,
                                    load_from_cache_file=False)

    # Training arguments
    args = TrainingArguments(
        "baseline-multiple-choice",
        evaluation_strategy="epoch",
        learning_rate=5e-5,
        overwrite_output_dir='True',
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epoch,
        weight_decay=0.01,
    )
    trainer = Trainer(
        model,
        args,
        train_dataset=encoded_datasets["train"],
        eval_dataset=encoded_datasets["val"],
        tokenizer=tokenizer,
        data_collator=DataCollatorForMultipleChoice(tokenizer),
        compute_metrics=compute_metrics,
    )
    trainer.train()
    prediction = trainer.predict(encoded_datasets["test"])
    trainer.save_model("/home/junjayz/eecs595/ZSAER/baseline/weights")
    print(prediction.metrics)
