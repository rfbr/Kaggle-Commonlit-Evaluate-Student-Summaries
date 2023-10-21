import os
import random
import re
from typing import List

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from autocorrect import Speller
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from src.config import CFG

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def text_cleaning(text):
    '''
    Cleans text into a basic form for NLP. Operations include the following:-
    1. Remove special charecters like &, #, etc
    2. Removes extra spaces
    3. Removes embedded URL links
    4. Removes HTML tags
    5. Removes emojis

    text - Text piece to be cleaned.
    '''
    template = re.compile(r'https?://\S+|www\.\S+')  # Removes website links
    text = template.sub(r'', text)

    soup = BeautifulSoup(text, 'lxml')  # Removes HTML tags
    only_text = soup.get_text()
    text = only_text

    emoji_pattern = re.compile(
        "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE,
    )
    text = emoji_pattern.sub(r'', text)

    # text = re.sub(r"[^a-zA-Z\d]", " ", text)  # Remove special Charecters
    text = re.sub('\n+', '\n', text)
    text = re.sub('\.+', '.', text)
    text = re.sub(' +', ' ', text)  # Remove Extra Spaces

    return text


def create_folds(data):
    cv_strategy = GroupKFold(4)
    for fold, (_, valid_idx) in enumerate(cv_strategy.split(data, groups=data.prompt_id)):
        data.loc[valid_idx, "fold"] = fold
    # data['fold'] = 0
    return data


def remove_randomly_n_sentences(text):
    ratio = 0.0
    # sentences = text.split('+|')
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
    random.shuffle(sentences)
    n = int(len(sentences) * ratio)
    to_drop = np.random.choice(list(range(len(sentences))), n, replace=False)
    return ' '.join([s.strip() for i, s in enumerate(sentences) if i not in to_drop])


class CommonLitDataset(Dataset):
    def __init__(self, df, tokenizer, max_length, student_text_max_length=512, aug=False):
        self._len = len(df)
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.student_text_max_length = student_text_max_length
        self.aug = aug

    def __len__(self):
        return self._len

    def _select_random_consecutive_sentences(self, text, n):
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
        num_sentences = len(sentences)
        if num_sentences <= n:
            return text
        start_idx = np.random.randint(0, num_sentences - n)
        return ". ".join(sentences[start_idx : start_idx + n])

    def _select_consecutive_sentences_tta(self, text, n):
        sentences = [s.strip() for s in text.split(".")]
        num_sentences = len(sentences)
        return [". ".join(sentences[start_idx : start_idx + n]) for start_idx in range(0, num_sentences, n)]

    def __getitem__(self, idx):
        row = self.df.loc[idx]
        student_text = row['fixed_text']
        prompt_question = row['prompt_question_token']
        # prompt_title = row['prompt_title_token']
        if self.aug:
            prompt_text = remove_randomly_n_sentences(row['prompt_text'])
            prompt_text = self.tokenizer(prompt_text, add_special_tokens=False)['input_ids']
        else:
            prompt_text = row['prompt_text_token']

        prompt_ids = [1] + prompt_question + prompt_text
        encoding = self.tokenizer(
            student_text,
            add_special_tokens=False,
        )
        text_input_ids = [2] + encoding.input_ids[: self.student_text_max_length] + [2]
        text_len = len(text_input_ids)

        prompt_ids = prompt_ids[: (self.max_length - text_len)]

        input_ids = prompt_ids + text_input_ids
        attention_mask = [1] * len(input_ids)
        input_dict = {"ids": input_ids, "masks": attention_mask, "sep": len(prompt_ids)}
        if "content" in row:
            input_dict["content"] = row['content']
            input_dict["wording"] = row['wording']
        else:
            input_dict["content"] = 0.0
            input_dict["wording"] = 0.0
        return input_dict


class Collate:
    def __call__(self, batch: List[dict]) -> dict:
        output = dict()

        # since our custom Dataset's __getitem__ method returns dictionary
        # the collate_fn function will receive list of dictionaries
        output['ids'] = [sample['ids'] for sample in batch]
        output['masks'] = [sample['masks'] for sample in batch]
        output['content'] = [sample['content'] for sample in batch]
        output['wording'] = [sample['wording'] for sample in batch]
        output['sep'] = [sample['sep'] for sample in batch]

        # calculate max token length of this batch
        batch_max = max([len(ids) for ids in output['ids']])

        # add padding
        output['ids'] = [sample + (batch_max - len(sample)) * [0] for sample in output['ids']]
        output['masks'] = [sample + (batch_max - len(sample)) * [0] for sample in output['masks']]

        # convert to tensors
        output['ids'] = torch.tensor(output['ids'], dtype=torch.long)
        output['masks'] = torch.tensor(output['masks'], dtype=torch.long)
        output['content'] = torch.tensor(output['content'], dtype=torch.float32)
        output['wording'] = torch.tensor(output['wording'], dtype=torch.float32)
        return output


from autocorrect.typos import Word


class MySpeller(Speller):
    def __init__(self):
        super().__init__()

    def autocorrect_word(self, word):
        """most likely correction for everything up to a double typo"""

        def get_candidates(word):
            w = Word(word, self.lang)
            candidates = self.existing([word]) or self.existing(w.typos()) or self.existing(w.double_typos()) or [word]
            return [(self.nlp_data.get(c, 0), c) for c in candidates]

        candidates = get_candidates(word)

        # in case the word is capitalized
        if word[0].isupper():
            candidates += get_candidates(word.lower())

        best_word = max(candidates)[1]

        if word[0].isupper():
            best_word = best_word.capitalize()
        return best_word


class CommonLitDataModule(pl.LightningDataModule):
    def __init__(self, fold, batch_size=8, model_name='deberta-base', max_length=512, num_workers=8, **kwargs):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_length = max_length
        df_summaries = pd.read_csv(os.path.join(CFG.data_path, "summaries_train.csv"))
        df_prompts = pd.read_csv(os.path.join(CFG.data_path, "prompts_train.csv"))
        df_prompts['prompt_text'] = df_prompts.prompt_text.str.replace('\r\n', '.').apply(text_cleaning)
        df_prompts['prompt_question_token'] = df_prompts.prompt_question.apply(
            lambda x: self.tokenizer(x, add_special_tokens=False)['input_ids']
        )
        df_prompts['prompt_text_token'] = df_prompts.prompt_text.apply(
            lambda x: self.tokenizer(x, add_special_tokens=False)['input_ids']
        )
        df_prompts['prompt_title_token'] = df_prompts.prompt_title.apply(
            lambda x: self.tokenizer(x, add_special_tokens=False)['input_ids']
        )
        df_summaries['fixed_text'] = df_summaries.text
        if 'fixed_text' not in df_summaries:
            speller = Speller(lang='en')
            df_prompts['prompt_tokens'] = df_prompts.prompt_text.apply(word_tokenize)
            df_prompts["prompt_tokens"].apply(lambda x: speller.nlp_data.update({tok: 1000 for tok in x}))
            df_summaries['fixed_text'] = df_summaries.text.apply(speller)
        df_main = df_summaries.merge(df_prompts, how='left')
        self.df = create_folds(df_main)
        self.fold = fold
        self.num_training_samples = len(self.df.query(f"fold != {self.fold}"))

    def setup(self, stage=None):
        df_train = self.df.query(f"fold != {self.fold}")
        df_valid = self.df.query(f"fold == {self.fold}")
        print(f"Validating on the prompt_id {df_valid.prompt_id.unique()}")
        if stage == "fit" or stage is None:
            self.train_dataset = CommonLitDataset(df_train, self.tokenizer, self.max_length, aug=False)
            self.valid_dataset = CommonLitDataset(df_valid, self.tokenizer, self.max_length)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            collate_fn=Collate(),
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            batch_size=int(self.batch_size * 2),
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=Collate(),
        )
