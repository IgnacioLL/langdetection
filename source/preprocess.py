import pandas as pd
import numpy as np
import random
from typing import Literal
from alphabet_detector import AlphabetDetector

ad = AlphabetDetector()

def preprocess(sentence, labels, method: Literal['sentence-splitting', 'alphabet-discrimination'] | None):
    '''
    Task: Given a sentence apply all the required preprocessing steps
    to compute train our classifier, such as sentence splitting, 
    tokenization or sentence splitting.

    Input: Sentence in string format
    Output: Preprocessed sentence either as a list or a string
    '''
    if method == 'alphabet-discrimination':
        sentence = sentence.apply(lambda x:_remove_numbers(x)) ## without it no difference 0.8918 vs 0.8920
        sentence = sentence.apply(lambda x:_delete_minority_alphabet(x)) ## clearly degrades the performance of the algo.
        return sentence,labels
    elif method == 'sentence-splitting':
        return _split_sentences(sentence, labels)
    elif method == 'character-splitter':
        sentence = sentence.apply(lambda x: _split_sentences_in_characters(x, 1))
        return sentence, labels
    else:
        return sentence, labels

def _split_sentences(sentence: pd.Series, labels: pd.Series) -> tuple[pd.Series, pd.Series]:
    df = (
        pd.DataFrame({
           "sentence": sentence,
            "language": labels
        })
        .assign(sentence=lambda df_: df_
            .sentence
            .astype(str)
            .str
            .split(r'[^\w\s]')
        )
        .explode(column="sentence")
    )
    return df.sentence, df.language


def _delete_minority_alphabet(sentence):
    alphabets = ['greek', 'cyrillic', 'latin', 'arabic', 'hebrew', 'cjk', 'hangul', 'hiragana', 'katakana', 'thai']
    random_letters = random.sample(sentence, min(len(sentence), 10))
    
    alphabet_counts = {alphabet: sum(getattr(ad, f'is_{alphabet}')(letter) for letter in random_letters) for alphabet in alphabets}
    
    majority_alphabet = max(alphabet_counts, key=alphabet_counts.get)
    
    filtered_letters = [letter for letter in sentence if getattr(ad, f'is_{majority_alphabet}')(letter) or letter == " "]
    
    return "".join(filtered_letters)


def _remove_numbers(text):
  no_digits = "".join(char for char in text if not char.isdigit())
  return no_digits

def _split_sentences_in_characters(text: str, characters_sep: int=2):
    if _count_number_blancks(text, threshold=.05): 
        result = [text[i:i+2] for i in range(0, len(text), characters_sep)]
        result_splitted = " ".join(result)
        return result_splitted
    else: 
        return text

def _count_number_blancks(text: str, threshold: float) -> bool:
    counter = text.count(' ')
    return (counter/len(text)) < threshold