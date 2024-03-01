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
    if method == 'sentence-splitting':
        sentence = sentence.apply(lambda x: remove_numbers(x)) ## without it no difference 0.8918 vs 0.8920
        sentence = sentence.apply(lambda x: delete_minority_alphabet(x)) ## clearly degrades the performance of the algo.
        return sentence,labels
    elif method == 'alphabet-discrimination':
        return _split_sentences(sentence, labels)
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

def delete_minority_alphabet(sentence):
    # Selecting five random letters
    max_len = np.minimum(len(sentence), 10)
    random_letters = random.sample(sentence,  max_len)

    # Counting the occurrences of each alphabet type
    alphabet_counts = {
        'greek':  0,
        'cyrillic':  0,
        'latin':  0,
        'arabic':  0,
        'hebrew':  0,
        'cjk':  0,
        'hangul':  0,
        'hiragana':  0,
        'katakana':  0,
        'thai':  0
    }

    for letter in random_letters:
        if ad.is_greek(letter):
            alphabet_counts['greek'] +=  1
        if ad.is_cyrillic(letter):
            alphabet_counts['cyrillic'] +=  1
        if ad.is_latin(letter):
            alphabet_counts['latin'] +=  1
        if ad.is_arabic(letter):
            alphabet_counts['arabic'] +=  1
        if ad.is_hebrew(letter):
            alphabet_counts['hebrew'] +=  1
        if ad.is_cjk(letter):
            alphabet_counts['cjk'] +=  1
        if ad.is_hangul(letter):
            alphabet_counts['hangul'] +=  1
        if ad.is_hiragana(letter):
            alphabet_counts['hiragana'] +=  1
        if ad.is_katakana(letter):
            alphabet_counts['katakana'] +=  1
        if ad.is_thai(letter):
            alphabet_counts['thai'] +=  1

    # Determining the majority alphabet type
    max_count = max(alphabet_counts.values())
    majority_alphabet = [key for key, value in alphabet_counts.items() if value == max_count]

    # Filtering letters based on the majority alphabet type
    filtered_letters = [letter for letter in sentence if getattr(ad, f'is_{majority_alphabet[0]}')(letter)]

    return "".join(filtered_letters)

def remove_numbers(text):
  """
  This function removes numbers from a given text string.

  Args:
      text: The text string to process.

  Returns:
      A new string with all numeric characters removed.
  """
  no_digits = "".join(char for char in text if not char.isdigit())
  return no_digits
