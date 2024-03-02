# Language Detection

## Introduction

Language detection is a common and useful task in natural language processing (NLP), which involves identifying the language of a given text from a set of possible languages. Language detection can be applied to various domains and applications, such as machine translation, text analysis, document retrieval, and multilingual communication. However, language detection also poses several challenges and difficulties, such as dealing with unstructured data, handling multiple languages and scripts, choosing appropriate features and models, and evaluating the performance and errors of the system.

In this report, we will explore the task of language detection using a dataset of 21859 sentences written in 22 languages, including several language families and writing scripts. We will follow a tutorial that provides a basic system for language detection using a naive Bayes classifier and frequency count features. We will also perform two exercises to modify and improve the system, by experimenting with different parameters, preprocessing steps, and classifier models. Our goal is to become familiar with NLP data and the additional challenges we may find compared to tasks based on structured data. We will also compare and analyze the results of the different systems, explaining the differences in performance and the kind of errors observed.

## First baseline

**How well does the vocabolary cover the data?**

Only 25% of the tokens in the text are also in the vocabulary, and the remaining 75% are unknown or out-of-vocabulary tokens. This implies that the vocabulary is too small or too specific to capture the diversity and variability of the text, and the language detector may have difficulties in identifying the language correctly.

**Which languages produce more errors? What do they have in common (family, script, etc)?**

As shown in the confusion matrix, languages from a common origin (asiatic, latin...) often produce more errors as they share similar vocabulary. It's been also observed some of the errors are due to cross language references or citations.

![Confusion Matrix](images/baseline_confusion.png)

**How languages overlap on the PCA plot? What could that overlapping mean?**

Although the PCA first two dimensiones do not capture a have percentage variance as first dimension captures 7.8\% and the second one captures 3.6\%, the PCA plot is pretty shows clear trends. Languages with similar alphabets overlapped. For example:
- In the upper right corner Thai, Hindi and Tamil.
- Below, Korean and Japanese.
- In the left hand side, european languages (Spanish, English, French, Portuguese...).
- In the lower right corner, arabic languages (Urdu, Arabic and Pushto).

![PCA Plot](images/baseline_pca.png)

## Preprocess

### Sentence splitting and tokenization

This preprocessing method splits each word of the sentences into a different record. Of course, repeating the same language label for each. Cross language references in sentences will now be split in one record each **assigning a wrong language**. In result, the training set will now contain a considerable number of misstagged records.

The results of this function are two series of preprocessed data that can be used as features and labels for the classifier. The function splits the sentences into words, which can capture the lexical and morphological features of different languages. The function also removes the punctuation marks, which can reduce the noise and sparsity of the data.

### Alphabet discrimination

This preprocessing method detects the most used alphabet being used taking 10 random characters from the sentence and deleting all characters which do not coincide with the same alphabet except the blank space. 

### Number removal

We also remove all numbers as they are not language specific and are not very useful and may create noise in the model. 


## Code

### Preprocess - Sentence splitting

The main functions used are `.split(r'[^\w\s]')` which returns a list of words for each sentence, and `.explode(...)` which transforms each element of a list-like to a row, replicating index values.

```python
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
```

### Preprocess - Alphabet discrimination

```python
def _delete_minority_alphabet(sentence):
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
    filtered_letters = [letter for letter in sentence if getattr(ad, f'is_{majority_alphabet[0]}')(letter) | (letter == " ")]

    return "".join(filtered_letters)
```

### Preprocess - Number removal

```python
def _remove_numbers(text):
  no_digits = "".join(char for char in text if not char.isdigit())
  return no_digits
```

## Experiments and results

### Preprocess - Sentence splitting

Splitting sentences by `[^\w\s]` regular expression isn't a good preprocessing method as there's no universal regular expression for this purpose. Thus, the matrix shows low performance on the following languages. Hindi and Urdu, which use complex ligatures or conjuncts to combine two or more characters into a single glyph. Thai and Vietnamese, use diacritical marks or tone marks to modify the pronunciation or meaning of the characters. Chinese and Japanese, do not use spaces or punctuation marks to separate words or sentences. Arabic, have different writing systems and scripts, such as the Arabic script and the Arabic numerals.

![Confusion Matrix](images/preprocess_split_sentence_confusion.png)

### Preprocess - Alphabet discrimination

Detecting the language and deleting all the non-equal alphabet characters degrades performance considerably. Specially in the hindi, tamil, japanese and chinese. As we don't understand the language and the posible variations is hard to determine why is it failing. We won't use it. 

![Confusion Matrix](images/preprocess_alphabet_discrimination_confusion.png)

### Preprocess - Number removal

Removing all numbers does not have an impact in the models ability to classify it. 
![Confusion Matrix](images/preprocess_number_removal_confusion.png)


## Conclusions