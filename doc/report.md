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

The PCA plot is pretty significant as it shows, languages with similar alphabets overlapped. For example:
- In the upper right corner Thai, Hindi and Tamil.
- Below, Korean and Japanese.
- In the left hand side, european languages (Spanish, English, French, Portuguese...).
- In the lower right corner, arabic languages (Urdu, Arabic and Pushto).

![PCA Plot](images/baseline_pca.png)

## Preprocess

### Sentence splitting and tokenization

Splitting sentences by `[^\w\s]` regular expression isn't a good preprocessing method as sentences are split by words in most cases. Thus, the matrix shows low performance on languages with a common origin e.g. Hindi, Tamil and other Asiatic languages. Note that some of this results are due to references to words in other languages, e.g. street addresses or citations.

![Confusion Matrix](images/preprocess_split_sentence_confusion.png)


