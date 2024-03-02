# Language Detection

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


