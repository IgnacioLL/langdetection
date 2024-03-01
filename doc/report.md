# Language Detection

## Preprocess

### Sentence splitting and tokenization

Splitting sentences by `[^\w\s]` regualar expression isn't a good preprocessing as sentences are split by words in most cases Thus, the matrix shows low performance on languages with a common origin e.g. Hindi, Tamil and other Asiatic languages. Note that some of this results are due to references to words in other languages, e.g. street addresses or citations.

![Confusion Matrix](images/preprocess_split_sentence_confusion.png)
