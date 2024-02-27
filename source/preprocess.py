import nltk

#Tokenizer function. You can add here different preprocesses.
def preprocess(sentence, labels):
    '''
    Task: Given a sentence apply all the required preprocessing steps
    to compute train our classifier, such as sentence splitting, 
    tokenization or sentence splitting.

    Input: Sentence in string format
    Output: Preprocessed sentence either as a list or a string
    '''
    # Place your code here
    # Keep in mind that sentence splitting affectes the number of sentences
    # and therefore, you should replicate labels to match.


    return sentence,labels





import random
from alphabet_detector import AlphabetDetector

ad = AlphabetDetector()

def get_random_letters(sentence):
    # Selecting five random letters
    random_letters = random.sample(sentence,  5)

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
    filtered_letters = [letter for letter in random_letters if getattr(ad, f'is_{majority_alphabet[0]}')(letter)]

    return filtered_letters

