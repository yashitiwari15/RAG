import nltk
import spacy
from textblob import TextBlob
import string

# 1. Tokenize a simple sentence using NLTK
nltk.download('punkt')  # Download the necessary tokenizer resources

sentence_nltk = "Natural Language Processing is fascinating!"
tokens = nltk.word_tokenize(sentence_nltk)
print("Tokenization using NLTK:")
print(tokens)
print()

# 2. Load the English language model in spaCy and perform POS tagging
# Download the English model if you haven't already
try:
    spacy.cli.download("en_core_web_sm")
except:
    pass

# Load the model
nlp = spacy.load("en_core_web_sm")

sentence_spacy = "The quick brown fox jumps over the lazy dog."
doc = nlp(sentence_spacy)

print("POS Tagging using spaCy:")
for token in doc:
    print(f"{token.text}: {token.pos_}")
print()

# 3. Perform basic sentiment analysis with TextBlob
sentence_textblob = "I love learning new things!"
blob = TextBlob(sentence_textblob)
sentiment = blob.sentiment

print("Sentiment Analysis using TextBlob:")
print(f"Polarity: {sentiment.polarity}, Subjectivity: {sentiment.subjectivity}")
print()

# 4. Write a Python function to clean text by converting it to lowercase and removing punctuation
def clean_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    return text

sentence_clean = "Hello, World! NLP is fun."
cleaned_sentence = clean_text(sentence_clean)
print("Cleaned Text:")
print(cleaned_sentence)
