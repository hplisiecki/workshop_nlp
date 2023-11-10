import pandas as pd
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from gensim.models import Word2Vec
import re

# Make sure to download the stopwords set
import nltk
nltk.download('stopwords')

# Sample text for demonstration
poetry_df = pd.read_csv('data/poetry.csv')
poems = poetry_df['content'].values

# Initialize the Porter Stemmer
ps = PorterStemmer()

# Set of English stopwords
stop_words = set(stopwords.words('english'))

# Preprocessing the text
def preprocess_poems(poems):
    processed_poems = []

    for poem in poems:
        # Convert to lowercase
        poem = poem.lower()
        # Remove punctuation
        poem = re.sub(r'[^\w\s]', '', poem)
        # Tokenize words
        words = word_tokenize(poem)
        # Remove stopwords and stem words
        stemmed_words = [ps.stem(word) for word in words if word not in stop_words]
        processed_poems.append(stemmed_words)

    return processed_poems

# Preprocess the poems
processed_poems = preprocess_poems(poems)

# Train a Word2Vec model with the stemmed text
poem_model = Word2Vec(sentences=processed_poems, vector_size=100, window=5, min_count=1, workers=4)

# save
poem_model.save('data/poem_model')

# Summarize the stemmed model
poem_model_summary = f"Word2Vec Model Trained on Poems Summary:\nVocabulary Size: {len(poem_model.wv.index_to_key)} words\nVector Size: {poem_model.vector_size}\nTraining Window Size: {poem_model.window}"
print(poem_model_summary)

def cosine_similarity_of_words(word1, word2):
    word1 = ps.stem(word1)
    word2 = ps.stem(word2)
    return poem_model.wv.similarity(word1, word2)

# Calculate cosine similarity of two words
print(cosine_similarity_of_words('and', 'relations'))

# Get first 10 words similar to 'relations'
print(poem_model.wv.most_similar('truth', topn=10))
