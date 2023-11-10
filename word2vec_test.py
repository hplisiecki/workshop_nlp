from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
import re

# Sample text for demonstration
text = """
In the town of Athy one Jeremy Lanigan
Battered away till he hadn't a pound.
His father died and made him a man again
Left him a farm and ten acres of ground.

He gave a grand party for friends and relations
Who didn't forget him when come to the wall,
And if you'll but listen I'll make your eyes glisten
Of the rows and the ructions of Lanigan's Ball.

Myself to be sure got free invitation,
For all the nice girls and boys I might ask,
And just in a minute both friends and relations
Were dancing as merry as bees round a cask.

Judy O'Leary that neat little milliner,
She tipped me a wink for to give her a call,
And soon I arrived with Peggy McGilligan
Just in time for Lanigan's Ball.
"""
# Preprocessing the text
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Split text into sentences
    sentences = text.split('\n')
    # Split each sentence into words
    word_lists = [sentence.split() for sentence in sentences if sentence]
    return word_lists

# Preprocess the sample text
processed_text = preprocess_text(text)


# Initialize the Porter Stemmer
ps = PorterStemmer()

# Function to stem the words in the text
def stem_text(word_lists):
    stemmed_text = []
    for sentence in word_lists:
        stemmed_sentence = [ps.stem(word) for word in sentence]
        stemmed_text.append(stemmed_sentence)
    return stemmed_text

# Stem the preprocessed text
stemmed_text = stem_text(processed_text)

# Train a Word2Vec model with the stemmed text
stemmed_model = Word2Vec(sentences=stemmed_text, vector_size=100, window=5, min_count=1, workers=4)

# save
stemmed_model.save('data/test_model')

# Summarize the stemmed model
stemmed_summary = f"Stemmed Word2Vec Model Summary:\nVocabulary Size: {len(stemmed_model.wv.index_to_key)} words\nVector Size: {stemmed_model.vector_size}\nTraining Window Size: {stemmed_model.window}"

def cosine_similarity_of_words(word1, word2):
    word1 = ps.stem(word1)
    word2 = ps.stem(word2)
    return stemmed_model.wv.similarity(word1, word2)

# calculate cosine similarity of two words
print(cosine_similarity_of_words('and', 'relations'))
