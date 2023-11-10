import pandas as pd
import nltk
import re
from tqdm import tqdm
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

test = pd.read_csv('data/reddit/test.csv')
train = pd.read_csv('data/reddit/train.csv')
val = pd.read_csv('data/reddit/val.csv')

# concat
df = pd.concat([test, train, val])

texts = df['text'].values

# trainword2vec


ps = PorterStemmer()

def preprocess(texts):
    processed_texts = []
    for text in tqdm(texts):
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        words = word_tokenize(text)
        stemmed_words = [ps.stem(word) for word in words if word not in stop_words]
        processed_texts.append(stemmed_words)
    return processed_texts

processed_texts = preprocess(texts)

model = Word2Vec(sentences=processed_texts, vector_size=100, window=5, min_count=1, workers=4)

# save
model.save('reddit_model.model')


def cosine_similarity_of_words(word1, word2):
    word1 = ps.stem(word1)
    word2 = ps.stem(word2)
    return model.wv.similarity(word1, word2)

# Calculate cosine similarity of two words
print(cosine_similarity_of_words('love', 'apple'))

# Get first 10 words similar to 'relations'
print(model.wv.most_similar(ps.stem('truth'), topn=10))


words = ['doing', 'walking', 'running', 'playing', 'apple', 'lemon', 'orange', 'pineapple', 'happy', 'sad', 'angry', 'disgusted']

vectors = [model.wv[ps.stem(word)] for word in words]



pca = PCA(n_components=6)
result = pca.fit_transform(vectors)

# create a scatter plot of the projection
plt.scatter(result[:, 0], result[:, 1])
for i, word in enumerate(words):
    plt.annotate(word, xy=(result[i, 0], result[i, 1]))


# show variance components
print(pca.explained_variance_ratio_)

# show 3d
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(result[:, 0], result[:, 1], result[:, 2])
for i, word in enumerate(words):
    ax.text(result[i, 0], result[i, 1], result[i, 2], word)
