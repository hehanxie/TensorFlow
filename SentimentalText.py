from keras.layers.core import Activation, Dense
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
import pandas as pd
import nltk
# nltk.download('stopwords')
import collections
import numpy as np

# length of len
maxLen = 0
wordFreqs = collections.Counter()

num_recs = 0

# save data set into data
data = pd.read_csv('/Users/Wihan/Desktop/SentimentalText/train_text.csv', encoding='latin-1')
cols = data['SentimentText']
sentiment = data['Sentiment']
print(len(cols))
# item.lower() for item in strings
for line in cols:
    # get all words in one line, and keep all words all lower
    words = nltk.word_tokenize(line.lower())
    # for word in words:
    #     if words in stopwords:
    # print(words)
    if len(words) > maxLen:
        maxLen = len(words)
    for word in words:
        wordFreqs[word] += 1
    num_recs += 1

print('max length of sentence ', maxLen)
print('the frequency of words ', len(wordFreqs))

# max length of sentence 206,204
MAX_SENTENCE_LENGTH = maxLen
# total 118049 words,118000
MAX_FEATURES = len(wordFreqs)

# look up table
vocab_size = min(MAX_FEATURES, len(wordFreqs)) + 2
# create a dictionary to look up words
word2index = {x[0]: i+2 for i, x in enumerate(wordFreqs.most_common(MAX_FEATURES))}
word2index['PAD'] = 0
word2index['UNK'] = 1
index2word = {v: k for k, v in word2index.items()}

X = np.empty(num_recs, dtype=list)
y = np.zeros(num_recs)

i = 0
for line in cols:
    # get all words in one line, and keep all words all lower
    words = nltk.word_tokenize(line.lower())
    seqs = []
    # print(words)
    for word in words:
        if word not in stopwords.words('english'):
            if word in words:
                seqs.append(word2index[word])
            else:
                seqs.append(word2index['UNK'])
    X[i] = seqs
    i += 1

# result of sentiment
j = 0
for s in sentiment:
    y[j] = s
    j += 1

X = sequence.pad_sequences(X, maxlen=MAX_SENTENCE_LENGTH)
# set test_size, train size
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size=0.2, random_state=0)

# build model

# you happy is ok
EMBEDDING_SIZE = 128
HIDDEN_LAYER_SIZE = 50
BATCH_SIZE = 35
NUM_EPOCHS = 10

model = Sequential()
model.add(Embedding(vocab_size, EMBEDDING_SIZE, input_length=MAX_SENTENCE_LENGTH))
model.add(LSTM(HIDDEN_LAYER_SIZE, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1))
model.add(Activation("sigmoid"))
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# train model
model.fit(Xtrain, Ytrain, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS,validation_data=(Xtest, Ytest))

# predict
score, acc = model.evaluate(Xtest, Ytest, batch_size=BATCH_SIZE)
print("\nTest score: %.3f, accuracy: %.3f" % (score, acc))
print('\n{}  {}    {}'.format('guess', 'label', 'sentence'))
for i in range(10):
    idx = np.random.randint(len(Xtest))
    xtest = Xtest[idx].reshape(1, MAX_SENTENCE_LENGTH)
    ylabel = Ytest[idx]
    ypred = model.predict(xtest)[0][0]
    sent = " ".join([index2word[x] for x in xtest[0] if x != 0])
    print(' {}      {}     {}'.format(int(round(ypred)), int(ylabel), sent))


# test
INPUT_SENTENCES = ['I love reading.', 'You are so boring.']
XX = np.empty(len(INPUT_SENTENCES), dtype=list)
i = 0
for sentence in INPUT_SENTENCES:
    words = nltk.word_tokenize(sentence.lower())
    seq = []
    for word in words:
        if word in word2index:
            seq.append(word2index[word])
        else:
            seq.append(word2index['UNK'])
    XX[i] = seq
    i += 1

XX = sequence.pad_sequences(XX, maxlen=MAX_SENTENCE_LENGTH)
labels = [int(round(x[0])) for x in model.predict(XX)]
label2word = {1: 'HAPPY', 0: 'BAD'}
for i in range(len(INPUT_SENTENCES)):
    print('{}   {}'.format(label2word[labels[i]], INPUT_SENTENCES[i]))