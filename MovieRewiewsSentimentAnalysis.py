# %%
import pandas as pd
import numpy as np
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re

# %%
data = pd.read_csv('train.tsv', sep = '\t')
data.head(10)

# %%
import sklearn
print('The scikit-learn version is {}.'.format(sklearn.__version__))

# %%
data['Sentiment'].value_counts()

# %%
lemma = WordNetLemmatizer()

# %%
def clean_data(data_column):
    revised_data = []
    for i in range(0,len(data_column)):
        review = str(data_column[i])
        review = re.sub('[^a-zA-Z]',' ', review)
        review = [lemma.lemmatize(y) for y in word_tokenize(review.lower())]
        review = ' '.join(review)
        revised_data.append(review)
    return revised_data

# %%
data['Revised_Phrase'] = clean_data(data.Phrase.values)
data.head()

# %%
from sklearn.utils import resample

# %%
data_1 = data[data['Sentiment'] == 1]
data_2 = data[data['Sentiment'] == 2]
data_3 = data[data['Sentiment'] == 3]
data_4 = data[data['Sentiment'] == 4]
data_5 = data[data['Sentiment'] == 0]

data_1_sample = resample(data_1, replace = True ,n_samples = 75000,
                        random_state = 123)
data_2_sample = resample(data_2, replace = True ,n_samples = 75000,
                        random_state = 123)
data_3_sample = resample(data_3, replace = True ,n_samples = 75000,
                        random_state = 123)
data_4_sample = resample(data_4, replace = True ,n_samples = 75000,
                        random_state = 123)
data_5_sample = resample(data_5, replace = True ,n_samples = 75000,
                        random_state = 123)

data_resampled = pd.concat([data_1, data_1_sample, data_2,
                               data_2_sample, data_3,
                               data_3_sample, data_4,
                               data_4_sample, data_5,
                               data_5_sample])

# %%
data_resampled.head()

# %%
from nltk.util import ngrams
from nltk.tokenize import TweetTokenizer
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer 

# %%
text = ' '.join(data_resampled.loc[data_resampled.Sentiment == 4,
                                  'Phrase'].values)
text_trigrams = [i for i in ngrams(text.split(), 3)]
text_trigrams

# %%
Counter(text_trigrams).most_common(30)

# %%
tokenizer = TweetTokenizer()
tokenizer.tokenize

# %%
vectorizer = TfidfVectorizer(ngram_range = (1,2), tokenizer = tokenizer.tokenize)
full_text = list(data_resampled['Revised_Phrase'].values)
vectorizer.fit(full_text)
data_resampled_vectorized = vectorizer.transform(data_resampled['Revised_Phrase'])

y = data_resampled['Sentiment']
data_resampled_vectorized

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split, cross_val_score

# %%
LogisReg = LogisticRegression()
ovr = OneVsRestClassifier(LogisReg)

# %%
%%time
ovr.fit(data_resampled_vectorized, y)

# %%
scores = cross_val_score(ovr, data_resampled_vectorized, y, scoring = 'accuracy',n_jobs = -1, cv = 3)
print('Cross-validation mean accuracy {0:.2f}%, std {1: .2f}.'.format(np.mean(scores) * 100, np.std(scores) * 100))

# %%
from tensorflow.keras.utils import to_categorical
X = data_resampled['Revised_Phrase'].values
Y = to_categorical(data_resampled['Sentiment'].values)
X

# %%
from sklearn.model_selection import train_test_split
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.25, random_state=123)
X_train

# %%
print(X_train.shape,Y_train.shape)
print(X_val.shape,Y_val.shape)

# %%
from nltk import FreqDist

# %%
all_words=' '.join(X_train)
all_words=word_tokenize(all_words)
#print(all_words)
dist=FreqDist(all_words)

num_unique_word=len(dist)
num_unique_word
#X_train.head()

# %%
r_len=[]
for text in X_train:
    word=word_tokenize(text)
    l=len(word)
    r_len.append(l)
    
MAX_REVIEW_LEN=np.max(r_len)
MAX_REVIEW_LEN

# %%
max_features = num_unique_word
max_words = MAX_REVIEW_LEN
batch_size = 128
epochs = 3
num_classes=5

# %%
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(X_train)
X_train = tokenizer.texts_to_sequences(X_train)
X_val = tokenizer.texts_to_sequences(X_val)

# %%
from tensorflow.keras.preprocessing import sequence,text
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
X_train = sequence.pad_sequences(X_train, maxlen=max_words)
X_val = sequence.pad_sequences(X_val, maxlen=max_words)
#print(X_train.shape,X_val.shape)
X_train

# %%
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.optimizers import Adam

# %%
model1=Sequential()
model1.add(Embedding(max_features,100,mask_zero=True))

model1.add(LSTM(64,dropout=0.4, recurrent_dropout=0.4,return_sequences=True))
model1.add(LSTM(32,dropout=0.5, recurrent_dropout=0.5,return_sequences=False))
model1.add(Dense(num_classes,activation='softmax'))


model1.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.001),metrics=['accuracy'])
model1.summary()

# %%
model1.fit(X_train, Y_train, validation_data=(X_val, Y_val),epochs=epochs, batch_size=batch_size, verbose=1)

# %%
text1 = "This movie is fantastic! I really like it because it is so good!"
text2 = "Good movie!"
text3 = "Maybe I like this movie."
text4 = "Meh ..."
text5 = "If I were a drunk teenager then this movie might be good."
text6 = "Bad movie!"
text7 = "Not a good movie!"
text8 = "This movie really sucks! Can I get my money back please?"
texts = [text1, text2, text3, text4, text5, text6, text7, text8]

# %%
tokens = tokenizer.texts_to_sequences(texts)
tokens

# %%
tokens_pad = pad_sequences(tokens, maxlen=MAX_REVIEW_LEN)
tokens_pad.shape

# %%
prediction=model1.predict_classes(tokens_pad,verbose=1)

# %%
print(prediction)

# %%
test = pd.read_csv('test.tsv', sep="\t")
test.head()

# %%
test['clean_review']=clean_data(test.Phrase.values)
test.head()

# %%
test_vectorized = vectorizer.transform(test['clean_review'])
test1 = test['clean_review'].values

# %%
X_test = tokenizer.texts_to_sequences(test1)
X_test

# %%
X_test = sequence.pad_sequences(X_test, maxlen = max_words)
X_test

# %%
pred = model1.predict_classes(X_test, verbose = 1)

# %%
sub = pd.read_csv('sampleSubmission.csv', sep = ",")
sub.Sentiment = pred
sub.to_csv('Result.csv', index = False)