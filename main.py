import nltk
nltk.download('stopwords')
nltk.download('punkt')
import pandas as pd
import numpy as np

with open('Suc ve Ceza.txt', encoding="utf8") as f:
   suçveceza = f.read()

suç_token = nltk.word_tokenize(suçveceza)
yaz1 = nltk.Text(suç_token)
print(len(nltk.word_tokenize(suçveceza)))
print(len(set(nltk.word_tokenize(suçveceza))))

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
lemmatized = [lemmatizer.lemmatize(w,'v') for w in yaz1]
print(len(set(lemmatized)))

oran=len(set(nltk.word_tokenize(suçveceza)))/len(nltk.word_tokenize(suçveceza))
print(oran)

nltk.download('wordnet')
import operator
print(sorted(yaz1.vocab().items(), key=operator.itemgetter(1), reverse=True)[:10] )

print(sorted([token for token, freq in yaz1.vocab().items() if len(token) > 3 and freq > 350]))

print(sorted([(token, len(token))for token, freq in yaz1.vocab().items()], key=operator.itemgetter(1), reverse=True)[0] )

print(np.mean([len(nltk.word_tokenize(sent)) for sent in nltk.sent_tokenize(suçveceza)]))