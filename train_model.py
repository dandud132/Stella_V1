import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
import words

# Обучение модели
vectorizer = CountVectorizer()
vectors = vectorizer.fit_transform(list(words.data_set.keys()))

clf = LogisticRegression()
clf.fit(vectors, list(words.data_set.values()))

# Сохранение модели и векторизатора в файл
with open('model.pkl', 'wb') as model_file:
    pickle.dump(clf, model_file)

with open('vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)
