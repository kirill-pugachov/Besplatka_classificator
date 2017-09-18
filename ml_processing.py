# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 00:57:15 2017

@author: Kirill
"""


#from scipy.sparse import hstack
from nltk.corpus import stopwords
import pandas
from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier

#читаем файл csv и разделяем метки класса и тексты для обучения
stop_w = stopwords.words('russian')
data_train = pandas.read_csv('storage/data_base_semantica.csv',  header=None)
y_train = data_train[0]
X_train_text = data_train[1]


#чистим тексты для обучения - все приводим к написанию строчными буквами
#удаляем из текстов все кроме букв и цифр
X_train_text = X_train_text.map(lambda kkk: kkk.lower())
X_train_text = X_train_text.replace('[^а-яА-Яa-zA-Z0-9]', ' ', regex=True)
X_train_text = X_train_text.map(lambda x: ' '.join([item.strip() for item in x.split(' ') if item not in stop_w]))



#Применим TfidfVectorizer для преобразования текстов в векторы признаков.
#min_df=2 - минимальная частота встречаемости слова в текстах
#vectorize = HashingVectorizer()
#vectorize = TfidfVectorizer(min_df=2, analyzer = 'word', stop_words = stop_w, lowercase = True)
min_df_list = [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1]
scoring = list()
for df_m in min_df_list:
    df_ma = 0.01 + df_m
    while df_ma <= 1:

        vectorize = TfidfVectorizer(min_df=df_m, max_df=df_ma)
        X_train = vectorize.fit_transform(X_train_text)
        
        
        #Используем SGDClassifier кол-во ядер все доступные n_jobs=-1
        classifier = SGDClassifier(loss='log', warm_start=True, n_jobs=-1, max_iter=200, tol=0.001)
        classifier.fit(X_train, y_train)
        
        #читаем файл csv с имеющимися запросами
        data_test = pandas.read_csv('storage/data_base_query.csv',  header=None)
        X_test_text = data_test[0]
        y_test_text = data_test[1]
        
        del data_test
        
        X_test_text = X_test_text.map(lambda kkk: kkk.lower())
        X_test_text = X_test_text.replace('[^а-яА-Яa-zA-Z0-9]', ' ', regex=True)
        X_test_text = X_test_text.map(lambda x: ' '.join([item.strip() for item in x.split(' ') if item not in stop_w]))
        X_test = vectorize.transform(X_test_text)
        #res = classifier.predict_proba(X_test)
        res = classifier.predict(X_test)
        scoring.append([classifier.score(X_test, y_test_text), df_m, df_ma])
        print (classifier.score(X_test, y_test_text), df_m, df_ma)
        df_ma += 0.01
result_three = pandas.concat([X_test_text,y_test_text], axis=1)
res_df = pandas.DataFrame(res)
final_res = pandas.concat([result_three, res_df], axis=1)

#data_train[1].map(lambda kkk: kkk.lower())
#data_train[1].map.replace('[^а-яА-Яa-zA-Z0-9]', ' ', regex=True)
#data_train[1].map.apply(lambda x: [item for item in x if item not in stop_w])