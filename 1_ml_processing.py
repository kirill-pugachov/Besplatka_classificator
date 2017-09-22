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


size = 20


def clean_text(data_to_learn):
    '''clean up income data'''
    data_to_learn[1] = data_to_learn[1].map(lambda x: x.lower())
    data_to_learn[1] = data_to_learn[1].replace('[^а-яА-Яa-zA-Z0-9]', ' ', regex=True)
    data_to_learn[1] = data_to_learn[1].map(lambda x: ' '.join([item.strip() for item in x.split(' ') if item not in stop_w]))
    return data_to_learn

     
def read_row(X):
    '''read the one row from income'''
    for ind in X.index:
        label = X.iloc[ind]
        print('label from DataRame', label, '\n')
        yield label
        
        
#def get_minibatch(read_row, size):
#    '''read the data batches'''
#    y = []
#    try:
#        for _ in range(size):
#            label = next(read_row)
#            y.append(label)
#    except StopIteration:
#        return None
#    return y


def get_minibatch_1(read_row, size):
    '''read the data batches'''
    y = pandas.DataFrame()
    try:
        for _ in range(size):
            label = next(read_row)
#            print(label, '\n')
#            print('frame', label.to_frame(),'\n')
#            y = pandas.DataFrame.merge(y, label.to_frame(), left_index=True, right_index=True)
            y = y.append(label)
    except StopIteration:
        return None
    return y


if __name__ == '__main__':
    
#читаем файл csv и разделяем метки класса и тексты для обучения для обучения
#чистим тексты для обучения - все приводим к написанию строчными буквами
#удаляем из текстов все кроме букв и цифр
    stop_w = stopwords.words('russian') + ['tag', 'no', 'title', '9989']
    data_train = pandas.read_csv('storage_1/data_base_semantica.csv',  header=None)
    classes_train = data_train[0]
    bag_of_words = clean_text(data_train)[1].tolist()
    vectorize = TfidfVectorizer(min_df=1, max_df=5, use_idf=True)
#    vectorize = HashingVectorizer
    vectorize.fit(bag_of_words)
    gen_text = read_row(data_train)
    data_to_learn = clean_text(get_minibatch_1(gen_text, size))
    classifier = SGDClassifier(loss='log', warm_start=True, n_jobs=-1, max_iter=5)
    k = 1
#    data_to_learn = clean_text(data_to_learn)
    classifier.partial_fit(vectorize.transform(data_to_learn[1]), data_to_learn[0], classes = classes_train)
    data_to_learn = clean_text(get_minibatch_1(gen_text, size))
    while data_to_learn:
#        data_to_learn = clean_text(data_to_learn)
        classifier.partial_fit(vectorize.transform(data_to_learn[1]), data_to_learn[0])
        data_to_learn = get_minibatch_1(gen_text, size)
        k +=1
        print('Обучено строк', k, data_to_learn[0][0])

#читаем файл csv с имеющимися запросами для тестирования
#чистим тексты для обучения - все приводим к написанию строчными буквами
#удаляем из текстов все кроме букв и цифр
#bag_of_words =' '.join(X_train_text.tolist())
    data_test = pandas.read_csv('storage_1/data_base_query.csv',  header=None)
    X_test_text = clean_text(data_test[0])
    y_test_text = data_test[1]
    X_test = vectorize.transform(X_test_text)
    res = classifier.predict(X_test)
    print (classifier.score(X_test, y_test_text))


#classifier.partial_fit(vectorize.transform(pandas.DataFrame([clean_text(data_to_learn[0])[1]])), data_to_learn[0][0], classes = classes_train)    
#    
##    bag_of_words = X_train_text.tolist()
#
##Применим TfidfVectorizer для преобразования текстов в векторы признаков.
##min_df=2 - минимальная частота встречаемости слова в текстах
##vectorize = HashingVectorizer()
##vectorize = TfidfVectorizer(min_df=2, analyzer = 'word', stop_words = stop_w, lowercase = True)
#min_df_list = [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1]
#scoring = list()
#for df_m in min_df_list:
#    df_ma = 0.01 + df_m
#    while df_ma <= 1:
#        vectorize = TfidfVectorizer(min_df=df_m, max_df=df_ma)
#        vectorize.fit(bag_of_words)
#        X_train = vectorize.transform(X_train_text)
#        
#        #Используем SGDClassifier кол-во ядер все доступные n_jobs=-1
#        classifier = SGDClassifier(loss='log', warm_start=True, n_jobs=-1, max_iter=200, tol=0.001)
#        classifier.fit(X_train, y_train)
#        
#        X_test = vectorize.transform(X_test_text)
#        #res = classifier.predict_proba(X_test)
#        res = classifier.predict(X_test)
#        scoring.append([classifier.score(X_test, y_test_text), df_m, df_ma])
#        print (classifier.score(X_test, y_test_text), df_m, df_ma)
#        df_ma += 0.01
#result_three = pandas.concat([X_test_text,y_test_text], axis=1)
#res_df = pandas.DataFrame(res)
#final_res = pandas.concat([result_three, res_df], axis=1)

#data_train[1].map(lambda kkk: kkk.lower())
#data_train[1].map.replace('[^а-яА-Яa-zA-Z0-9]', ' ', regex=True)
#data_train[1].map.apply(lambda x: [item for item in x if item not in stop_w])