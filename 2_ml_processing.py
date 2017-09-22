# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 00:57:15 2017

@author: Kirill
"""


#from scipy.sparse import hstack
from nltk.corpus import stopwords
import pandas
#from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.externals import joblib
import csv

size = 10


def clean_text(data_to_learn):
    '''clean up income data'''
    data_to_learn[1] = data_to_learn[1].map(lambda x: str(x))
    data_to_learn[1] = data_to_learn[1].map(lambda x: x.lower())
    data_to_learn[1] = data_to_learn[1].replace('[^а-яА-Яa-zA-Z0-9]', ' ', regex=True)
    data_to_learn[1] = data_to_learn[1].map(lambda x: ' '.join([item.strip() for item in x.split(' ') if item not in stop_w]))
    return data_to_learn

     
def read_row(X):
    '''read the one row from income'''
    for ind in X.index:
        label = X.iloc[ind]
#        print('label from DataRame', label, '\n')
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


def model_education():
    data_train = pandas.read_csv('storage_1/data_base_semantica.csv',  header=None)
    gen_text = read_row(data_train)
    data_to_learn = clean_text(get_minibatch_1(gen_text, size))
    k = 0
    cls_list = list()
    while list(data_to_learn.index):
        vectorize = HashingVectorizer(decode_error='ignore', n_features=2 ** 21)
        classifier = SGDClassifier(loss='log', warm_start=True, n_jobs=-1, max_iter=5)        
#        cls_list.append(classifier.fit(vectorize.transform(data_to_learn[1]), data_to_learn[0]))
        classifier.fit(vectorize.transform(data_to_learn[1]), data_to_learn[0])
        _ = joblib.dump(classifier, str(k), compress=9)
        cls_list.append(str(k))
        k += size
        print('Обучено строк', k)
        try:
            data_to_learn = clean_text(get_minibatch_1(gen_text, size))
        except TypeError:
            break    
    return cls_list, _

if __name__ == '__main__':
    
##читаем файл csv и разделяем метки класса и тексты для обучения для обучения
##чистим тексты для обучения - все приводим к написанию строчными буквами
##удаляем из текстов все кроме букв и цифр
    stop_w = stopwords.words('russian') + ['tag', 'no', 'title', '9989']
#    cls_list = model_education()
    

            

#читаем файл csv с имеющимися запросами для тестирования
#чистим тексты для обучения - все приводим к написанию строчными буквами
#удаляем из текстов все кроме букв и цифр
#bag_of_words =' '.join(X_train_text.tolist())
#    res_list_url = list()
#    res_list_proba = list()
    total_result = list()
    data_test = pandas.read_csv('storage_1/data_base_query.csv',  header=None)
    data_test.columns = [1,0]
    data_test = clean_text(data_test)
    cls_list = [str(x) for x in range(0,360,10)]
    vectorize = HashingVectorizer(decode_error='ignore', n_features=2 ** 21)
    for data_ind in data_test.index:
        X_test_text = data_test.iloc[data_ind][1]
        y_test_text = data_test.iloc[data_ind][0]
        X_test = vectorize.transform([X_test_text])
        res_list_url = list()
        res_list_proba = list()
        for cls in cls_list:
#    for cls in cls_list[0]:
            classifier = joblib.load(cls)
            res_list_url.append(classifier.predict(X_test))
            res_list_proba.append(classifier.predict_proba(X_test)[0][0])
        total_result.append((X_test_text, res_list_url[res_list_proba.index(max(res_list_proba))][0],max(res_list_proba)))
        with open('result_transport_query_classification.csv', 'a') as file:
            writer = csv.writer(file, delimiter=";")
            writer.writerow((X_test_text, res_list_url[res_list_proba.index(max(res_list_proba))][0],max(res_list_proba)))



#    print(total_result)
    



#        result = zip(res_list_url, res_list_proba)
        
#    X_test_text = clean_text(data_test)[1]
#    y_test_text = data_test[0]
#    vectorize = HashingVectorizer(decode_error='ignore', n_features=2 ** 21)
##    X_test = vectorize.transform(X_test_text)
#    res_list_url = list()
#    res_list_proba = list()
#    cls_list = [str(x) for x in range(0,360,10)]
#    for cls in cls_list:
#    for cls in cls_list[0]:
#        classifier = joblib.load(cls)
#        res_list_url.append(classifier.predict(X_test))
#        res_list_proba.append(classifier.predict_proba(X_test))
#    result = zip(res_list_url, res_list_proba)
#    print (classifier.score(X_test, y_test_text))    

#    data_train = pandas.read_csv('storage_1/data_base_semantica.csv',  header=None)
#    gen_text = read_row(data_train)
#    data_to_learn = clean_text(get_minibatch_1(gen_text, size))
#    k = 0
#    cls_list = list()
#    while list(data_to_learn.index):
#        vectorize = HashingVectorizer(decode_error='ignore', n_features=2 ** 18, non_negative=True)
#        classifier = SGDClassifier(loss='log', warm_start=True, n_jobs=-1, max_iter=5)        
##        cls_list.append(classifier.fit(vectorize.transform(data_to_learn[1]), data_to_learn[0]))
#        classifier.fit(vectorize.transform(data_to_learn[1]), data_to_learn[0])
#        _ = joblib.dump(classifier, str(k), compress=9)
#        cls_list.append(str(k))
#        k += size
#        print('Обучено строк', k)
#        try:
#            data_to_learn = clean_text(get_minibatch_1(gen_text, size))
#        except TypeError:
#            break



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