# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 11:00:53 2017

@author: Kirill
"""

import csv

file_name = 'SemanticSearch.csv'
file_name_save = 'Serch_1_res'
location = 'storage_1/'

def get_url(stroka):
    '''
    Возвращает урл без запроса из урл типа /q-
    '''
    return stroka[:stroka.find('/q-')]


def get_query(stroka):
    '''
    Возвращает запрос из урл типа /q-
    '''
    return stroka[stroka.find('/q-')+len('/q-'):stroka.find(';')]


def get_file_row(file_name):
    '''
    Генератор для чтения строк csv файла
    '''
    with open(file_name, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            yield row


def put_result_to_file(file_name_save, location, row):
    '''
    Запись отобранных строк в файл
    '''
    with open(location + file_name, 'a', encoding='utf-8') as file:
        writer = csv.writer(file, dialect='excel', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(row)
        
        
    
    
            
if __name__ == '__main__':
    
    stroka = get_file_row(file_name)
    line = next(stroka)
    while line:
        if '/transport/' in line[0]:
            url = get_url(line[0])
            query = get_query(line[0])
            print(query, '\t', url)
            put_result_to_file(file_name_save, location, (url, query))
        try:
            line = next(stroka)
        except StopIteration:
            break