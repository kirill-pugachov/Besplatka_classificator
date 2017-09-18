# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 21:57:53 2017

@author: Kirill
"""

#читаем sitemap.xml
#все категории (не объявление и без /q-) разбираем - title, description, keywords
#все /q- разбираем на запросы и запросы складываем в словарь, где ключ урл бед /q-
#обучаем модель по разобранным категориям - слова входные значения, метка класса урл
#прогоняем запросы из /q- через модель - все что распозналось сохраняем 
#в отдельный список т.к. это запросы категорий и их не должно быть в другом месте
#все что не распозналось - должны быть размещены на страницах типа /q-, если таких
#страниц нет - сохраняем в список для дальнейшего использования
#
#
from multiprocessing.dummy import Pool as ThreadPool
import shelve
import urllib
import xml.etree.ElementTree as ET
import requests
from lxml import html
import csv

sitemap_url = 'https://besplatka.ua/sitemap.xml'
result_file = 'C:/Python/Service/1_research_sitemap_Besplatka.csv'
result_dict = {}
shelve_semantica = 'data_base_semantica'
shelve_query = 'data_base_query'
shelve_url = 'data_base_url'

task_list = ['data_base_semantica', 'data_base_query', 'data_base_url']
csv_columns = ['Key', 'Value']

def product_card(url_text):
    '''Возвращает значение последнего 
    элемента урл, разделенное по слешам'''
    part_list=url_text.split('/')
    return part_list[len(part_list)-1]    


def get_childs(sitemap_url):
    '''Забирает данные из сайтмапа и 
    разбирает на фрагменты'''
    parser = ET.XMLParser(encoding="utf-8")
#    parser = ET.XMLParser(recover=True)
    '''Новый экземпляр парсера с учетом кодировки'''
    REQ = urllib.request.Request(sitemap_url)
    try:        
        with urllib.request.urlopen(REQ) as RESPONSE:
            try:
                PARSED_XML = ET.parse(RESPONSE, parser=parser)
                ROOT = PARSED_XML.getroot()
                childs = ROOT.getchildren()
#                print(type(childs))
            except ET.ParseError:
#                print(RESPONSE)
                childs=None
#                print(type(childs), sitemap_url)
    except urllib.error.HTTPError as e:
#        print(e.code)
#        print(e.read())
        childs=None
#        print(type(childs), sitemap_url)
    return childs

def file_writer(result_file, result_dict):
    '''Записывает полученный список урл с дублями 
    (больше чем одна урл содержит ключ) в итоговый файл'''
    f = open(result_file, 'w')
    for key, value in sorted(result_dict.items()):
        if result_dict[key][0] > 1:
            f.write(str(key.encode().decode('utf-8', 'ignore') + '\t' + str(value))+'\n')
    f.close()


def tags_to_string(header_tag):
    string = ''
#    print(len(header_tag))
    if len(header_tag) > 1:
        for tag in header_tag:
            if len(tag.strip()) > 5:
                string += str(tag).strip() + ' '
            else:
                string += 'no_tag_'               
#        print(string)
        return string
    elif len(header_tag) == 1:
        if len(header_tag[0].strip()) > 5:
            string = str(header_tag[0].strip())
        else:
            string = 'no_tag'            
        return string.strip()
    elif len(header_tag) == 0:
        string = 'no_tag'
        return string.strip()


def get_title(parsed_body):
    '''
    Получаем тайтл из тега <title>
    '''
    try:
        title = parsed_body.xpath('//title/text()')[0]
    except:
        title = 'no_title'
    return title

    
def get_title_1(parsed_body):
    '''
    Получаем тайтл из тега <meta name = 'title' content = '...'>
    '''
    try:
        title = parsed_body.xpath('/html/head/meta[@name="title"]/@content')[0]
    except:
        title = 'no_title_1'
    return title

    
def get_description(parsed_body):
    '''
    Получаем description из тега <meta name = 'description' content = '...'>
    '''
    try:
        description = parsed_body.xpath('/html/head/meta[@name="description"]/@content')[0]
    except:
        description = 'no_description'
    return description


def get_keywords(parsed_body):
    '''
    Получаем тайтл из тега <meta name = 'keywords' content = '...'>
    '''
    try:
        keywords = parsed_body.xpath('/html/head/meta[@name="keywords"]/@content')[0]
    except:
        keywords = 'no_keywords'
    return keywords    


def get_h1(parsed_body):
    '''
    Получаем h1 из тега <h1>
    '''
    try:
        h1 = parsed_body.xpath('//h1/text()')
    except:
        h1 = ['no_h1']
    return tags_to_string(h1)


def get_h2(parsed_body):
    '''
    Получаем h2 из тега <h2>
    '''
    try:
        h2 = parsed_body.xpath('//h2/text()')
#        h2 = ' '.join(parsed_body.xpath('//h2/text()'))
    except:
        h2 = ['no_h2']
    return tags_to_string(h2)


def get_h3(parsed_body):
    '''
    Получаем h3 из тега <h3>
    '''
    try:
        h3 = parsed_body.xpath('//h3/text()')
#        h3 = ' '.join(parsed_body.xpath('//h3/text()'))
    except:
        h3 = ['no_h3']
    return tags_to_string(h3)

    
def get_h4(parsed_body):
    '''
    Получаем h4 из тега <h4>
    '''
    try:
        h4 = parsed_body.xpath('//h4/text()')
#        h4 = ' '.join(parsed_body.xpath('//h4/text()'))
    except:
        h4 = ['no_h4']
    return tags_to_string(h4)


def get_ads_title(parsed_body):
    '''
    Получаем заголовки объявлений
    из категории
    '''
    try:
        ads_title = parsed_body.xpath('//div[@class="title"]/a/text()')
#        print(tags_to_string(ads_title))
    except:
        ads_title = ['no ads at all']
#        print(tags_to_string(ads_title))
    return tags_to_string(ads_title)
        
            
            

def page_tags(parsed_body):
    '''
    Возвращает список тегов на странице
    title, title in meta, description, keywords
    h1, h2, h3, h4
    '''
    page_tags_list = []
    page_tags_list.append(get_title(parsed_body))
    page_tags_list.append(get_title_1(parsed_body))
    page_tags_list.append(get_description(parsed_body))
    page_tags_list.append(get_keywords(parsed_body))
    page_tags_list.append(get_h1(parsed_body))
    page_tags_list.append(get_h2(parsed_body))
    page_tags_list.append(get_h3(parsed_body))
    page_tags_list.append(get_h4(parsed_body)) 
    page_tags_list.append(get_ads_title(parsed_body))
    return page_tags_list


def build_data_semantica(shelve_semantica, sitemap_url, semantica):
    '''
    All semantica from sitamap.xml
    dictionary url:string in shelve
    '''
    with shelve.open(shelve_semantica, writeback=True) as db:
        if sitemap_url in db:
            db[sitemap_url].append(semantica)
        else:
            db[sitemap_url] = [semantica]
        db.sync()


def build_data_semantica_dict(dict_name, sitemap_url, semantica):
    '''
    !!without shelve!!
    All semantica from sitamap.xml
    dictionary url:string
    '''
    if sitemap_url in dict_name:
        dict_name[sitemap_url].append(semantica)
    else:
        dict_name[sitemap_url] = [semantica]
    return dict_name



def get_url(stroka):
    '''
    Возвращает урл без запроса из урл типа /q-
    '''
    return stroka[:stroka.find('/q-')]


def get_query(stroka):
    '''
    Возвращает запрос из урл типа /q-
    '''
    return stroka[stroka.find('/q-')+len('/q-'):]

def build_base(income_name):
    semantica_dict = dict()
    query_dict = dict()
    url_dict = dict()
    
    if income_name == shelve_semantica:
#        print(income_name)
        sitemaps = get_childs(sitemap_url)
        if isinstance(sitemaps, list):
            for sitemap in sitemaps:
                children = sitemap.getchildren()
                FINAL_sitemap_url = children[0].text
                if '-obyavleniya-' not in FINAL_sitemap_url:
                    sitemaps1 = get_childs(FINAL_sitemap_url)
                    if isinstance(sitemaps1, list):
                        for sitemap1 in sitemaps1:
                            children1 = sitemap1.getchildren()
                            FINAL_sitemap_url1 = children1[0].text
#                            print(FINAL_sitemap_url1)
                            if '/transport/' in FINAL_sitemap_url1 and '/q-' not in FINAL_sitemap_url1:
#                                print(FINAL_sitemap_url1)
                                response = requests.get(FINAL_sitemap_url1)
                                parsed_body = html.fromstring(response.text)
                                semantica = ', '.join(page_tags(parsed_body))
                                semantica_dict = build_data_semantica_dict(semantica_dict, FINAL_sitemap_url1, semantica)
        return semantica_dict
    elif income_name == shelve_query:
#        print(income_name)
        sitemaps = get_childs(sitemap_url)
        if isinstance(sitemaps, list):
            for sitemap in sitemaps:
                children = sitemap.getchildren()
                FINAL_sitemap_url = children[0].text
                if '-obyavleniya-' not in FINAL_sitemap_url:
                    sitemaps1 = get_childs(FINAL_sitemap_url)
                    if isinstance(sitemaps1, list):
                        for sitemap1 in sitemaps1:
                            children1 = sitemap1.getchildren()
                            FINAL_sitemap_url1 = urllib.parse.unquote(children1[0].text, encoding='utf-8', errors='replace')
#                            print(FINAL_sitemap_url1)
                            if '/transport/' in FINAL_sitemap_url1 and '/q-' in FINAL_sitemap_url1 and '/all/' not in FINAL_sitemap_url1:
#                                print(FINAL_sitemap_url1)
                                query_dict = build_data_semantica_dict(query_dict, get_query(FINAL_sitemap_url1), get_url(FINAL_sitemap_url1))
        return query_dict
    elif income_name == shelve_url:
#        print(income_name)
        sitemaps = get_childs(sitemap_url)
        if isinstance(sitemaps, list):
            for sitemap in sitemaps:
                children = sitemap.getchildren()
                FINAL_sitemap_url = children[0].text
                if '-obyavleniya-' not in FINAL_sitemap_url:
                    sitemaps1 = get_childs(FINAL_sitemap_url)
                    if isinstance(sitemaps1, list):
                        for sitemap1 in sitemaps1:
                            children1 = sitemap1.getchildren()
                            FINAL_sitemap_url1 = urllib.parse.unquote(children1[0].text, encoding='utf-8', errors='replace')
#                            print(FINAL_sitemap_url1)
                            if '/transport/' in FINAL_sitemap_url1 or '/all/' in FINAL_sitemap_url1 and '/q-' in FINAL_sitemap_url1:
#                                print(FINAL_sitemap_url1)
                                url_dict = build_data_semantica_dict(url_dict, get_query(FINAL_sitemap_url1), get_url(FINAL_sitemap_url1))
        return url_dict


def build_data_shelve(task_list, result_grouped_list):
    '''
    All semantica from sitamap.xml
    dictionary url:string in shelve
    '''
    for I in range(len(task_list)):
        with shelve.open(task_list[I], writeback=True) as db:
            for key in result_grouped_list[I].keys():
                db[key] = result_grouped_list[I][key]


def write_to_csv(csv_file, csv_columns, dict_data):
    '''
    Записывает словарь в файл
    '''
#    try:
    with open(csv_file, 'w', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile, dialect='excel', quoting=csv.QUOTE_MINIMAL)
        for data in dict_data:
            writer.writerow([data, ','.join(map(str,dict_data[data]))])


def save_to_file(task_list, result_grouped_list):
    '''
    Делает имена файлов csv
    для каждого словаря из результирующего
    списка
    '''
    for I in range(len(task_list)):
#        print(I)
        write_to_csv(task_list[I] + '.csv', csv_columns, result_grouped_list[I])
    

def data_preparing(build_base, task_list):
    '''
    Запускает в отдельных процессах
    сбор данных из sitemap.xml
    '''
    result_grouped_list = []
    pool = ThreadPool()
    result_grouped_list.append(pool.map(build_base, task_list))
    pool.close()
    pool.join()
    build_data_shelve(task_list, result_grouped_list)
    save_to_file(task_list, result_grouped_list)
    return result_grouped_list
               


if __name__ == '__main__':
#    results = data_preparing(build_base, task_list)
#    build_data_shelve(task_list, results)
#    save_to_file(task_list, results)


        
#    result_grouped_list = []
#    pool = ThreadPool()
#    result_grouped_list.append(pool.map(build_base, task_list))
#    pool.close()
#    pool.join()
#    build_data_shelve(task_list, result_grouped_list)
#    save_to_file(task_list, result_grouped_list)
    
    result_grouped_list = []
    for task in task_list:
        result_grouped_list.append(build_base(task))
    build_data_shelve(task_list, result_grouped_list)
    save_to_file(task_list, result_grouped_list)    