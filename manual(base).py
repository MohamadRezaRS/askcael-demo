

import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor
import math
import re

stop_words=['i', 'me', 'my', 'myself', 'we', 'our', 'ours','ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are','was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing','a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y','ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
url = 'https://www.imdb.com/chart/top/'
headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36  (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'}
page = requests.get(url, headers=headers)
print(page.status_code)
soup = BeautifulSoup(page.text, 'html.parser')
movies = soup.find_all(class_='ipc-metadata-list-summary-item__c')

def get_summary(movie_url):
    page = requests.get(movie_url, headers=headers)
    soup = BeautifulSoup(page.text, 'html.parser')
    datas= soup.find_all('div', class_='ipc-metadata-list-item__content-container')
    summaries=[]
    for data in datas:
        summaryp=data.find('div',class_='ipc-html-content-inner-div').get_text(strip=True)
        summaries.append(summaryp)
    return summaries[1]
  
def get_movie_info(movie):
    base='https://www.imdb.com'
    film=movie.find(class_='ipc-title ipc-title--base ipc-title--title ipc-title-link-no-icon ipc-title--on-textPrimary sc-b189961a-9 bnSrml cli-title')
    movie_title = film.find('a').get_text(strip=True)
    movie_url =  base + '/'+'title'+ '/'+ movie.find('a')['href'].split('/')[2] + '/plotsummary/?ref_=tt_stry_pl'
    print(movie_title)
    summary = get_summary(movie_url)     
    return movie_title , summary

with ThreadPoolExecutor() as executor:
    movie_info = executor.map(get_movie_info, movies)

movies_titles , movies_summaries = [],[]

for title, summary in movie_info:
    movies_titles.append(title)
    movies_summaries.append(summary)#['summary1','summary2']

def Normalize_text(lst,stp):
    b=[re.findall(r'\b\w+\b', string.lower())for string in lst]
    x=[[word for word in string if word not in stp] for string in b]
    return x

def get_all_words(lst):
    unique=[]
    for s in lst:
        for i in s:
            if i not in unique:
                unique.append(i)
    return unique

def TF(all_wrd,normalized_summ): #list, nested list
    b=[]
    for i in range(len(normalized_summ)):
        c=[]
        for j in range(len(all_wrd)):
            sui=normalized_summ[i].count(all_wrd[j])
            c.append(sui/len(normalized_summ[i]))
        b.append(c)
    return b

def IDF(all_wrd,normalized_summ): # list , nested list
    all_wrd_idf={}
    for i in range(len(all_wrd)):
        sui=0
        for j in range(len(normalized_summ)):
            if all_wrd[i] in normalized_summ[j]:
                sui+=1
        all_wrd_idf[all_wrd[i]]=math.log(len(normalized_summ)/sui)+1
    return all_wrd_idf

def tf_idf(tfs,idfs):
    idf_values=[idfs[key] for key in idfs.keys()]
    tf_idfs=[[tf*idf for tf,idf in zip(doc,idf_values)] for doc in tfs]
    return tf_idfs

normalized_summaries=Normalize_text(movies_summaries,stop_words)
all_words=get_all_words(normalized_summaries)
summaries_tfs=TF(all_words,normalized_summaries)
words_idfs=IDF(all_words,normalized_summaries)
summaries_tf_idfs=tf_idf(summaries_tfs,words_idfs)

def make_dict(num):
    b={}
    for i in range(num):
        b[movies_titles[i]]=summaries_tf_idfs[i]
    return b

movie_and_vector=make_dict(len(movies_titles))

def make_new_plot_Normall(new_movie_plot,stp):
    a=re.findall(r'\b\w+\b',new_movie_plot.lower())
    normal=[w for w in a if w not in stp]#['cijiefoe','wienoiefn']
    return normal

def get_new_plot_tf(all_wrd,lst):
    b=[]
    for i in range(len(all_wrd)):
        sui=lst.count(all_wrd[i])
        b.append(sui/len(lst))
    return b

def get_new_plot_tf_idfs(liste_tf_ha,idfs):
    idfs_vals=list(idfs.values())
    b=[]
    for i in range(len(idfs_vals)):
        b.append(liste_tf_ha[i]*idfs_vals[i])
    return b

def cosine_similarity(v1,v2):
    dot_product=sum(a*b for a,b in zip(v1,v2))
    magnitude1= math.sqrt(sum(a*a for a in v1))
    magnitude2= math.sqrt(sum(b*b for b in v2))
    if magnitude1==0 or magnitude2==0:
        return 0
    return dot_product/(magnitude1*magnitude2)

def main():
    running=True
    while running:
        film=input('give me the plot summary or story line (story line preferred):...')
        if film.lower()=='exit':
            print('okay')
            running=False
        else:
            notch=int(input('how many do you need?(an integer number please):...')) 
            normall_plot=make_new_plot_Normall(film,stop_words)
            plot_tf=get_new_plot_tf(all_words,normall_plot)
            plot_tf_idf=get_new_plot_tf_idfs(plot_tf,words_idfs) 
            similarities={key: cosine_similarity(value,plot_tf_idf) for key , value in movie_and_vector.items()}
            top_10_keys=sorted(similarities,key=similarities.get, reverse=True)[:notch]
            for i in top_10_keys:
                print(i)

if __name__=='__main__':
    main()
   
