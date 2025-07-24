

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor

url = 'https://www.imdb.com/chart/top/'
headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36  (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'}
page = requests.get(url, headers=headers)
soup = BeautifulSoup(page.text, 'html.parser')
movies = soup.find_all(class_='ipc-metadata-list-summary-item__tc')

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
    movie_title = movie.find('a').get_text(strip=True)
    movie_url =  base + '/'+'title'+ '/'+ movie.find('a')['href'].split('/')[2] + '/plotsummary/?ref_=tt_stry_pl'
    summary = get_summary(movie_url)     
    return movie_title , summary

with ThreadPoolExecutor() as executor:
    movie_info = executor.map(get_movie_info, movies)

movies_titles , movies_summaries = [],[]

for title, summary in movie_info:
    movies_titles.append(title)
    movies_summaries.append(summary)#['summary1','summary2']

def clean_text(text, stop_words):
    return ' '.join([word.lower() for word in text.lower().split(' ') if word.lower() not in stop_words])

def clean_summary(summaries,stop_words):
    return [clean_text(story_line,stop_words) for story_line in summaries]

def tfidf_calculater(summaries):
    vectorizer= TfidfVectorizer()
    return vectorizer , vectorizer.fit_transform(summaries)

def process_new_story(new, vectorizer,stop_words):
    cleanStory= clean_text(new , stop_words)
    return vectorizer.transform([cleanStory])

def cosine(new_v,tfidfs):
    return cosine_similarity(new_v, tfidfs).flatten()

def find_top10(cosines,movie_names):
    top10=cosines.argsort()[:-11:-1]
    return [movie_names[index] for index in top10]

stop_words=['i', 'me', 'my', 'myself', 'we', 'our', 'ours','ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are','was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing','a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y','ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
cleaned_stories=clean_summary(movies_summaries , stop_words)
vectorizer, tfidfs= tfidf_calculater(cleaned_stories)
while True:
    new_film=input('new story: ')
    if new_film=='exit':
        print('okay')
        break
    clean_new=clean_text(new_film,stop_words)
    new_vec=process_new_story(clean_new, vectorizer, stop_words)
    cosinesss=cosine(new_vec,tfidfs)
    top_ten=find_top10(cosinesss, movies_titles)
    for i in top_ten:
        print(i)
