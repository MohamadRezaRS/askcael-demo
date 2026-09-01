import concurrent
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor

import requests
from bs4 import BeautifulSoup

header = {"User-Agent": "PostmanRuntime/7.32.3"}


def scrape_movie_urls():
    pattern = r'/title/tt[0-9]+/'
    top_page_url = 'https://m.imdb.com/chart/top/'
    top_page_response = requests.get(top_page_url, headers=header)
    print(top_page_response.content)
    top_page_response = BeautifulSoup(top_page_response.content, 'html.parser')
    raw_datas = {}
    movies = top_page_response.find_all('li', class_='ipc-metadata-list-summary-item sc-10233bc-0 iherUv cli-parent')
    for details in movies:
        movie = details.find('a', class_='ipc-title-link-wrapper')
        title = movie.find('h3').get_text()
        url = movie.get('href')
        url = re.findall(pattern, url)
        raw_datas[title] = url

    raw_datas = json.dumps(raw_datas, indent=1)
    with open('data/movie_urls.json', 'w') as file:
        file.write(raw_datas)


def scrape_movie_data():
    print('scraping movie data...')
    movies_infos = {}
    max_threads = 4
    with open('data/movie_urls.json') as movie_urls:
        movie_urls = json.load(movie_urls)
        thread_urls = []
        create_url_list(movie_urls, thread_urls)
        requests_count = 0

        def parse_page(url):
            nonlocal requests_count
            if requests_count % 48 == 0:
                time.sleep(4)
            movie_info = requests.get('https://m.imdb.com' + url + 'plotsummary/?ref_=tt_stry_pl', headers=header)
            movie_info = BeautifulSoup(movie_info.content, 'html.parser')
            requests_count += 1
            summaries_section = movie_info.find('section', class_='ipc-page-section ipc-page-section--base').find('ul')
            all_summaries = summaries_section.find_all('div', class_='ipc-html-content-inner-div')
            title = movie_info.find('h2').get_text()
            print(f'\r█{'█' * (requests_count // 10)} %{int((requests_count / 250) * 100)} | ({requests_count}/250)',
                  end='')
            summaries_content = []
            for summary in set(all_summaries):
                summaries_content.append(summary.get_text())
            summaries_content = set(summaries_content)
            for summary in summaries_content:
                if title in movies_infos.keys():
                    movies_infos[title] += summary
                else:
                    movies_infos[title] = summary

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_threads) as executor:
            executor.map(parse_page, thread_urls)
    print('\nFinished Scraping...')
    movies_infos = json.dumps(movies_infos, indent=1)
    with open('data/movies.json', 'w') as file:
        file.write(movies_infos)


def create_url_list(movie_urls, thread_urls):
    for titles in movie_urls:
        thread_urls.append(movie_urls[titles][0])


def start_process():
    scrape_movie_data()
    checker()


def checker():
    with open('data/movies_infos_raw.json', 'r') as on_check_file:
        on_check_file = json.load(on_check_file)
        print(len(on_check_file.keys()))