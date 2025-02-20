import json
import time
import sys

from bs4 import BeautifulSoup
from selenium import webdriver

#URL = 'https://www.yanolja.com/reviews/domestic/10054600?sort=HOST_CHOICE'

def crawl_reviews(name, url):
    review_list = []
    driver = webdriver.Chrome()
    #driver.get(URL)
    driver.get(url)

    time.sleep(3)

    scroll_count = 10
    for i in range(scroll_count):
        driver.execute_script('window.scrollTo(0, document.body.scrollHeight);')
        time.sleep(2)

    html = driver.page_source
    soup = BeautifulSoup(html, 'html.parser')

    review_containers = soup.select('#__next > section > div > div.css-1js0bc8 > div > div > div')
    review_date = soup.select('#__next > section > div > div.css-1js0bc8 > div > div > div > div.css-1toaz2b > div > div.css-1ivchjf')

    for i in range(len(review_containers)):
        review_text = review_containers[i].find('p', class_='content-text').text
        review_stars = review_containers[i].select('path[fill="currentColor"]')
        star_cnt = sum(1 for star in review_stars if not star.has_attr('fill-rule'))
        date = review_date[i].text

        review_dict = {
            'review': review_text,
            'stars': star_cnt,
            'date': date
        }

        review_list.append(review_dict)

    with open(f'/Users/kilminkyu/Desktop/프롬프트 엔지니어링으로 시작하는 LLM 서비스 개발/야놀자 리뷰/res/{name}.json', 'w') as f:
        json.dump(review_list, f, indent=4, ensure_ascii=False)


if __name__ == '__main__':
    name,url = sys.argv[1], sys.argv[2]
    crawl_reviews(name, url)