import os

import pandas as pd

from scraping import SephoraScraper

# System path to chrome webdriver and urls to scrape
PATH = 'C:/WebDrivers/chromedriver.exe'

MOISTURIZERS = {
    'url': 'https://www.sephora.com/shop/moisturizer-skincare',
    'path': '../data/moisturizers' }

TONERS = {
    'url': 'https://www.sephora.com/shop/facial-toner-skin-toner',
    'path': '../data/toners' }

FACE_MASKS = {
    'url': 'https://www.sephora.com/shop/facial-treatment-masks',
    'path': '../data/face_masks' }

SCRUBS = {
    'url': 'https://www.sephora.com/shop/exfoliating-scrub-exfoliator',
    'path': '../data/scrubs'}

def download_data():
    categories = [MOISTURIZERS, TONERS, FACE_MASKS, SCRUBS]
    for category in categories:
        path = category['path']
        if not os.listdir(path):
            scraper = SephoraScraper(driver_path=PATH,
                                    base_url=category['url'])
            scraper.scrape(num_pages_reviews=2, num_pages=1)
            scraper.save_products_as_csv(os.path.join(path, 'products.csv'))
            scraper.save_ratings_as_csv(os.path.join(path, 'ratings.csv'))

def remove_nicknames(ratings):
    ratings['buyer_id'] = pd.factorize(ratings['buyer_nickname'])[0]
    ratings = ratings.drop(['buyer_nickname'], axis=1)
    return ratings


#download_data()
ratings = pd.read_csv('../data/moisturizers/ratings.csv', sep='\t')
remove_nicknames(ratings)