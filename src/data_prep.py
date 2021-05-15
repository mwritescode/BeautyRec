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
            scraper.scrape(num_pages_reviews=150)
            scraper.save_products_as_csv(os.path.join(path, 'products.csv'))
            scraper.save_ratings_as_csv(os.path.join(path, 'ratings.csv'))

download_data()