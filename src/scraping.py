import time

import pandas as pd
import pickle

from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException, TimeoutException
from selenium.common.exceptions import StaleElementReferenceException
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions

from css_selectors import CSS_SELECTORS

PAGE_BOTTOM = 7000
PAGE_MIDDLE = 3500

class SephoraScraper:

    def __init__(self, driver_path, **kwargs):
        self.driver = webdriver.Chrome(driver_path, **kwargs)
        self.driver_path = driver_path
        self.kwargs = kwargs
        self.base_url = ' '
        self.ignored_exceptions=(NoSuchElementException,StaleElementReferenceException)
        self.product_links = []
        self.product_id = 0
        self.products = self._initialize_product_dict()
        self.ratings = self._initialize_ratings()
    
    def __getstate__(self):
        state = self.__dict__
        del state['driver']
        state['driver_path'] = self.driver_path
        state['driver_kwargs'] = self.kwargs
        return state
    
    def __setstate__(self, dict):
        dict['driver'] = webdriver.Chrome(dict['driver_path'], **dict['driver_kwargs'])
        del dict['driver_path']
        del dict['driver_kwargs']
        self.__dict__ = dict
    
    def save_products_as_csv(self, path):
        products_df = pd.DataFrame.from_dict(self.products)
        products_df.to_csv(path, sep='\t', index=False)
    
    def save_ratings_as_csv(self, path):
        ratings_df = pd.DataFrame.from_dict(self.ratings)
        ratings_df.to_csv(path, sep='\t', index=False)

    def scrape_links(self, starting_links, num_pages=-1):
        for url in starting_links:
            print('\n' + 60*'=')
            print('Scraping links at ' + url + '...')
            self.base_url = url
            self._get_product_list(num_pages)
        self._save_product_links()
        return self.product_links

    def scrape_products_and_reviews(self, num_pages_reviews=-1, product_links=[], checkpoints=True, checkpoint_after=100):
        self._instatiate_product_links(product_links)
        self.product_id += 1
        for product in self.product_links:
            self._get_product_info(product)
            self._get_product_ratings(num_pages_reviews)
            if checkpoints:
                self._make_checkpoint(checkpoint_after)
            self.product_id +=1
        print(60*'=')
        print('Scraping complete!')
        return self.products, self.ratings
    
    def _make_checkpoint(self, checkpoint_after):
        pathname = '../data/' + str(self.product_id / checkpoint_after) + 'pkl'
        with open(pathname, 'wb') as checkpoint:
            pickle(checkpoint, self)
    
    def _instatiate_product_links(self, product_links):
        if not self.product_links:
            if product_links:
                self.product_links = product_links
            else:
                raise TypeError('No list of product links was found. Either  scrape one '\
                    +'using .scrape_links() or pass one in the product_links parameter')

    def _save_product_links(self):
        links_df = pd.DataFrame(self.product_links, columns=['links'])
        file_name = '../data/product_links.csv'
        links_df.to_csv(file_name, sep='\t', index=False)

    def _initialize_product_dict(self):
        product_dict = {
            'id':[],
            'name':[],
            'seller':[]
        }
        return product_dict
    
    def  _initialize_ratings(self):
        rating_dict = {
            'product_id':[],
            'rating':[],
            'buyer_nickname':[]
        }
        return rating_dict
    
    def _close_popup(self):
        popup = self.driver.find_elements_by_css_selector(CSS_SELECTORS['popup'])
        if popup:
            popup[0].click()
            time.sleep(1)
    
    def _update_ratings(self, ratings, users):
        for rating, user in zip(ratings, users):
            rating = int(rating.get_attribute('aria-label').split(" ")[0])
            user = user.text
            self.ratings['rating'].append(rating)
            self.ratings['product_id'].append(self.product_id)
            self.ratings['buyer_nickname'].append(user)
    
    def _adjust_page_num(self, num_pages, selector):
        try:
            available_pages = WebDriverWait(self.driver, 20, ignored_exceptions=self.ignored_exceptions)\
                              .until(expected_conditions.presence_of_all_elements_located((By.CSS_SELECTOR, selector)))
            available_pages = int(available_pages[-1].text)
        except TimeoutException:
            available_pages = 1
        if num_pages == -1 or num_pages > available_pages:
            num_pages = available_pages
        return num_pages
    

    def _get_product_ratings(self, num_pages_reviews):
        num_pages_reviews = self._adjust_page_num(num_pages_reviews, CSS_SELECTORS['review_pages'])
        for page in range(1, num_pages_reviews + 1):
            print(f'Scraping reviews on page {page}')
            ratings = WebDriverWait(self.driver, 20, ignored_exceptions=self.ignored_exceptions)\
                            .until(expected_conditions.presence_of_all_elements_located((By.CSS_SELECTOR, CSS_SELECTORS['rating'])))
            users = WebDriverWait(self.driver, 20, ignored_exceptions=self.ignored_exceptions) \
                            .until(expected_conditions.presence_of_all_elements_located((By.CSS_SELECTOR, CSS_SELECTORS['user'])))
            self._update_ratings(ratings, users, self.product_id)
            if page - num_pages_reviews > 0:
                self.driver.find_element_by_css_selector(CSS_SELECTORS['next_review_page']).click()
                WebDriverWait(self.driver, 20).until(expected_conditions.staleness_of(users[0]))
    
    def _scrape_name_and_seller(self):
        name = WebDriverWait(self.driver, 20, ignored_exceptions=self.ignored_exceptions)\
                            .until(expected_conditions.presence_of_all_elements_located((By.CSS_SELECTOR, CSS_SELECTORS['product_name'])))
        seller = WebDriverWait(self.driver, 20, ignored_exceptions=self.ignored_exceptions) \
                            .until(expected_conditions.presence_of_all_elements_located((By.CSS_SELECTOR, CSS_SELECTORS['seller'])))
        name = name[0].text
        seller = seller[0].text
        print(name, seller)

        return name, seller

    def _get_product_info(self, product_link):
        print(f'Scraping product number {self.product_id}/{len(self.product_links)}...')
        self.driver.get(product_link)
        self._scroll_to(PAGE_MIDDLE, step=500)
        self._close_popup()

        name, seller = self._scrape_name_and_seller()
        self.products['name'].append(name)
        self.products['seller'].append(seller)
        self.products['id'].append(self.product_id)

    def _get_product_list(self, num_pages):
        print(60*'=')
        self._scrape_page(1)
        num_pages = self._adjust_page_num(num_pages, CSS_SELECTORS['product_pages'])

        for page_num in range(2, num_pages + 1):
            self._scrape_page(page_num)
        print('Finished scraping page for links')
        print(60*'=')
    
    def _scroll_to(self, pos, step):
        start = 0
        while start < pos:
            self.driver.execute_script("window.scrollTo(%i, %i);" % (start, start+step))
            start += step
            time.sleep(3)

    def _scrape_page(self, page_num):
        print(f'Scraping page {page_num}...')
        url = self.base_url + f'?currentPage={page_num}'
        self.driver.get(url)
        self._close_popup()
        self._scroll_to(PAGE_BOTTOM, step=1000)
        products = self.driver.find_elements_by_css_selector(CSS_SELECTORS['product_links'])
        print(f'I got {len(products)} links!')
        for product in products:
            link = product.get_attribute('href')
            self.product_links.append(link)