import time

import pandas as pd

from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException
from selenium.common.exceptions import StaleElementReferenceException
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions

from css_selectors import CSS_SELECTORS

PAGE_BOTTOM = 7000
PAGE_MIDDLE = 4000

class SephoraScraper:

    def __init__(self, driver_path, **kwargs):
        self.driver = webdriver.Chrome(driver_path, **kwargs)
        self.base_url = ' '
        self.ignored_exceptions=(NoSuchElementException,StaleElementReferenceException)
        self.product_links = []
        self.products = self._initialize_product_dict()
        self.ratings = self._initialize_ratings()
    
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
    
    def load_checkpoint(self, products_path, reviews_path, product_links):
        self.ratings = pd.read_csv(reviews_path, sep ='\t').to_dict('list')
        self.products = pd.read_csv(products_path).to_dict('list')
        num_products = len(self.products['id'])
        self.product_links = product_links[num_products +1:]

    def scrape_products_and_reviews(self, num_pages_reviews=-1, product_links=[], checkpoints=True, checkpoint_after=100):
        self._instatiate_product_links(product_links)
        product_id = 1
        for product in self.product_links:
            self._get_product_info(product, product_id)
            self._get_product_ratings(product_id, num_pages_reviews)
            if (product_id - 1) % checkpoint_after == 0 and checkpoints:
                self._make_checkpoint(f'../data/chechpoint{int((product_id - 1)/ checkpoint_after)}')
            product_id +=1
        print(60*'=')
        print('Scraping complete!')
        return self.products, self.ratings
    
    def _make_checkpoint(self, pathname):
        name_reviews = pathname + '_reviews.csv'
        name_products = pathname + '_products.csv'
        self.save_ratings_as_csv(name_reviews)
        self.save_products_as_csv(name_products)
    
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
    
    def _update_ratings(self, ratings, users, product_id):
        for rating, user in zip(ratings, users):
            rating = int(rating.get_attribute('aria-label').split(" ")[0])
            user = user.text
            self.ratings['rating'].append(rating)
            self.ratings['product_id'].append(product_id)
            self.ratings['buyer_nickname'].append(user)
    
    def _adjust_page_num(self, num_pages, selector):
        available_pages = self.driver.find_elements_by_css_selector(selector)
        while not available_pages:
            time.sleep(0.5)
            available_pages = self.driver.find_elements_by_css_selector(selector)
        available_pages = int(available_pages[-1].text)
        if num_pages == -1 or num_pages > available_pages:
            num_pages = available_pages
        return num_pages
    

    def _get_product_ratings(self, product_id, num_pages_reviews):
        num_pages_reviews = self._adjust_page_num(num_pages_reviews, CSS_SELECTORS['review_pages'])
        for page in range(1, num_pages_reviews + 1):
            print(f'Scraping reviews on page {page}')
            ratings = WebDriverWait(self.driver, 20, ignored_exceptions=self.ignored_exceptions)\
                            .until(expected_conditions.presence_of_all_elements_located((By.CSS_SELECTOR, CSS_SELECTORS['rating'])))
            users = WebDriverWait(self.driver, 20, ignored_exceptions=self.ignored_exceptions) \
                            .until(expected_conditions.presence_of_all_elements_located((By.CSS_SELECTOR, CSS_SELECTORS['user'])))
            self._update_ratings(ratings, users, product_id)
            self.driver.find_element_by_css_selector(CSS_SELECTORS['next_review_page']).click()
            time.sleep(2)
    
    def _scrape_name_and_seller(self):
        name = self.driver.find_elements_by_css_selector(CSS_SELECTORS['product_name'])
        seller = self.driver.find_elements_by_css_selector(CSS_SELECTORS['seller'])
        while not name or not seller:
            time.sleep(0.5)
            name = self.driver.find_elements_by_css_selector(CSS_SELECTORS['product_name'])
            seller = self.driver.find_elements_by_css_selector(CSS_SELECTORS['seller'])

        name = name[0].text
        seller = seller[0].text

        return name, seller

    def _get_product_info(self, product_link, product_id):
        print(f'Scraping product number {product_id}/{len(self.product_links)}...')
        self.driver.get(product_link)
        self._scroll_to(PAGE_MIDDLE)
        self._close_popup()

        name, seller = self._scrape_name_and_seller()
        self.products['name'].append(name)
        self.products['seller'].append(seller)
        self.products['id'].append(product_id)

    def _get_product_list(self, num_pages):
        print(60*'=')
        self._scrape_page(1)
        num_pages = self._adjust_page_num(num_pages, CSS_SELECTORS['product_pages'])

        for page_num in range(2, num_pages + 1):
            self._scrape_page(page_num)
        print('Finished scraping page for links')
        print(60*'=')
    
    def _scroll_to(self, pos):
        start = 0
        while start < pos:
            self.driver.execute_script("window.scrollTo(%i, %i);" % (start, start+1000))
            start += 1000
            time.sleep(3)

    def _scrape_page(self, page_num):
        print(f'Scraping page {page_num}...')
        url = self.base_url + f'?currentPage={page_num}'
        self.driver.get(url)
        self._close_popup()
        self._scroll_to(PAGE_BOTTOM)
        products = self.driver.find_elements_by_css_selector(CSS_SELECTORS['product_links'])
        print(f'I got {len(products)} links!')
        for product in products:
            link = product.get_attribute('href')
            self.product_links.append(link)