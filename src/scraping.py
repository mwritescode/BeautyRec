import time

import pandas as pd

from selenium import webdriver

class SephoraScraper:

    def __init__(self, driver_path, base_url, **kwargs):
        self.driver = webdriver.Chrome(driver_path, **kwargs)
        self.base_url = base_url
        self.body_height = 7000
        self.product_links = []
        self.products = self._initialize_product_dict()
        self.ratings = self._initialize_ratings()
    
    def save_products_as_csv(self, path):
        products_df = pd.DataFrame.from_dict(self.products)
        products_df.to_csv(path, sep='\t')
    
    def save_ratings_as_csv(self, path):
        ratings_df = pd.DataFrame.from_dict(self.ratings)
        ratings_df.to_csv(path, sep='\t')

    def scrape(self, num_pages=-1, num_pages_reviews=-1,):
        self._get_product_list(num_pages)
        product_id = 1
        for product in self.product_links:
            self._get_product_info(product, product_id)
            product_id +=1
        return self.products, self.ratings

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
        popup = self.driver.find_elements_by_class_name("css-1kna575")
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
    
    def _adjust_ratings_page_num(self, num_page_reviews):
        page_num = self.driver.find_elements_by_class_name('css-exi524')[-1].text
        page_num = int(page_num)
        if num_page_reviews == -1 or num_page_reviews > page_num:
            num_page_reviews = page_num

    def _get_product_ratings(self, product_id, num_pages_reviews):
        num_pages_reviews = self._adjust_ratings_page_num(num_pages_reviews)
        for _ in range(1, num_pages_reviews + 1):
            self._close_popup()
            ratings = self.driver.find_elements_by_css_selector("div[class='css-4qxrld']")
            users = self.driver.find_elements_by_css_selector("div[class='css-z04cd8 eanm77i0'] strong")
            self._update_ratings(ratings, users, product_id)
            self.driver.find_element_by_css_selector("button[class='css-2anst8']").click()
            time.sleep(1)  
    
    def _get_product_info(self, product_link, product_id):
        print(60*'=')
        print(f'Scraping product number {product_id}/{len(self.products)}...')
        self.driver.get(product_link)

        name = self.driver.find_element_by_css_selector("span[class='css-1pgnl76 e65zztl0']").text
        seller = self.driver.find_element_by_css_selector("a[class='css-nc375s e65zztl0']").text

        self.products['name'].append(name)
        self.products['seller'].append(seller)
        self.products['id'].append(product_id)

        self._get_product_ratings(product_id)

    def _get_product_list(self, num_pages):        
        self._scrape_page(1)
        available_pages = self.driver.find_elements_by_css_selector("button[class='css-p4voop eanm77i0']")[-1].text
        available_pages = int(available_pages)
        if num_pages == -1 or num_pages > available_pages:
            num_pages = available_pages

        for page_num in range(2, num_pages + 1):
            self._scrape_page(page_num)
    
    def _scroll_to_bottom(self):
        start = 0
        while start < self.body_height:
            self.driver.execute_script("window.scrollTo(%i, %i);" % (start, start+1000))
            start += 1000
            time.sleep(2)

    def _scrape_page(self, page_num):
        print(60*'=')
        print(f'Scraping page {page_num}...')
        url = self.base_url + f'?currentPage={page_num}'
        self.driver.get(url)
        self._scroll_to_bottom()
        products = self.driver.find_elements_by_css_selector("div[class='css-dkxsdo'] a")
        for product in products:
            link = product.get_attribute('href')
            self.product_links.append(link)
        print('Finished scraping pages for links')