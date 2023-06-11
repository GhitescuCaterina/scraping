import requests
import pandas as pd
from bs4 import BeautifulSoup
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

df = pd.read_csv('stores.csv')
urls = df['URLs'].tolist()

successful_websites = []
product_names_all = []

for url in urls:
    try:
        response = requests.get(url, verify=False)

        if response.status_code == 200:
            html_content = response.text
            soup = BeautifulSoup(html_content, 'html.parser')
            product_tags = soup.select('.product-title, .prod-title, .product__title')
            product_names = [tag.text.strip() for tag in product_tags]
            product_names_all.extend(product_names)
            successful_websites.append(url)
        else:
            print("Failed to retrieve the webpage. URL:", url, "Status Code:", response.status_code)

    except requests.exceptions.RequestException as e:
        print(f"Failed to retrieve the webpage. URL: {url} Error: {e}")

print()

print("Product Names for Successful Websites:")
for name in product_names_all:
    print(name)

print()

print("Successful Websites:")
for url in successful_websites:
    print(url)
