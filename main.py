import spacy
import random
from spacy.util import minibatch, compounding
import pandas as pd
import requests
import urllib3
from bs4 import BeautifulSoup
import re
from spacy.lang.en import English
from spacy.tokens import Doc
from spacy.training import Example
import warnings

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

nlp = spacy.blank('en')

ner = nlp.add_pipe("ner")

optimizer = nlp.begin_training()

TRAIN_DATA = [
    ("The new sofa is very comfortable", {"entities": [(8, 12, "PRODUCT")]}),
    ("I just bought a new dining table", {"entities": [(19, 31, "PRODUCT")]}),
    ("The new sofa is very comfortable", {"entities": [(8, 12, "PRODUCT")]}),
    ("I just bought a new dining table", {"entities": [(21, 33, "PRODUCT")]}),
    ("IKEA's Malm bed frame is my favorite", {"entities": [(8, 18, "PRODUCT")]}),
    ("The EKTORP three-seat sofa is incredibly cozy", {"entities": [(4, 25, "PRODUCT")]}),
    ("Wayfair's sectional sofas are the best", {"entities": [(11, 27, "PRODUCT")]}),
    ("I need a new office chair for my work from home setup", {"entities": [(13, 24, "PRODUCT")]}),
    ("The PAX wardrobe system from IKEA offers great storage solutions", {"entities": [(4, 16, "PRODUCT")]}),
    ("The coffee table from West Elm has a modern design", {"entities": [(4, 16, "PRODUCT")]}),
    ("My living room features a beautiful leather Chesterfield sofa", {"entities": [(38, 56, "PRODUCT")]}),
    ("I recently got an amazing deal on a Tempur-Pedic mattress", {"entities": [(37, 56, "PRODUCT")]}),
    ("Restoration Hardware has the most luxurious throw pillows", {"entities": [(44, 56, "PRODUCT")]}),
    ("I can't wait to install my new Hemnes bookcase", {"entities": [(30, 44, "PRODUCT")]}),
    ("Ashley Furniture's recliners are extremely comfortable", {"entities": [(20, 29, "PRODUCT")]}),
    ("The outdoor dining set from Lowe's is perfect for summer", {"entities": [(4, 21, "PRODUCT")]}),
    ("I purchased a new patio set for my backyard", {"entities": [(18, 27, "PRODUCT")]}),
    ("Crate & Barrel's area rugs are stylish and affordable", {"entities": [(18, 27, "PRODUCT")]}),
    ("The wing chair is a classic", {"entities": [(4, 23, "PRODUCT")]}),
    ("I adore the new dressing table in my bedroom", {"entities": [(11, 30, "PRODUCT")]}),
]

examples = []
for text, annots in TRAIN_DATA:
    doc = nlp.make_doc(text)
    example = Example.from_dict(doc, annots)
    examples.append(example)

for i in range(20):
    random.shuffle(examples)
    for batch in spacy.util.minibatch(examples, size=4):
        nlp.update(batch)

doc = nlp("I just bought a new dining table")
print("Entities:", [(ent.text, ent.label_) for ent in doc.ents])

nlp.to_disk("product_ner")

nlp = spacy.load("product_ner")

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
            page_text = soup.get_text()

            for text in page_text.split("\n"):
                doc = nlp(text)
                for ent in doc.ents:
                    product_names_all.append(ent.text)

            successful_websites.append(url)
        else:
            print("Failed to retrieve the webpage. URL:", url, "Status Code:", response.status_code)

    except requests.exceptions.RequestException as e:
        print(f"Failed to retrieve the webpage. URL: {url} Error: {e}")


def clean_product_names(product_names_all):
    product_names_clean = [re.sub('[^A-Za-z0-9 ]+', '', name) for name in product_names_all]

    unnecessary_words = ['New', 'Buy Now', 'Sale']
    for word in unnecessary_words:
        product_names_clean = [name.replace(word, '') for name in product_names_clean]

    product_names_clean = [re.sub(' +', ' ', name.strip()) for name in product_names_clean]

    product_names_clean = [name for name in product_names_clean if name != '']

    return product_names_clean


product_names_all = clean_product_names(product_names_all)

print("\nSuccessful websites:")
for url in successful_websites:
    print(url)

print("\n\nPredicted product entities:")
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning)
    for name in product_names_all:
        doc = nlp(name)
        if doc.ents:
            print(f"'{doc.text}' has been classified as '{doc.ents[0].label_}'")