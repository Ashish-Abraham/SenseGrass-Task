import requests
url = "http://localhost:9696/predict"

wine_record = {
    'user_name': ['@vboone'],
    'country': ['US'],
    'review_title': ['Hendry 2012 Blocks 7 & 22 Zinfandel (Napa Valley)'],
    'review_description': ['This opens with herbaceous dollops of thyme and fresh-dug earth and evolves slowly in the glass, revealing juicy black plum and berry. Medium in weight and density, its soft and leathery, with a lengthy, supple finish.'],
    'designation': ['Blocks 7 & 22'],
    'points': [90],
    'price': [35],
    'province': ['California'],
    'region_1': ['Napa Valley'],
    'region_2': ['Napa'],
    'winery': ['Hendry']
}

r = requests.post(url, json = wine_record)
print(r.text.strip())


