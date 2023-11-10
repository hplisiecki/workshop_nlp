import pandas as pd
import requests

url = 'https://huggingface.co/datasets/merve/poetry/raw/main/poetry.csv'

# download the file
req = requests.get(url)
url_content = req.content
# to csv
csv_file = open('poetry.csv', 'wb')
csv_file.write(url_content)
csv_file.close()

# read csv
df = pd.read_csv('poetry.csv')

# save
df.to_csv('data/poetry.csv', index=False)
