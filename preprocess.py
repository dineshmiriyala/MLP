# this is an extension of preprocessing script in Bigram model repo (www.github.com/dineshmiriyala/Bigram_model/)

import regex
import warnings

warnings.filterwarnings('ignore')
import os
import requests
import zipfile
import io


def text_processing(text):
    """returns the string removing any emojis in them and also special charecters"""
    text = regex.sub(r'\p{So}', '', text)
    pattern = regex.compile('\u30c4')
    text = pattern.sub('', text)
    text = regex.sub('[^a-zA-Z\s]', '', text)
    text = regex.sub('\s+', ' ', text).strip()
    return text


def text_clean(lines):
    """Deletes items with more than three repeating characters."""
    new_string = []
    pattern = regex.compile(r'\b\w*(\w)\1{2,}\w\b')
    for line in lines:
        line = regex.sub(pattern, '', line)
        new_string.append(regex.sub(r"(.)\1+", r"\1", line))
    return new_string


url = 'https://affect.media.mit.edu/neural_chat/datasets/reddit_casual.zip'
req = requests.get(url)
zip = zipfile.ZipFile(io.BytesIO(req.content))

data = zip.read('reddit_casual.json').decode('utf8')

convos = []
for line in data:
    char = []
    for item in line['lines']:
        text = item['text']
    convos.append(text)
unique_convos = set(convos)
unique_convos = [*unique_convos]
final_list = []
for item in unique_convos:
    filtered = text_processing(item)
    final_list.append(filtered)

final_list = text_clean(final_list)

print(f'Sample data: \n{final_list[:5]}')

if not os.path.exists('reddit_convos.txt'):
    with open('reddit_convos.txt', 'w') as file:
        file.write(final_list)
        print("Created processed text file successfully!")
else:
    print("Processed text file exists.")
