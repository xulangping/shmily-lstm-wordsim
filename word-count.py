from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from cleantext import clean
import os
import pandas as pd
import re

class KeyWord:
    def __init__(self):
        self.stopwords = set(stopwords.words('english'))

    def clean_text(self, text):

        text = re.sub('¬\n', '', text)
        text = re.sub('\n', ' ', text)
        text = re.sub('■', '', text)
        text = re.sub('.(\.\.+)', '', text)
        fields = re.split('([A-Z]+,)', text)
        print(len(fields))
        values = fields[2::2]
        delimiters = fields[1::2]
        print(len(values))

        result = ''
        num = 0
        for delimiter, value in zip(delimiters, values):
            result = result + delimiter + value + '\n'
            num += 1


        return result, num


    def cut_text(self, text):
        l = []
        text = self.clean_text(text)
        for word in word_tokenize(text):
            if word not in self.stopwords and word != '':
                l.append(word)
        return l

    def word_count(self, text):
        return len(self.cut_text(text))

if __name__=="__main__":
    path = '/Users/xulangping/Downloads/nls-text-encyclopaediaBritannica'
    f_ans = open('word_count.txt', 'w')
    kw = KeyWord()
    d = {'file': [], 'text': [], 'num_article': []}
    for f in os.listdir(path):
        s = open(os.path.join(path, f), 'r').read()
        d['file'].append(f)
        text, num_article = kw.clean_text(s)
        d['text'].append(text)
        d['num_article'].append(num_article)
        # f_ans.write(f + ',' + str(kw.word_count(s)) + '\n')
    pd.DataFrame(d).to_csv('data_all.csv', encoding='utf-8-sig', index=0)
