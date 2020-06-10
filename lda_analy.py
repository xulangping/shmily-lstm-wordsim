from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import os



path = '/Users/xulangping/Downloads/nls-text-encyclopaediaBritannica'

df1 = pd.read_csv('data_all.csv')[['file', 'text', 'num_article']]
df_edit = pd.read_csv(os.path.join(path, 'encyclopaediaBritannica-inventory.csv'), header=None)
df_edit.columns = ['file', '0', 'edition']
df_edit = df_edit[['file', 'edition']]
df = pd.merge(df1, df_edit, on='file')
print(df.head())
df.dropna(inplace=True)
docs = []
d = {'article': [], 'topic': []}
for edit in df['edition'].unique():
    df2 = df[df['edition']==edit]
    num = 0
    for i in df2['text']:
        for article in i.split('\n'):
            d['article'].append(edit + '_' + str(num))
            docs.append(article)

cntVector = CountVectorizer(stop_words='english')
cntTf = cntVector.fit_transform(docs)
feat_name = cntVector.get_feature_names()

k=50
lda = LatentDirichletAllocation(n_components=k, random_state=0)
docres = lda.fit_transform(cntTf)

with open('lda_key_word.txt', 'w', encoding='utf-8') as f:
    for i, component in enumerate(lda.components_):
        top_k = []
        component /= np.sum(component)
        for word, score in sorted(zip(feat_name, component), key=lambda x: x[1], reverse=True)[:10]:
            top_k.append(word + ': ' + str(round(score, 4)))
        f.write('topic ' + str(i) + '\n')
        f.write(' '.join(top_k) + '\n')
        f.write('******************************' + '\n')

d['topic'] = np.argmax(docres, axis=1)
pd.DataFrame(d).to_csv('article_topic.csv', index=0, encoding='utf-8-sig')