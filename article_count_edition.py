import pandas as pd
import os


path = '/Users/xulangping/Downloads/nls-text-encyclopaediaBritannica'

df1 = pd.read_csv('data_all.csv')[['file', 'text', 'num_article']]
df_edit = pd.read_csv(os.path.join(path, 'encyclopaediaBritannica-inventory.csv'), header=None)
df_edit.columns = ['file', '0', 'edition']
df_edit = df_edit[['file', 'edition']]
df = pd.merge(df1, df_edit, on='file')
s = df.groupby('edition').sum()['num_article']
s.to_csv('article_num_edition.csv', encoding='utf-8-sig')

d = {'article': [], 'num_word': []}
for edit in df['edition'].unique():
    df2 = df[df['edition']==edit]
    num = 0
    for i in df2['text']:
        for article in i.split('\n'):
            d['article'].append(edit + '_' + str(num))
            d['num_word'].append(len(article.strip().split()))
            num += 1
pd.DataFrame(d).to_csv('article_word_num.csv', index=0, encoding='utf-8-sig')
