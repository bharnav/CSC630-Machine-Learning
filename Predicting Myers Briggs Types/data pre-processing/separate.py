import os
import pandas as pd

read_path = os.path.join('..', 'datasets', 'raw.csv')
df = pd.read_csv(read_path)

# [E, S, T, P]

labels = [df['type'].str.contains('E'), df['type'].str.contains('S'), df['type'].str.contains('T'), df['type'].str.contains('P')]

texts = df['posts']

for i, label in enumerate(labels):
    new_df = pd.DataFrame(data={'text': texts, 'label': label.astype(int)})
    write_path = os.path.join('..','datasets',str(i)+'.csv')
    new_df.to_csv(write_path, index=False)
