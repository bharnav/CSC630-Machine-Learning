import pandas as pd

df = pd.read_csv('../datasets/raw.csv')

# [E, S, T, P]

labels = [df['type'].str.contains('E'), df['type'].str.contains('S'), df['type'].str.contains('T'), df['type'].str.contains('P')]

texts = df['posts']

for i, label in enumerate(labels):
    new_df = pd.DataFrame(data={'text': texts, 'label': label.astype(int)})
    new_df.to_csv('../datasets/' + str(i)+'.csv', index=False)
