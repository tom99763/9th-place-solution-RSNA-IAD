from pandas import read_csv

df = read_csv("/home/sersasj/RSNA-IAD-Codebase/data/train.csv")

print(df.head())

df_fold0 = df[df["fold_id"] == 0]

print(df_fold0.head())

df_fold0.to_csv("/home/sersasj/RSNA-IAD-Codebase/data/train_small.csv", index=False)