import pandas as pd
cnn = pd.read_csv('submission_cnn.csv', index_col = 0)
res = pd.read_csv('submission_res.csv', index_col = 0)
res18 = pd.read_csv('submission_res18.csv', index_col = 0)

df_combine = pd.concat([cnn, res, res18],axis=1, )
df_combine = df_combine.mode(axis=1).dropna(axis=1)

df_combine = df_combine.astype('int32')
df_combine.columns = ['Category']
print(df_combine.head())

df_combine.to_csv('predict_by_voting.csv',index=True)