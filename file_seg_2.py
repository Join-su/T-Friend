import pandas as pd

PATH = 'C:\\Users\\ialab\\Desktop\\T-Friend\\'
filename = 'in_file.xlsx'

df = pd.read_excel(PATH + filename, encoding='utf-8', index_col=0)
print(df.head())

#pre_data = df.loc[:, [name,name2]].astype('str')

## 매입/매춰 나누기
C_12 = []
C_34 = []
C_Y2 = []
save_Y2 = 0

for i in range(len(df)):
    #print(df.loc[i,['CD_SCRP']].values)
    name = df.loc[i,['CD_SCRP']].item()
    if name == 'home1in' or name == 'home2in': C_12.append(i)
    elif name == 'home3in' or name == 'home4in': C_34.append(i)
    else : C_Y2.append(i)

df_12 = df.iloc[C_12,:]
df_12 = df_12.reset_index()
df_12 = df_12.drop(['index'], axis=1)

df_34 = df.iloc[C_34,:]
df_34 = df_34.reset_index()
df_34 = df_34.drop(['index'], axis=1)

print(len(C_Y2))
if len(C_Y2) > 0 :
    df_Y2 = df.iloc[C_Y2,:]
    df_Y2 = df_Y2.reset_index()
    df_Y2 = df_Y2.drop(['index'], axis=1)
    save_Y2 = 1


df_12.to_excel('12_file.xlsx', 'w', encoding='utf-8')
df_34.to_excel('34_file.xlsx', 'w', encoding='utf-8')
if save_Y2 == 1 : df_Y2.to_excel('Y2_file.xlsx', 'w', encoding='utf-8')

