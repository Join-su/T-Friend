import pandas as pd




class FileSeg():
    def file_seg(self, save_path, path_2):

        filename = 'test_file.xlsx'

        df = pd.read_excel(path_2 + filename, encoding='utf-8', index_col=0)
        print(df.head())

        # pre_data = df.loc[:, [name,name2]].astype('str')

        ## 매입/매춰 나누기
        C_in = []
        C_out = []

        for i in range(len(df)):
            # print(df.loc[i,['CD_SCRP']].values)
            if df.loc[i,['TP_BIZ_C']].isnull().values.any():df.loc[i,['TP_BIZ_C']]=0
            name = df.loc[i, ['CD_TRAN']].item()
            name_list = list(name)
            last_num = len(name_list) - 1
            if name_list[last_num] == 'n':
                C_in.append(i)
            else:
                C_out.append(i)

        df_in = df.iloc[C_in, :]
        df_in = df_in.reset_index()
        df_in = df_in.drop(['index'], axis=1)

        df_out = df.iloc[C_out, :]
        df_out = df_out.reset_index()
        df_out = df_out.drop(['index'], axis=1)
        df_out['CD_ACCOUNT'] = 401
        #df_out['CD_DEDU'] = 0

        #save_path = 'C:\\Users\\ialab\\Desktop\\T-Friend\\process\\'

        df_in.to_excel(save_path + 'in_file.xlsx', 'w', encoding='utf-8')
        df_out.to_excel(save_path + 'out_file.xlsx', 'w', encoding='utf-8')

