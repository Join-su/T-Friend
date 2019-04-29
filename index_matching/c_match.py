import pandas as pd


raw_DATA = '3friend_raw_data.xlsx'
d18_train = 'train_test18.xlsx'
d17_train = 'train_test17.xlsx'
excel_PATH = "C:\\Git\\T-Friend\\3friend_raw_data\\"
img_PATH = "C:\\Git\\T-Friend\\img_predict_data"
img2_PATH = "C:\\Git\\T-Friend\\new_train_img_data"
predict_PATH = "C:\\Git\\T-Friend\\predict_excel\\"
predict_FILE = 'predict.xlsx'


data = pd.read_excel(predict_PATH + predict_FILE, encoding='utf-8')

data = data.iloc[:, 1]
LIST = list(data)


index = ['관리비', '기타', '비용', '상품', '수도료', '수수료', '식품', '통신비']


predicted_label = []

for i, pre in enumerate(LIST):
    predicted_label.append(index[pre])

df = pd.DataFrame(predicted_label)
df.to_excel(predict_PATH + "predict_category.xlsx")