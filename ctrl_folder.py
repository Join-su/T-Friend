import os

test_folder = "./test"
train_folder = "./train"
req_folder = "./REQ"
res_folder = "./RES"

folder_list = [test_folder, train_folder, req_folder, res_folder]

for folder in folder_list:
    if not os.path.exists(folder):
        os.mkdir(folder)
