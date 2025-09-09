import os

data_path = './data/extract_data'
file_list = os.listdir(data_path)

for file in file_list:
    data = os.path.join(data_path, file)
    if len(os.listdir(data)) == 0:
        print(1)
        os.rmdir(data)