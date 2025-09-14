import os

data_path = './data/extract_data'

for i in range(3):
    total = 0
    file_list = os.listdir(f'{data_path}/fold{i}')
    for file in file_list:
        data = os.path.join(f'{data_path}/fold{i}', file)
        if len(os.listdir(data)) == 0:
            total+=1
            os.rmdir(data)
    print(total)