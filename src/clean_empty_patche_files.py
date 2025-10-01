import os

data_path = './data/patch_data'

for i in range(5):
    dir_path = os.listdir(f'{data_path}/fold{i}')
    count_zero = 0
    for uid in dir_path:
        if len(os.listdir(f'{data_path}/fold{i}/{uid}'))==0:
            count_zero+=1
            os.rmdir(f'{data_path}/fold{i}/{uid}')
    print(count_zero)