import pandas as pd
import numpy as np
import os

data_path = 'data/'
res_file_name = 'all_data.csv'
df = pd.DataFrame(columns=['X', 'Y', 'W', 'Lidar_Data'])

for filename in os.listdir(data_path):
    with open(os.path.join(data_path, filename), 'r') as f:
        for line in f:
            readed_line = line
            curr_data = [readed_line[:-1].split(';')[0].split(','),
                         readed_line[:-1].split(';')[1].split(
                             ',')]  # [0] - odo, [1] - lidar
            parsed_data = []
            for i, l in enumerate(curr_data):
                # print(l,i)
                if i == 0:
                    for j, el in enumerate(l):
                        parsed_data.append(float(el))
                else:
                    parsed_data.append([float(item) for item in l])
            # в итоге 4 ячейки: float, float, float, list
            data = parsed_data[:3]
            data.append(parsed_data[3][85:-85])
            df.loc[len(df)] = data
df.to_csv(res_file_name, header=True, sep=';', index=False)
np.save('all_data_like_pd', df.to_numpy())
