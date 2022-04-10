import pandas as pd
data = pd.read_excel(io=r'./data/mark/mark.xlsx')
print(data)
print(data[(data['记录点编号']==1483) & (data['数据上传时间']==20220320095836) ])
