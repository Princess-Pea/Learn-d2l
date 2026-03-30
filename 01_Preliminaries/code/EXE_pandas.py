# 本节练习中，我们将使用pandas库来处理数据。
# pandas是一个强大的数据分析工具，提供了丰富的数据结构和函数来处理和分析数据。
# 我们将通过一个简单的例子来演示如何使用pandas来读取、清洗和转换数据，以便后续的机器学习模型训练。

import os, pandas as pd, torch
os.makedirs('data', exist_ok=True)
data_file = os.path.join( 'data', 'house.csv')

with open(data_file,'w') as f:
    f.write('Name,Grade,Score,Mark\n')
    f.write('Jack,2,98,A\n')
    f.write('Rose,NA,95,A\n')
    f.write('Lucy,NA,85,A\n')
    f.write('Tom,NA,78,B\n')
    f.write('Syalis,4,63,C\n')
    f.write('Jerry,NA,NA,NA\n')
    f.write('NA,NA,NA,NA\n')
# 以上代码创建了一个名为house.csv的CSV文件，并写入了一些示例数据。数据包含了姓名、年级、分数和等级四个字段，其中有一些缺失值（NA）。

data = pd.read_csv(data_file)
print(data)
# pd.read_csv()函数用于读取CSV文件，并将其转换为pandas数据框（DataFrame）。数据框是一种二维表格数据结构，类似于Excel表格，可以方便地进行数据操作和分析。

missing_counts = data.isna().sum() # 寻找缺失值最多的一列
print(missing_counts)
col_to_drop = missing_counts.idxmax()
data = data.drop(columns=col_to_drop) # 删除缺失值最多的一列
# data.drop()函数用于删除数据框中的指定列或行。这里我们通过missing_counts.idxmax()找到缺失值最多的列，并将其删除。

inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2:3]

inputs = inputs.fillna(inputs.mean(numeric_only=True)) 
inputs = pd.get_dummies(inputs,dummy_na=True)
outputs = outputs.fillna(outputs.mode().iloc[0])
outputs = pd.get_dummies(outputs,dummy_na=True)
print(inputs)
print(outputs)
# fillna()函数用于填充缺失值。对于数值特征，我们使用均值进行填充；对于分类特征，我们使用众数进行填充。
# get_dummies()函数用于将分类特征转换为数值特征。它会为每个分类值创建一个新的二元特征（0或1），表示该分类值是否存在。dummy_na=True参数表示将缺失值也视为一个分类值进行处理。

inputs = torch.tensor(inputs.to_numpy(dtype=float))
outputs = torch.tensor(outputs.to_numpy(dtype=float))
# to_numpy()将pandas数据框转换为NumPy数组，dtype=float指定数据类型为浮点数。
# torch.tensor()将NumPy数组转换为PyTorch张量。
# 这样我们就得到了一个数值特征的张量inputs和一个数值特征的张量outputs，可以用于后续的机器学习模型训练。

print(inputs)
print(outputs)
# 输入inputs为7×8的数值特征张量
# 输出outputs为7×4的数值特征张量