'''
Date: 2022-09-15 16:25:23
LastEditors: ZSudoku
LastEditTime: 2022-09-15 19:53:58
FilePath: \Pistachio_DeepLearning\point.py
'''
lis = []
index = 0
for i in range(4,10):
    for j in range(90,101):
        lis.append([])
        lis[index].append(i)
        lis[index].append(j)
        index += 1
print(lis)