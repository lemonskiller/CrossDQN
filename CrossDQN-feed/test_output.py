my_list = [10, 7, 15, 4, 8, 12, 9]

# 找到前3个最大值
top3_max_values = sorted(my_list, reverse=True)[:3]
print(my_list, top3_max_values)

# 将前3个最大值设置为1，其余位置设置为0
new_list = [1 if x in top3_max_values else 0 for x in my_list]

print(new_list)
