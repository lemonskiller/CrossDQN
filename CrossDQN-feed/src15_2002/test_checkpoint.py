# -*- coding: utf-8 -*-
from main import *

if __name__ == '__main__':
    #  --- 查看网络结构 ---
    # 加载已保存的模型
    checkpoint_path = MODEL_SAVE_PATH

    # 列出检查点文件中的所有变量
    variable_list = tf.train.list_variables(checkpoint_path)

    # 打印变量的名称、形状和数值
    for var_name, var_shape in variable_list:
        var_value = tf.train.load_variable(checkpoint_path, var_name)
        print("Variable name: ", var_name)
        print("Variable shape: ", var_shape)
        # print("Variable value: ", var_value)
        print("---")
