# -*- coding: utf-8 -*-
from main import *

if __name__ == '__main__':
    estimator = create_estimator()
    # 指定.pb文件的路径
    pb_path = 'pb/1684252273/saved_model.pb'
    pb_dir = 'pb/1684252273/'

    # --- 方法1 ---
    # with tf.gfile.GFile(pb_path, 'rb') as f:
    #     graph_def = tf.GraphDef()
    #     graph_def.ParseFromString(f.read()) # 报错 google.protobuf.message.DecodeError: Error parsing message
    #     print(graph_def)

    # --- 方法2 ---
    # 加载Estimator保存的.pb文件
    # loaded_model = tf.saved_model.load(pb_dir) # 报错
    # loaded_model = tf.saved_model.load_v2(pb_dir)  # 空
    # print(loaded_model)

    # 获取模型的变量
    # variables = loaded_model.variables
    # print(variables)

    # # 遍历每个变量并打印变量名、大小和数值
    # for var in variables:
    #     var_name = var.name.decode()  # 将变量名从字节字符串解码为字符串
    #     var_size = var.shape.num_elements()
    #     var_value = var.numpy()
    #     print('Variable Name:', var_name)
    #     print('Variable Size:', var_size)
    #     print('Variable Value:')
    #     print(var_value)
    #     print('---------------------------')

    # --- 方法3 ---
    # model = tf.estimator.Estimator(
    #     model_fn=model_fn, model_dir=model_config.get("log_dir"), config=config)

    # --- 方法4 ---
    # 加载 .pb 文件中的模型
    graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(pb_path, 'rb') as f:
        graph_def.ParseFromString(f.read())

    # 获取模型中的变量列表
    variable_list = tf.graph_util.list_variables(graph_def)

    # 遍历每个变量并打印变量名称、形状和数据类型
    for var in variable_list:
        print('Variable Name:', var[0])
        print('Variable Shape:', var[1])
        print('Variable Data Type:', var[2])
        print('---------------------------')
