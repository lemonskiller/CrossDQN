# CrossDQN

This repository is the implementation of CrossCDN.

## Required packages

The code has been tested running under Python 3.6.9, with the following packages installed (along with their dependencies):
- tensorflow==1.14.0
- numpy==1.17.3
- scikit-learn==0.21.3

##  Run the code

Due to data sensitivity, we only provide a data demo in tfrecord format, but the model code can be well applied to the same scenario.

```
$ cd src
$ python main.py
```

# Replicate
- tensorflow 1.14
  - mac m1 install python3.6.9: https://stackoverflow.com/questions/70205633/cannot-install-python-3-7-on-osx-arm64
  ```
  conda config --env --set subdir osx-64
  ```
- change format
  ```
    receiver_tensors = {
      'user_id': tf.placeholder(tf.int64, [None, 1], name='user_id'),
      'behavior_poi_id_list': tf.placeholder(tf.int64, [None, 10], name='behavior_poi_id_list'),
      'ad_id_list': tf.placeholder(tf.int32, [None, 5], name='ad_id_list'),
      'oi_id_list': tf.placeholder(tf.int32, [None, 5], name='oi_id_list'),
      'context_id': tf.placeholder(tf.int32, [None, 1], name='context_id'),
      'action': tf.placeholder(tf.int32, [None, 5], name='action')
      }
  from tf.float to tf.int32 
  ```




