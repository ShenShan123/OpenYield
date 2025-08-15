import pandas as pd
import pickle
import os
import time
import numpy as np
import torch
import random
from  prettytable import PrettyTable
 
#将一个 Python 对象以二进制形式保存到指定的文件（存取操作）
def save_obj(obj,fname):
    with open(fname, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)

def seed_set(seed=42):
    """
        fix the random seed（固定随机数种子）
    """
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)#设置 Python 的哈希种子，确保哈希值的一致性。这对于一些依赖哈希值的操作（如字典的遍历顺序）是必要的
    np.random.seed(seed)
    torch.manual_seed(seed)#设置 PyTorch 的 CPU 随机数种子，确保在 CPU 上运行的 PyTorch 代码生成的随机数序列是可重复的
    torch.cuda.manual_seed(seed)#设置 PyTorch 的当前 GPU 随机数种子，确保在当前 GPU 上运行的 PyTorch 代码生成的随机数序列是可重复的
    torch.cuda.manual_seed_all(seed)   #设置 PyTorch 的所有 GPU 随机数种子，确保在所有 GPU 上运行的 PyTorch 代码生成的随机数序列是可重复的
    torch.backends.cudnn.deterministic = True    #将 CuDNN 的确定性模式设置为 True，确保 CuDNN 卷积操作的结果是可重复的。
    torch.backends.cudnn.benchmark = False  
    torch.backends.cudnn.enabled = True     #启用 CuDNN 库，以提高 PyTorch 在 GPU 上的计算性能。


#将数据保存为 CSV 文件
def write_data2csv(tgt_dir, tgt_name, head_info, data_info):
    """
        save csv to current dir, with no new dir created.
        :param tgt_dir: dir of target file
        :type: string, need '/' as an end

        :param tgt_name: name of target filex
        :type: string, contains suffix

        :param head_info: eg, ('iter_n', 'tr_loss', 'te_loss')
        :type: tuple

        :param data_info: eg,
        [[1, 2, 3], [0.92, 0.32, 0,43], [1.23, 23.1, 2.32]]
        :type : list of every column data
    """

    if not os.path.exists(tgt_dir):
        os.makedirs(tgt_dir)
    data_frame = pd.DataFrame(columns=head_info)

    # pandas create dataframe
    for col, col_name in enumerate(head_info):
        data_frame[col_name] = np.asarray(data_info[col])
    # save txt with dataframe
    csv_path = os.path.join(tgt_dir, tgt_name)
    if os.path.exists(csv_path):
        data_frame.to_csv(csv_path, mode='a', encoding='utf-8', index=False, header=None)
    else:
        data_frame.to_csv(csv_path, mode='a', encoding='utf-8', index=False)
    # print(f"{tgt_name} saved to dir--{tgt_dir}, successfully.")

#所属类对象的字符串表示中提取出类名并返回
def get_model_class_name(instance):
    """
        get class name of instance given
        :param instance: an instance of certain Model Class
        :return: name of Class specified
    """
    return str(type(instance)).split('.')[-1][:-2]


#将一组模型的评估指标以表格的形式打印到终端窗口中
def print_metrics_v2(metrics, metric_names):
    """
        adapted from emukit code
        print the result in the terminal window as table.
        :param metrics:
        :type: a dict,
         eg,
          di = {'LAR': {'r2': 0.92,
                        'rmse': 0.32,
                        'mnll': 0.42},
                'NAR': {'r2': 0.76,
                        'rmse': 0.23,
                        'mnll': 0.32}}
        :param metric_names: specify all scores
        :type: a list,
         eg, ['r2', 'rmse', 'mnll']
    """
    model_names = metrics.keys()
    table = PrettyTable(['model'] + metric_names)

    for model_name in model_names:
        info = [model_name]
        for metric_name in metric_names:
            info.append(metrics[model_name][metric_name])
        table.add_row(info)

    print(table)

if __name__ == "__main__":
    metrics = {'MC': {'Pfail': 3.032630836157212e-05, 'Num': 14000, 'Speedup': '1.0x', 'Error': '0.0%', 'Success': 'Y'}, 'MNIS': {'Pfail': 3.032630836157212e-05, 'Num': 14000, 'Speedup': '1.0x', 'Error': '0.0%', 'Success': 'Y'}, 'AIS': {'Pfail': 3.032630836157212e-05, 'Num': 14000, 'Speedup': '1.0x', 'Error': '0.0%', 'Success': 'Y'}, 'HSCS': {'Pfail': 3.032630836157212e-05, 'Num': 14000, 'Speedup': '1.0x', 'Error': '0.0%', 'Success': 'Y'}}
    name = ['Pfail','Num','Speedup','Error','Success']
    print_metrics_v2(metrics, name)