import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from pathlib import Path
import random
import numpy
import keras

# 获取当前 config.py 所在的绝对目录 (项目根目录)
BASE_DIR = Path(__file__).resolve().parent

# 1. 音频参数
SR = 44100               #采样率
DURATION = 0.5
OVERLAP = 0.5            # 50% 的重叠率 (滑动窗口切片)
N_FFT = 2048             # FFT窗口大小
HOP_LENGTH = 1024        # 滑动步长

# 2. 训练参数
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
steel_lable = 1
alumi_lable = 0
random_state = False#种子是否随机生成，默认为False

# 3. 路径配置 (使用 pathlib 拼接路径)
ORIGIN_DATA_DIR = BASE_DIR / "data_origin"   # 原始训练数据文件夹
TRAIN_DATA_DIR = BASE_DIR/"data_train"       # 处理后训练数据文件夹
TEST_DIR = BASE_DIR / "data_test"            # 盲测数据文件夹
MODEL_SAVE_DIR = BASE_DIR / "model"          # 模型保存文件夹
MODEL_NAME = "metal_classifier.keras"
GATHER_DATA_DIR = ORIGIN_DATA_DIR            # record程序保存文件夹，默认为ORIGIN_DATA_DIR，如果你想为盲测收集数据，将此项改为TRAIN_DATA_DIR

def set_seed_42(random_state,seed=42):
    """
    设定随机数种子为42
    """
    if (random_state == False):
        random.seed(seed)                     
        numpy.random.seed(seed)               # NumPy
        keras.utils.set_random_seed(seed)     # Keras
        print("0.已固定随机数种子为42")
    else:
        print("0.随机数种子未固定")

def init_project_dirs():
    """
    初始化项目所需的文件夹
    """
    # parents=True 相当于 mkdir -p，exist_ok=True 表示存在也不报错
    ORIGIN_DATA_DIR.mkdir(parents=True, exist_ok=True)
    TEST_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    print("项目目录初始化完成。")

# 如果有人直接运行 python config.py，则执行创建目录
if __name__ == "__main__":
    init_project_dirs()