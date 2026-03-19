import os
os.environ["KERAS_BACKEND"] = "torch"  # 选择引擎
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import keras
from pathlib import Path
import numpy as np

from config import TEST_DIR,MODEL_SAVE_DIR,MODEL_NAME
from data_utils import compute_stft_features 

def load_system():
    """加载模型和标准化参数"""
    print("正在初始化预测系统...")
    
    # 加载模型
    model_path = Path(MODEL_SAVE_DIR/MODEL_NAME)
    if not model_path.exists:
        raise FileNotFoundError("找不到模型文件 metal_classifier.keras！请先运行 train.py")
    model = keras.saving.load_model(model_path)
    mean_val = np.load(Path(MODEL_SAVE_DIR/"mean_val.npy"))
    std_val = np.load (Path(MODEL_SAVE_DIR/"std_val.npy"))
    
    print("系统加载完毕！")
    return model, mean_val, std_val

def predict_audio(file_path, model, mean_val, std_val):
    """对单个音频文件进行盲测"""
    try:
        # 1. 提取特征 
        feature = compute_stft_features(file_path) 
        
        # 2. 核心：使用【训练集的均值和标准差】进行标准化！
        feature_norm = (feature - mean_val) / (std_val + 1e-7)
        
        # 3. 转换数据类型为 GPU 友好的 float32
        feature_norm = feature_norm.astype(np.float32)
        
        # 4. 调整形状以匹配 CNN 输入 (Batch, Height, Width, Channels)
        # 你的单个样本是 (22, 1025)，需要变成 (1, 22, 1025, 1)
        # np.expand_dims(..., axis=-1) 增加 Channel 维度
        feature_ready = np.expand_dims(feature_norm, axis=-1)
        
        # 5. 模型推理
        prediction = model.predict(feature_ready, verbose=0)
        
        # 6. 解析结果 (假设 0 是铝球，1 是钢球，请根据你的 config.py 调整)
        probability = prediction[0][0]  # 获取二分类的概率值
        
        if probability > 0.5:
            result_label = "钢球 (Steel)"
            confidence = probability * 100
        else:
            result_label = "铝球 (Aluminum)"
            confidence = (1 - probability) * 100
            
        print(f"文件: {file_path.name}")
        print(f"   -> 预测结果: {result_label}  (置信度: {confidence:.2f}%)")
        print("-" * 40)
        
        return probability
        
    except Exception as e:
        print(f" 处理文件 {file_path} 时出错: {e}")
        return None
if __name__ == "__main__":
    # 加载环境
    model, mean_val, std_val = load_system()
    blind_test_dir = TEST_DIR #  在项目里建一个这个文件夹，把新录的音频扔进去
    if blind_test_dir.exists:
        wav_files = blind_test_dir.rglob("*.wav")
        first_file = next(wav_files, None)
        if first_file is None:
            print("盲测文件夹是空的，请放入 .wav 文件")
        else:
            for wav in wav_files:
                predict_audio(wav, model, mean_val, std_val)