from pathlib import Path
import librosa
import numpy as np
from sklearn.model_selection import train_test_split

from config import SR, DURATION, OVERLAP, N_FFT, HOP_LENGTH,steel_lable,alumi_lable

TARGET_SAMPLES = int(SR * DURATION)          # 每个切片的采样点数
STEP_SAMPLES = int(TARGET_SAMPLES * (1 - OVERLAP)) # 每次滑动的步长

def get_audio_input_shape():
    """计算特征形状"""
    total_samples = int(SR * DURATION)
    time_frames = (total_samples // HOP_LENGTH) + 1
    freq_bins = (N_FFT // 2) + 1
    return (time_frames, freq_bins, 1)

def compute_stft_features(audio_path):
    """
    读取音频并使用滑动窗口计算 STFT
    """
    try:
        y, sr = librosa.load(audio_path, sr=SR)
    except Exception as e:
        print(f"读取文件出错 {audio_path}: {e}")
        return []
    
    features = []
    
    # 使用带有步长的滑动窗口来切片 (Data Augmentation)
    for i in range(0, len(y) - TARGET_SAMPLES + 1, STEP_SAMPLES):
        chunk = y[i : i + TARGET_SAMPLES]
        
        # 计算 STFT
        D = librosa.stft(chunk, n_fft=N_FFT, hop_length=HOP_LENGTH)
        
        # 获取振幅谱 (取绝对值)
        amplitude_spectrogram = np.abs(D)
        
        # 转换为分贝值 
        log_spectrogram = librosa.amplitude_to_db(amplitude_spectrogram, ref=np.max)
        
        features.append(log_spectrogram.T)
        
    return features

def process_file_list(file_list, label):
    """
    辅助函数：把给定的文件列表转换成特征矩阵
    """
    X_temp = []
    y_temp = []
    for path in file_list:
        feats = compute_stft_features(path)
        X_temp.extend(feats)
        y_temp.extend([label] * len(feats))
    return X_temp, y_temp

def build_dataset(steel_dir, alumi_dir):
    """
    无数据泄露的数据集构建：按文件进行打乱和划分
    """
    # 1. 获取所有文件的完整路径
    steel_files = list(steel_dir.glob("*.wav"))
    alumi_files = list(alumi_dir.glob("*.wav"))
    
    # 2. 按“文件列表”进行 8:2 划分
    steel_train, steel_test = train_test_split(steel_files, test_size=0.2, random_state=42)
    alumi_train, alumi_test = train_test_split(alumi_files, test_size=0.2, random_state=42)
    
    print(f"分配完毕：钢球训练集 {len(steel_train)}个文件, 测试集 {len(steel_test)}个文件")
    print(f"分配完毕：铝球训练集 {len(alumi_train)}个文件, 测试集 {len(alumi_test)}个文件")

    # 3. 分别从训练文件和测试文件中提取特征 (切片)
    print("正在提取训练集特征...")
    X_train_steel, y_train_steel = process_file_list(steel_train, steel_lable)
    X_train_alumi, y_train_alumi = process_file_list(alumi_train, alumi_lable)
    
    print("正在提取测试集特征...")
    X_test_steel, y_test_steel = process_file_list(steel_test, steel_lable)
    X_test_alumi, y_test_alumi = process_file_list(alumi_test, alumi_lable)
    
    # 4. 合并数据
    X_train = np.array(X_train_steel + X_train_alumi)[..., np.newaxis]
    y_train = np.array(y_train_steel + y_train_alumi)
    
    X_test = np.array(X_test_steel + X_test_alumi)[..., np.newaxis]
    y_test = np.array(y_test_steel + y_test_alumi)
    
    return X_train, y_train, X_test, y_test


