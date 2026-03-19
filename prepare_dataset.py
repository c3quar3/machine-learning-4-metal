from pathlib import Path
import numpy as np
from data_utils import build_dataset, get_audio_input_shape
import config as cfg

if __name__ == "__main__":
    steel_folder = Path(cfg.ORIGIN_DATA_DIR/"steel")  
    alumi_folder = Path(cfg.ORIGIN_DATA_DIR/"alumi") 
    save_dir = cfg.TRAIN_DATA_DIR
    
    

    # 打印预期的模型输入尺寸
    expected_shape = get_audio_input_shape()
    print(f"预期推算出的 CNN 输入形状: {expected_shape}")
    
    if Path.exists(steel_folder) and Path.exists(alumi_folder):
        X_train, y_train, X_test, y_test = build_dataset(steel_folder, alumi_folder)
        
        if len(X_train) > 0:
            Path.mkdir(save_dir, exist_ok=True)

            np.save(Path(save_dir/ "X_train.npy"), X_train)
            np.save(Path(save_dir/ "Y_train.npy"), y_train)
            np.save(Path(save_dir/ "X_test.npy"), X_test)
            np.save(Path(save_dir/ "Y_test.npy"), y_test)
            
            print(f"\n 数据预处理完成！")
            print(f"真实生成的 X_train 形状: {X_train.shape}")
            print(f"真实生成的 X_test 形状: {X_test.shape}")
            
            # 进行一次安全校验
            if X_train.shape[1:] != expected_shape:
                print(f"警告：真实特征形状 {X_train.shape[1:]} 与预测形状 {expected_shape} 不符！")
            else:
                print("真实特征形状与推算形状一致！")
    else:
        print("错误：未找到原始音频文件夹，请先运行 record.py 采集数据。")