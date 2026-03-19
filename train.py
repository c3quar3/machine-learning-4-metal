import os
os.environ["KERAS_BACKEND"] = "torch"#选择引擎
from pathlib import Path
import keras
import numpy as np
import matplotlib.pyplot as plt
import config as cfg
from config import random_state
from config import set_seed_42
from model import build_cnn_model

def main():
    print("1. 正在加载数据...")
    X_train = np.load(Path(cfg.TRAIN_DATA_DIR/ "X_train.npy"))
    y_train = np.load(Path(cfg.TRAIN_DATA_DIR/ "Y_train.npy"))
    X_test = np.load(Path(cfg.TRAIN_DATA_DIR/ "X_test.npy"))
    y_test = np.load(Path(cfg.TRAIN_DATA_DIR/ "Y_test.npy"))
    
    print("2. 正在进行标准化 ")
    # 计算均值和标准差
    mean_val = np.mean(X_train)
    std_val = np.std(X_train)
    
    # 用训练集的参数去归一化训练集和测试集
    X_train_norm = (X_train - mean_val) / (std_val + 1e-7)
    X_test_norm = (X_test - mean_val) / (std_val + 1e-7)

    #转换为32位浮点数，方便GPU运算
    X_train_norm = X_train_norm.astype(np.float32)
    X_test_norm = X_test_norm.astype(np.float32)
    y_train = y_train.astype(np.float32)
    y_test = y_test.astype(np.float32)
    
    print("3. 构建并编译模型...")
    input_shape = X_train_norm.shape[1:] 
    model = build_cnn_model(input_shape)
    
    # 在训练脚本中进行 compile (符合模型只管结构，训练管策略的原则)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001), 
        loss='binary_crossentropy', 
        metrics=['accuracy']
    )
    model.summary()
    
    print(" 4. 开始训练...")
    # 设置早停机制
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=5, 
        restore_best_weights=True
    )
    
    history = model.fit(
        X_train_norm, y_train,
        epochs=cfg.EPOCHS, 
        batch_size=cfg.BATCH_SIZE,
        shuffle=True, 
        validation_data=(X_test_norm, y_test),
        callbacks=[early_stopping],
        verbose=1 
    )
    print("5. 保存模型与标准化参数...")
    Path.mkdir(cfg.MODEL_SAVE_DIR, exist_ok=True)
    
    # 
    model_path = Path(cfg.MODEL_SAVE_DIR/"metal_classifier.keras")
    model.save(model_path)
    
    np.save(Path(cfg.MODEL_SAVE_DIR/ "mean_val.npy"), mean_val)
    np.save(Path(cfg.MODEL_SAVE_DIR/ "std_val.npy"), std_val)
    print(f"✅训练完成！资产已保存至: {cfg.MODEL_SAVE_DIR}")
    
    print("6. 绘制训练曲线...")
    plot_history(history)
def plot_history(history):
    """提取出的绘图辅助函数"""
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss')
    plt.legend()
    
    plot_path = Path(cfg.MODEL_SAVE_DIR/ "training_history.png")
    plt.savefig(plot_path)
    print(f"曲线图已保存至: {plot_path}")
    plt.show()
if __name__ == "__main__":
    set_seed_42(random_state)
    main()