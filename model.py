import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from keras import layers, models

def build_cnn_model(input_shape):
    """
    定义CNN神经网络骨架
    """
    model = models.Sequential([

        # 第一层卷积
        layers.Conv2D(32, (3, 3), activation='relu',padding='same',input_shape=input_shape),
        layers.MaxPooling2D((1, 4)),

        # 第二层卷积
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 4)),
        
        # 展平与全连接层
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5), # 防止过拟合
        layers.Dense(1, activation='sigmoid') # 二分类(钢球/铝球)

    ])

    return model
