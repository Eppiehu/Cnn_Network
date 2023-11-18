from cnn_Network import model
import numpy as np
from keras.utils import to_categorical
import pandas as pd
from PIL import Image
import numpy as np
#import os
#print("Current Working Directory:", os.getcwd())


# 读取 CSV 文件
data = pd.read_csv('./test.csv')

# 检查数据
print(data.head(8))

def load_and_preprocess_image(path, target_size=(444, 200)):
    # 使用 Pillow 打开图像
    image = Image.open(path)

    # 调整图像尺寸
    image = image.resize(target_size)

    # 将图像转换为灰度（如果需要）
    image = image.convert('L')

    # 将图像转换为 NumPy 数组并归一化
    image = np.array(image)
    image = image.astype('float32') / 255.0

    # 如果需要，增加一个通道维度
    image = np.expand_dims(image, axis=-1)

    return image

# 应用于所有图像
X = np.array([load_and_preprocess_image(img_path) for img_path in data['image_path']])
y = np.array(data['label'])
y = to_categorical(y)#soft函数

X_train,y_train= (X, y)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=27, batch_size=8)

image_path = './test/test1.jpg'  # 替换为您的图像路径
image_to_predict = load_and_preprocess_image(image_path)

# 将图像扩展为一个批次
image_batch = np.expand_dims(image_to_predict, axis=0)

# 使用模型进行预测
prediction = model.predict(image_batch)

# 输出预测结果
print(prediction)

predicted_class_idx = np.argmax(prediction, axis=1)[0]  # 获取数组中的第一个元素

if predicted_class_idx == 0:
    predicted_class_name = '瓶子'
elif predicted_class_idx == 1:
    predicted_class_name = '笔'
else:
    predicted_class_name = '未知'

print("Predicted class:", predicted_class_name)
