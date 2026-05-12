import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

DATA_DIR = 'new_dataset'
CLASSES = ['46', '47', '48', '49', '50']
IMG_SIZE = 64

X, y = [], []

print("กำลังโหลดรูปภาพ...")
for idx, class_name in enumerate(CLASSES):
    path = os.path.join(DATA_DIR, class_name)
    for img_name in os.listdir(path):
        img = cv2.imread(os.path.join(path, img_name), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            # แปลงเป็นขาวดำและ Invert สี (พื้นดำ ตัวขาว)
            _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
            X.append(thresh)
            y.append(idx)

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1) / 255.0
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# สร้างโมเดล
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(CLASSES), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print("เริ่มเทรนโมเดล...")
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

model.save('best_model.keras')
print("บันทึกโมเดลเรียบร้อย (best_model.keras)")