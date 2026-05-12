import os
import cv2
import numpy as np
import random

# โฟลเดอร์ต้นทางและปลายทาง
SRC_DIR = 'dataset'
OUT_DIR = 'new_dataset'
# กำหนดคู่ตัวเลขที่ต้องการสร้าง
TARGETS = {
    '46': ('4', '6'), '47': ('4', '7'), '48': ('4', '8'), 
    '49': ('4', '9'), '50': ('5', '0')
}
SAMPLES_PER_CLASS = 500 # สร้างคลาสละ 500 รูป

os.makedirs(OUT_DIR, exist_ok=True)

for target, (digit1, digit2) in TARGETS.items():
    print(f"กำลังสร้างรูปภาพคลาส {target}...")
    os.makedirs(os.path.join(OUT_DIR, target), exist_ok=True)
    
    # อ่านไฟล์ทั้งหมดในโฟลเดอร์เลขนั้นๆ
    files1 = os.listdir(os.path.join(SRC_DIR, digit1))
    files2 = os.listdir(os.path.join(SRC_DIR, digit2))
    
    for i in range(SAMPLES_PER_CLASS):
        # สุ่มหยิบรูปมาอย่างละ 1 รูป
        img1_path = os.path.join(SRC_DIR, digit1, random.choice(files1))
        img2_path = os.path.join(SRC_DIR, digit2, random.choice(files2))
        
        img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
        
        if img1 is None or img2 is None: continue
        
        # ปรับขนาดให้สูงเท่ากันก่อนนำมาต่อ (สมมติว่า 28x28)
        img1 = cv2.resize(img1, (28, 28))
        img2 = cv2.resize(img2, (28, 28))
        
        # นำรูปมาต่อกันแนวนอน (Horizontal Concatenation) -> ขนาดจะเป็น 56(กว้าง) x 28(สูง)
        combined = cv2.hconcat([img1, img2])
        
        # ปรับขนาดสุดท้ายให้เป็นสี่เหลี่ยมจัตุรัส 64x64 เพื่อให้เข้า CNN ได้ง่าย
        final_img = cv2.resize(combined, (64, 64))
        
        # เซฟรูปลงโฟลเดอร์ new_dataset
        cv2.imwrite(os.path.join(OUT_DIR, target, f"{target}_{i}.png"), final_img)

print("สร้าง Dataset ใหม่เสร็จสิ้น! เช็คดูที่โฟลเดอร์ new_dataset ได้เลย")