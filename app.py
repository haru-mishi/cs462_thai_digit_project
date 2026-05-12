from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
import cv2
import base64

app = Flask(__name__)
model = tf.keras.models.load_model('best_model.keras')
CLASSES = ['46', '47', '48', '49', '50']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        image_data = data['image'].split(',')[1]
        img_bytes = base64.b64decode(image_data)
        
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        
        # --- [จุดสำคัญที่เพิ่มมา] กลับสีภาพให้เป็น "พื้นดำ ลายเส้นขาว" ---
        _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
        
        # หาขอบเขตและตัดขอบดำทิ้ง
        coords = cv2.findNonZero(img) 
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            img = img[y:y+h, x:x+w] 
            
            # เติมขอบดำรอบๆ (Padding)
            size = max(w, h)
            pad_y = (size - h) // 2 + 15
            pad_x = (size - w) // 2 + 15
            img = cv2.copyMakeBorder(img, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_CONSTANT, value=0)
            
        # ปรับภาพให้เป็น 64x64 ให้ตรงกับที่ AI เรียนมา
        img = cv2.resize(img, (64, 64))
        img = img.reshape(1, 64, 64, 1) / 255.0
        
        # ทายผล
        prediction = model.predict(img)[0]
        class_idx = np.argmax(prediction)
        
        return jsonify({
            'class_idx': int(class_idx),
            'confidence': float(prediction[class_idx]),
            'probs': [float(p) for p in prediction]
        })
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)