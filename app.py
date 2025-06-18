from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Allow requests from frontend (port 8000)

# Load the trained model
model = tf.keras.models.load_model('KRITINET_MODEL.h5')
print("✅ Model loaded successfully!")

# Define class labels
class_labels = ['Blast', 'Brownspot', 'healthy', 'BacterialBlight']

# Home route for testing
@app.route('/')
def home():
    return "✅ KRITINET Flask API is running!"

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']

    try:
        # Load and preprocess the image
        image = Image.open(io.BytesIO(file.read())).convert('RGB')
        image = image.resize((224, 224))
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        predictions = model.predict(img_array)[0]
        predicted_index = int(np.argmax(predictions))
        confidence = float(predictions[predicted_index])
        predicted_label = class_labels[predicted_index]

        # Return all class probabilities too
        probs = {label: float(f"{prob*100:.2f}") for label, prob in zip(class_labels, predictions)}

        return jsonify({
            'class_index': predicted_index,
            'class_label': predicted_label,
            'confidence': round(confidence * 100, 2),
            'all_confidences': probs
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    import socket
    ip = socket.gethostbyname(socket.gethostname())
    print("✅ Server is starting...")
    print(f"🌐 Access: http://127.0.0.1:5000 or http://{ip}:5000 (LAN)")
    app.run(debug=True, host='0.0.0.0', port=5000)
