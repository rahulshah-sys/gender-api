from flask import Flask, request, jsonify
from deepface import DeepFace
import base64
import os

app = Flask(__name__)

@app.route('/analyze', methods=['POST'])
def analyze_gender():
    try:
        data = request.json
        base64_string = data.get('image', '')
        if not base64_string:
            return jsonify({"error": "No image provided"}), 400

        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]

        img_data = base64.b64decode(base64_string)
        img_path = "temp_image.jpg"
        
        with open(img_path, "wb") as f:
            f.write(img_data)

        analysis = DeepFace.analyze(img_path=img_path, actions=['gender'], enforce_detection=False)
        dominant_gender = analysis[0]['dominant_gender'] if isinstance(analysis, list) else analysis['dominant_gender']

        if os.path.exists(img_path):
            os.remove(img_path)

        return jsonify({"gender": dominant_gender}), 200
    except Exception as e:
        if os.path.exists("temp_image.jpg"):
            os.remove("temp_image.jpg")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
