from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
from inference import analyze_frame

app = Flask(__name__)
CORS(app)

@app.route('/proctor', methods=['POST'])
def proctor_exam():
    try:
        data = request.json
        if 'image' not in data:
            return jsonify({"error": "No image provided"}), 400
            
        # 2. The webcam sends a Base64 string. We strip the header and decode it into raw bytes
        base64_string = data['image'].split(',')[1]
        image_bytes = base64.b64decode(base64_string)
        
        status = analyze_frame(image_bytes)
        
        return jsonify({"status": status})
        
    except Exception as e:
        print(f"Server Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("🚀 Proctoring AI Server Running on Port 5000...")
    app.run(port=5000, debug=True)