# app.py
from flask import Flask, request, jsonify
import cv2
import numpy as np
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# User model to store preferences
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50))
    preferences = db.Column(db.String(200))

# Facial analysis function
def analyze_face(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    results = []
    for (x, y, w, h) in faces:
        results.append({
            'position': {'x': int(x), 'y': int(y), 'width': int(w), 'height': int(h)},
            'suggestions': 'Style recommendation for face shape.'
        })
    return results if results else {'error': 'No face detected'}

# Recommendation function based on preferences
def get_recommendations(preferences):
    recommendations = {
        "fashion": ["Blazer", "T-shirt", "Jeans"],
        "skincare": ["Cleanser", "Moisturizer", "Sunscreen"]
    }
    return recommendations.get(preferences, [])

@app.route('/analyze', methods=['POST'])
def analyze():
    image_file = request.files.get('image')
    if image_file:
        image = np.frombuffer(image_file.read(), np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        analysis_result = analyze_face(image)
        return jsonify(analysis_result)
    return jsonify({'error': 'No image provided'}), 400

@app.route('/recommend', methods=['GET'])
def recommend():
    user_id = request.args.get('user_id')
    user = User.query.get(user_id)
    if user:
        recommendations = get_recommendations(user.preferences)
        return jsonify(recommendations)
    return jsonify({'error': 'User not found'}), 404

if __name__ == '__main__':
    db.create_all()  # Initialize the database
    app.run(host='0.0.0.0', port=5000)
  
