from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)
model = tf.keras.models.load_model('misc\model\EfficientNetB0-525-(224 X 224)- 98.97.h5')
loaded_model.compile(Adamax(learning_rate= 0.001), loss= 'categorical_crossentropy', metrics= ['accuracy'])

@app.before_first_request
def load_classes():
    global class_labels
    with open('classes.json', 'r') as f:
        class_indices = json.load(f)
        class_labels = {v: k for k, v in class_indices.items()}

load_classes()

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "Aucun fichier envoyé", 400

    file = request.files['file']
    if not file:
        return "Aucun fichier envoyé", 400

    image = Image.open(io.BytesIO(file.read()))
    img = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    prediction = model.predict(img_array)
    score = tf.nn.softmax(prediction[0])
    predicted_class = class_labels[np.argmax(score)]

    return jsonify({'prediction': predicted_class})
