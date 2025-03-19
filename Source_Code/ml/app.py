import os
from flask import Flask, request, render_template, flash, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Initialize the Flask application
app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Needed for flash messages

# Load the trained model
model_path = 'xception_model.keras'  # Ensure the path is correct
model = load_model(model_path)

# Define the upload folder
UPLOAD_FOLDER = 'static/uploads/'  # Ensure the uploads folder is in the static directory
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Home Page Route
@app.route('/')
def home():
    return render_template('home.html')

# About Page Route
@app.route('/about')
def about():
    return render_template('about.html')

# Predictions Page Route
@app.route('/predictions')
def predictions():
    return render_template('predictions.html')

# Metrics Page Route
@app.route('/metrics')
def metrics():
    return render_template('metrics.html')

# Flowchart Page Route
@app.route('/flowchart')
def flowchart():
    return render_template('flowchart.html')

# Route for handling the image upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        flash('No file part', 'error')
        return redirect(url_for('predictions'))

    file = request.files['file']

    if file.filename == '':
        flash('No selected file', 'error')
        return redirect(url_for('predictions'))

    if file:
        try:
            # Save the file to the uploads folder
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)

            # Prepare the image for prediction
            img = image.load_img(file_path, target_size=(299, 299))
            img_array = image.img_to_array(img) / 255.0  # Normalize the image
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

            # Make prediction
            predictions = model.predict(img_array)
            predicted_class = np.argmax(predictions[0])  # Get the predicted class index

            # Define the mapping according to your requirement
            if predicted_class == 0:  # Assuming 0 is Fire
                predicted_label = 'Fire'
            else:  # Assuming 1 is No Fire
                predicted_label = 'No Fire'

            # Return the predicted class and display the image
            return render_template('result.html', predicted_label=predicted_label, image_file=file.filename)

        except Exception as e:
            flash(f'Error processing file: {str(e)}', 'error')
            return redirect(url_for('predictions'))

    flash('File upload failed', 'error')
    return redirect(url_for('predictions'))

if __name__ == '__main__':
    app.run(debug=True)
