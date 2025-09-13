from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import PyPDF2
import re
import io
import logging
import warnings

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Suppress scikit-learn version mismatch warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# Load the model
try:
    model = joblib.load("heart_disease_model.joblib")
    logging.info("Model loaded successfully")
except FileNotFoundError:
    logging.error("heart_disease_model.joblib not found in project directory")
    model = None

def extract_parameters_from_text(text):
    """Extract 13 heart disease parameters from text using regex."""
    logging.debug(f"Extracted text from file: {text[:500]}...")  # Log first 500 chars for brevity

    parameters = {
        'age': None, 'sex': None, 'cp': None, 'trestbps': None, 'chol': None,
        'fbs': None, 'restecg': None, 'thalach': None, 'exang': None,
        'oldpeak': None, 'slope': None, 'ca': None, 'thal': None
    }

    # Regex patterns for each parameter
    patterns = {
        'age': r'(?:Age|age)[:\s=]*(\d{1,3})',
        'sex': r'(?:Sex|sex|Gender|gender)[:\s=]*(Male|Female|1|0)',
        'cp': r'(?:Chest\s*Pain\s*Type|cp|ChestPain)[:\s=]*(0|1|2|3|Typical\s*Angina|Atypical\s*Angina|Non-anginal\s*Pain|Asymptomatic)',
        'trestbps': r'(?:Resting\s*BP|trestbps|Blood\s*Pressure)[:\s=]*(\d{1,3})',
        'chol': r'(?:Cholesterol|chol)[:\s=]*(\d{1,4})',
        'fbs': r'(?:Fasting\s*Blood\s*Sugar|fbs)[:\s=]*(0|1|Yes|No|True|False)',
        'restecg': r'(?:Resting\s*ECG|restecg|ECG\s*Result|Rest\s*ECG|resting_ecg|ECG|Electrocardiogram)[:\s=]*(0|1|2|Normal|Minor\s*Irregularity|Thick\s*Heart\s*Muscle)',
        'thalach': r'(?:Max\s*Heart\s*Rate|thalach|Maximum\s*Heart\s*Rate)[:\s=]*(\d{1,3})',
        'exang': r'(?:Exercise\s*Angina|exang)[:\s=]*(0|1|Yes|No|True|False)',
        'oldpeak': r'(?:Oldpeak|oldpeak)[:\s=]*(\d*\.?\d*)',
        'slope': r'(?:Slope|slope)[:\s=]*(0|1|2|Upsloping|Flat|Downsloping)',
        'ca': r'(?:Visible\s*Heart\s*Vessels|ca)[:\s=]*(0|1|2|3)',
        'thal': r'(?:Thal|thal|Thalassemia)[:\s=]*(1|2|3|Normal|Fixed\s*Defect|Reversible\s*Defect)'
    }

    for param, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            value = match.group(1)
            logging.debug(f"Found {param}: {value}")
            if param == 'sex':
                value = 1 if value.lower() in ['male', '1'] else 0
            elif param == 'cp':
                value = {'0': 0, '1': 1, '2': 2, '3': 3, 'typical angina': 0, 'atypical angina': 1,
                         'non-anginal pain': 2, 'asymptomatic': 3}.get(value.lower(), value)
            elif param == 'fbs':
                value = 1 if value.lower() in ['yes', 'true', '1'] else 0
            elif param == 'restecg':
                value = {'0': 0, '1': 1, '2': 2, 'normal': 0, 'minor irregularity': 1,
                         'thick heart muscle': 2}.get(value.lower(), value)
            elif param == 'exang':
                value = 1 if value.lower() in ['yes', 'true', '1'] else 0
            elif param == 'slope':
                value = {'0': 0, '1': 1, '2': 2, 'upsloping': 0, 'flat': 1, 'downsloping': 2}.get(value.lower(), value)
            elif param == 'thal':
                value = {'1': 1, '2': 2, '3': 3, 'normal': 1, 'fixed defect': 2, 'reversible defect': 3}.get(value.lower(), value)
            parameters[param] = float(value) if param in ['age', 'trestbps', 'chol', 'thalach', 'oldpeak'] else int(value)
        else:
            logging.debug(f"No match for {param}")

    return parameters

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/calculator")
def calculator():
    return render_template("calculator.html")

@app.route("/doctor")
def doctor():
    return render_template("doctor.html")

@app.route("/aboutus")
def aboutus():
    return render_template("aboutus.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")

@app.route("/article1_bp")
def article1_bp():
    return render_template("article1_bp.html")

@app.route("/article2_cho")
def article2_cho():
    return render_template("article2_cho.html")

@app.route("/article3_diabetes")
def article3_diabetes():
    return render_template("article3_diabetes.html")

@app.route("/article4_habit")
def article4_habit():
    return render_template("article4_habit.html")

@app.route("/article5_stress")
def article5_stress():
    return render_template("article5_stress.html")

@app.route("/chat", methods=["POST"])
def chat():
    return jsonify({"reply": "Chatbot not implemented yet"})

@app.route("/favicon.ico")
def favicon():
    return "", 204

@app.route("/predict", methods=["POST"])
def predict():
    try:
        input_method = request.form.get('inputMethod')
        if not input_method:
            return jsonify({"prediction_text": "Error: Input method not specified."}), 400

        params = {}
        if input_method == 'manual':
            ranges = {
                'age': (0, 120),
                'sex': (0, 2),
                'cp': (0, 3),
                'trestbps': (50, 200),
                'chol': (100, 600),
                'fbs': (0, 1),
                'restecg': (0, 2),
                'thalach': (60, 220),
                'exang': (0, 1),
                'oldpeak': (0, 10),
                'slope': (0, 2),
                'ca': (0, 3),
                'thal': (1, 3)
            }

            for field in ranges:
                value = request.form.get(field)
                if value is None or value == '':
                    return jsonify({"prediction_text": f"Error: Missing parameter {field}."}), 400
                try:
                    params[field] = float(value)
                    min_val, max_val = ranges[field]
                    if not (min_val <= params[field] <= max_val):
                        return jsonify({"prediction_text": f"Error: {field} must be between {min_val} and {max_val}."}), 400
                except ValueError:
                    return jsonify({"prediction_text": f"Error: Invalid value for {field}."}), 400

            if params['sex'] == 2:
                params['sex'] = 1

        else:  # input_method == 'upload'
            if 'report' not in request.files:
                return jsonify({"prediction_text": "Error: No file uploaded."}), 400

            file = request.files['report']
            if file.filename == '':
                return jsonify({"prediction_text": "Error: No file selected."}), 400

            if not (file.filename.endswith('.pdf') or file.filename.endswith('.txt')):
                return jsonify({"prediction_text": "Error: File must be a PDF or TXT."}), 400

            text = ""
            if file.filename.endswith('.pdf'):
                try:
                    pdf_reader = PyPDF2.PdfReader(io.BytesIO(file.read()))
                    for page in pdf_reader.pages:
                        extracted_text = page.extract_text()
                        if extracted_text:
                            text += extracted_text + "\n"
                except Exception as e:
                    logging.error(f"PDF parsing error: {str(e)}")
                    return jsonify({"prediction_text": f"Error: Failed to read PDF file. {str(e)}"}), 400
            else:  # .txt
                text = file.read().decode('utf-8', errors='ignore')

            if not text.strip():
                return jsonify({"prediction_text": "Error: No readable text found in the file. Ensure the PDF is not scanned or empty."}), 400

            params = extract_parameters_from_text(text)
            missing_params = [key for key, value in params.items() if value is None]
            if missing_params:
                return jsonify({"prediction_text": f"Error: Missing parameters in report: {', '.join(missing_params)}.<br>Please ensure the report includes: Age, Sex, Chest Pain Type, Resting BP, Cholesterol, Fasting Blood Sugar, Resting ECG, Max Heart Rate, Exercise Angina, Oldpeak, Slope, Visible Heart Vessels, Thal."}), 400

        # Validate parameters
        if params['age'] <= 0:
            raise ValueError("Age must be positive")
        if params['sex'] not in [0, 1]:
            raise ValueError("Sex must be 0 (Female) or 1 (Male)")
        if params['cp'] not in [0, 1, 2, 3]:
            raise ValueError("Chest Pain Type must be between 0 and 3")
        if params['trestbps'] <= 0:
            raise ValueError("Resting BP must be positive")
        if params['chol'] <= 0:
            raise ValueError("Cholesterol must be positive")
        if params['fbs'] not in [0, 1]:
            raise ValueError("Fasting Blood Sugar must be 0 or 1")
        if params['restecg'] not in [0, 1, 2]:
            raise ValueError("Resting ECG must be 0, 1, or 2")
        if params['thalach'] <= 0:
            raise ValueError("Max Heart Rate must be positive")
        if params['exang'] not in [0, 1]:
            raise ValueError("Exercise Angina must be 0 or 1")
        if params['oldpeak'] < 0:
            raise ValueError("Oldpeak cannot be negative")
        if params['slope'] not in [0, 1, 2]:
            raise ValueError("Slope must be 0, 1, or 2")
        if params['ca'] not in [0, 1, 2, 3]:
            raise ValueError("Visible Heart Vessels must be between 0 and 3")
        if params['thal'] not in [1, 2, 3]:
            raise ValueError("Thalassemia result must be between 1 and 3")

        # Create input array
        input_data = np.array([[
            params['age'], params['sex'], params['cp'], params['trestbps'], params['chol'],
            params['fbs'], params['restecg'], params['thalach'], params['exang'],
            params['oldpeak'], params['slope'], params['ca'], params['thal']
        ]])

        # Make prediction
        if model is None:
            logging.error("Model not loaded")
            return jsonify({"prediction_text": "Error: Model not loaded."}), 500

        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]
        prob_text = f"Confidence: {max(probability) * 100:.2f}%"
        result = ("No Heart Disease" if prediction == 0 else "Heart Disease Detected") + f" ({prob_text})"
        logging.info(f"Prediction successful: {result}")

        return jsonify({"prediction_text": result}), 200

    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        return jsonify({"prediction_text": f"Error: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True)