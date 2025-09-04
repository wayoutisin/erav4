
import os
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
from google.cloud import vision
from google.oauth2 import service_account
import google.generativeai as genai
import json
import traceback # Import traceback for detailed error logging

print("*** app.py has started and is loading... ***")

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB limit

# Ensure the upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Removed genai.configure from global scope

def detect_text(image_path):
    """Detects text in the image using the Google Vision API with an API Key."""
    api_key = os.environ.get('GOOGLE_VISION_API_KEY')
    if not api_key:
        raise ValueError("GOOGLE_VISION_API_KEY environment variable not set.")

    client = vision.ImageAnnotatorClient(client_options={"api_key": api_key})

    with open(image_path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)
    response = client.text_detection(image=image)
    texts = response.text_annotations

    if texts:
        return texts[0].description
    return "No text detected."


def get_health_summary(content_text):
    """Gets a health summary from Gemini API based on food item content."""
    if not content_text or content_text == "No text detected.":
        return {"positives": ["N/A"], "negatives": ["N/A"], "indicator": "grey", "raw_response": "No content to analyze."}

    gemini_api_key = os.environ.get('GEMINI_API_KEY')
    if not gemini_api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set.")

    genai.configure(api_key=gemini_api_key)

    # --- DEBUGGING: List available models --- START
    # print("--- Available Gemini Models (for generateContent) ---")
    # for m in genai.list_models():
    #     if "generateContent" in m.supported_generation_methods:
    #         print(f"  - {m.name}")
    # print("---------------------------------------------------")
    # --- DEBUGGING: List available models --- END

    model = genai.GenerativeModel('models/gemini-1.5-flash') # Updated model name
    prompt = f"""You are a health expert, who is looking at the content of a food item provided to you as input TEXT: `{content_text}`. Please suggest upto 3 positives and 3 negatives about the content based on the ingredient information shared. Also provide an indicator about red, amber, green depending on the harmfulness of the content. Respond only with a JSON object in the format: {{"positives": ["positive1", "positive2"], "negatives": ["negative1", "negative2"], "indicator": "red/amber/green"}}. Do not include any other text or markdown outside the JSON.
"""

    try:
        response = model.generate_content(prompt)
        # Assuming Gemini responds with a text that can be parsed as JSON
        response_text = response.text
        print(f"Gemini Raw Response: {response_text}") # Added for debugging

        # More robust method to extract JSON content
        json_start = response_text.find('{')
        json_end = response_text.rfind('}')

        if json_start != -1 and json_end != -1 and json_end > json_start:
            response_text = response_text[json_start : json_end + 1]
        else:
            # If JSON structure is not found, handle as a parsing error
            raise json.JSONDecodeError("Could not find valid JSON object in Gemini response.", response_text, 0)

        summary_data = json.loads(response_text)
        return summary_data
    except json.JSONDecodeError as e:
        print(f"JSON Decode Error from Gemini response: {e}")
        print(f"Problematic Gemini Raw Response: {response_text}")
        return {"positives": ["Parsing Error"], "negatives": ["Parsing Error"], "indicator": "grey", "raw_response": response_text}
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        traceback.print_exc() # Print the full traceback
        return {"positives": ["API Error"], "negatives": ["API Error"], "indicator": "grey", "raw_response": str(e)}


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        return redirect(url_for('uploaded_file', filename=filename))

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    extracted_text = detect_text(filepath)
    
    health_summary = get_health_summary(extracted_text)

    return render_template('upload_success.html', filename=filename, extracted_text=extracted_text, health_summary=health_summary)

@app.route('/uploaded_file_serve/<filename>')
def uploaded_file_serve(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
