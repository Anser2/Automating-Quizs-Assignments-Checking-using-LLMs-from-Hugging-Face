from flask import Flask, render_template, request, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
import os
from werkzeug.utils import secure_filename
import google.generativeai as genai
from PIL import Image

app = Flask(__name__)
GOOGLE_API_KEY = 'AIzaSyAhHD0-ikdD475FnBlXjQ9Nhj6-xcMNLoo'
genai.configure(api_key=GOOGLE_API_KEY)

# Configure the database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///files.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'uploads'

# Ensure the upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

db = SQLAlchemy(app)

class File(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(150), nullable=False)
    data = db.Column(db.LargeBinary, nullable=False)

with app.app_context():
    db.create_all()

# model_name = input("Enter the model name: e.g gemini-1.5-pro-latest")
model = genai.GenerativeModel("gemini-1.5-pro-latest")
prompt ="""You are provided with a set of questions and answers. Your task is to grade the answers with a focus on the following:

1. *Subject Relevance*: For each subject (e.g., grammar, physics, etc.), the grading criteria will differ slightly:
   - *For English or Grammar quizzes: Focus on **spelling, **grammar, and **sentence structure*. Make sure the answer is clear and well-formed.
   - *For Science or Physics quizzes: Focus on **concept accuracy, **relevance of explanation, and whether the **fundamental principles* of the subject are properly addressed.
   
2. *Grading Criteria*:
   - *Accuracy*: Does the answer correctly address the question, with factual or conceptual correctness?
   - *Completeness*: Is the answer fully developed? Does it include all necessary components (e.g., calculations for physics, complete sentences for English)?
   - *Clarity*: Is the answer clear, concise, and logically structured? Is the spelling and grammar correct for English quizzes? Are the scientific explanations easily understandable for physics quizzes?

3. *Scoring*:
   - Provide a score out of *10* based on the quality of the response, considering the subject-specific guidelines above.
   - Keep the feedback brief and focused. *Point out any mistakes* without overwhelming the response with text.
   - Provide the total score at the end."""

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Process the image with the OCR model
            image = Image.open(filepath)
            # Assuming the model has a method to process the image and return the response
            response = grade_text(image, prompt)

            return render_template('index.html', result=response)
    return render_template('index.html')

def grade_text(image, prompt):
    # Convert image to a format suitable for the model if needed
    # For example, convert to bytes or base64
    image_bytes = image.tobytes()
    response = model.generate_content([prompt, image])
    
    # Extract the text content from the response
    if response and response.candidates:
        graded_text = response.candidates[0].content.parts[0].text
    else:
        graded_text = "No response received from the model."
    
    return graded_text

if __name__ == '__main__':
    app.run(debug=True)