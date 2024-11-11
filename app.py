from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from transformers import AutoModel, AutoTokenizer
import torch
import os
import requests

app = Flask(__name__)

# Configure the upload folder
app.config['UPLOAD_FOLDER'] = 'uploads'

# Initialize the OCR model (ucaslcl/GOT-OCR2_0)
tokenizer = AutoTokenizer.from_pretrained('ucaslcl/GOT-OCR2_0', trust_remote_code=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AutoModel.from_pretrained(
    'ucaslcl/GOT-OCR2_0',
    trust_remote_code=True,
    low_cpu_mem_usage=True,
    use_safetensors=True,
    pad_token_id=tokenizer.eos_token_id
).to(device)
model = model.eval()

# NVIDIA OpenAI API for grading (replace with actual NVIDIA API endpoint if needed)
def grade_answers_with_nvidia(res_text):
    api_url = "https://integrate.api.nvidia.com/v1/chat/completions"  # Correct NVIDIA API URL
    api_key = "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"  # Replace with your valid NVIDIA API key
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }
    prompt_qa = f"""
You are provided with a set of questions and answers. Your task is to grade the answers with a focus on the following:

1. **Subject Relevance**: For each subject (e.g., grammar, physics, etc.), the grading criteria will differ slightly:
   - **For English or Grammar quizzes**: Focus on **spelling**, **grammar**, and **sentence structure**. Make sure the answer is clear and well-formed.
   - **For Science or Physics quizzes**: Focus on **concept accuracy**, **relevance of explanation**, and whether the **fundamental principles** of the subject are properly addressed.
   
2. **Grading Criteria**:
   - **Accuracy**: Does the answer correctly address the question, with factual or conceptual correctness?
   - **Completeness**: Is the answer fully developed? Does it include all necessary components (e.g., calculations for physics, complete sentences for English)?
   - **Clarity**: Is the answer clear, concise, and logically structured? Is the spelling and grammar correct for English quizzes? Are the scientific explanations easily understandable for physics quizzes?

3. **Scoring**:
   - Provide a score out of **10** based on the quality of the response, considering the subject-specific guidelines above.
   - Keep the feedback brief and focused. **Point out any mistakes** without overwhelming the response with text.
   - Provide the total score at the end.

**Here is the text to be graded**:
{res_text}

    """
    data = {
        "model": "nvidia/mistral-nemo-minitron-8b-8k-instruct",  # Use correct model name here
        "messages": [{"role": "user", "content": prompt_qa}],
        "temperature": 0.5,
        "max_tokens": 1024
    }

    response = requests.post(api_url, headers=headers, json=data)
    if response.status_code == 200:
        graded_text = response.json().get('choices', [{}])[0].get('message', {}).get('content', 'No response')
        return graded_text
    else:
        return f"Error: {response.status_code}"

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    result = None
    graded_result = None
    if request.method == 'POST':
        action = request.form.get('action')
        if 'file' not in request.files:
            print("No file part in request.files")  # Debug print
            return "No file part", 400
        file = request.files['file']
        if file.filename == '':
            print("No selected file")  # Debug print
            return "No selected file", 400
        if file:
            # Save the file securely
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Process the image with the OCR model
            image_file = filepath
            res = model.chat(tokenizer, image_file, ocr_type='ocr')
            result = res  # Store OCR result

            # If the "Check" button is clicked, grade the response
            if action == 'check':
                graded_result = grade_answers_with_nvidia(result)  # Grade using NVIDIA API

    return render_template('index.html', result=result, graded_result=graded_result)

if __name__ == '__main__':
    app.run(debug=True)