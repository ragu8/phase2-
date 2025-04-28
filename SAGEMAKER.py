from flask import Flask, request, jsonify
import os
import requests
import json
import re
from dotenv import load_dotenv


app = Flask(__name__)

SAGEMAKER_API_KEY = os.getenv("SAGEMAKER_API_KEY")
SAGEMAKER_API_URL = os.getenv("SAGEMAKER_API_URL")

def fruit_quality_grading(image_url):
    headers = {
        "Content-Type": "application/json",
        "api-key": SAGEMAKER_API_KEY,
    }

    payload = {
        "messages": [
            {"role": "system", "content": "You are a fruit quality evaluation assistant. Analyze the given fruit image."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Analyze the fruit image and return JSON only like { \"fruit_name\": \"str\", \"quality\": \"ripe/unripe/old/damage/Complex\" }"},
                    {"type": "image_url", "image_url": {"url": image_url}}
                ]
            }
        ],
        "temperature": 0.0
    }

    response = requests.post(SAGEMAKER_API_URL, headers=headers, json=payload)

    if response.status_code == 200:
        result = response.json()
        content = result['choices'][0]['message']['content']

        content_cleaned = re.sub(r"```json|```", "", content).strip()

        parsed = json.loads(content_cleaned)
        return parsed
    else:
        return {"error": response.text}

@app.route('/evaluate', methods=['POST'])
def evaluate():
    data = request.get_json()
    if not data or 'image_url' not in data:
        return jsonify({"error": "Please provide an 'image_url' in JSON body"}), 400

    image_url = data['image_url']
    result = fruit_quality_grading(image_url)

    if isinstance(result, dict) and 'error' in result:
        return jsonify(result), 500

    return jsonify(result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
