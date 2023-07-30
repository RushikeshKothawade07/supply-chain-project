from flask import Flask, render_template, request, jsonify
import requests

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    predicted_class = None
    probabilities = None

    if request.method == 'POST':
        title = request.form['title']
        paragraph = request.form['paragraph']

        # Call the FastAPI endpoint to get predictions
        url = "http://127.0.0.1:8000/predict/"
        data = {"title": title, "paragraph": paragraph}
        response = requests.post(url, json=data)

        if response.status_code == 200:
            result = response.json()
            predicted_class = result["predicted_class"]
            probabilities = result["probabilities"]

    return render_template('index.html', predicted_class=predicted_class, probabilities=probabilities)

if __name__ == '__main__':
    app.run(debug=True)
