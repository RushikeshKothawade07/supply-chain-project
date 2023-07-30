# Supply Chain Logistics Multi-class Classification (FastAPI and Flask)

![Python](https://img.shields.io/badge/python-3.9-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.68.1-green.svg)
![Flask](https://img.shields.io/badge/Flask-2.0.1-yellow.svg)
![Docker](https://img.shields.io/badge/Docker-v2.x-blue.svg)

## Description
This project is a Supply Chain Logistics Multi-class Classification web application built using FastAPI and Flask. It uses a pre-trained DistilBERT model for text classification into various classes related to supply chain logistics. The API is built using FastAPI, and the front-end web interface is created with Flask.

## Features
- Text classification for supply chain logistics using DistilBERT
- User-friendly web interface to interact with the model
- Docker containerization for easy deployment

## Requirements
- Python 3.9 or higher
- Docker (optional, for containerization)

## Installation and Setup
1. Clone this repository to your local machine.
2. Install the required Python packages using the following command:

```
pip install -r requirements.txt
```

3. To run the FastAPI server, execute the following command:
```
uvicorn ml_api:app --host 0.0.0.0 --port 8000
```

4. To run the Flask web app, execute the following command:
```
python flaskapp.py
```


 ## Usage
1. Open your web browser and navigate to http://localhost:5000.
2. Enter the title and paragraph text, and click the "PREDICT" button.
3. The predicted class and probabilities will be displayed on the web page.

## Docker Deployment (Optional)

If you prefer to run the application in a Docker container, follow these steps:

1. Build the Docker image:

```
docker build -t my_fastapi_app:latest .
```

2. Run the Docker container:

```
docker run -d -p 5000:5000 my_fastapi_app:latest
```

3. Access the web app in your browser at http://localhost:5000.

## Some Snapshots of the project : 

![Basic EDA](https://github.com/RushikeshKothawade07/supply-chain-project/blob/main/screenshots/eda.png)

![Textual Data Analysis](https://github.com/RushikeshKothawade07/supply-chain-project/blob/main/screenshots/eda-words.png)

![Flask Web App Home Page](https://github.com/RushikeshKothawade07/supply-chain-project/blob/main/screenshots/home-input.jpg)

![Output predictions](https://github.com/RushikeshKothawade07/supply-chain-project/blob/main/screenshots/output.jpg)

![FastAPI](https://github.com/RushikeshKothawade07/supply-chain-project/blob/main/screenshots/fastapi.jpg)

![Postman JSON queries](https://github.com/RushikeshKothawade07/supply-chain-project/blob/main/screenshots/postman-json.jpg)

![Docker Deployment](https://github.com/RushikeshKothawade07/supply-chain-project/blob/main/screenshots/docker.jpg)



## Contributing 

Contributions are welcome! If you find any issues or want to add new features, please open an issue or submit a pull request.



