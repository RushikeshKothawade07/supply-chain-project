# Supply Chain Logistics -  Multi-class Classification (FastAPI and Flask) - DistilBERT

![Python](https://img.shields.io/badge/python-3.11-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100-green.svg)
![Flask](https://img.shields.io/badge/Flask-2.2.2-yellow.svg)
![Docker](https://img.shields.io/badge/Docker-24.0.2-blue.svg) 

## Description
This project is a Supply Chain Logistics Multi-class Classification web application built using FastAPI and Flask. It uses a pre-trained DistilBERT model for text classification into various classes related to supply chain logistics. The API is built using FastAPI, and the front-end web interface is created with Flask. The main aim of this project was not to focus on model's accuracy but to implement an end-to-end deployment of a NLP Multi-class Classification Task using a Transfer Learning approach and fine tune a state-of-the-art pre-trained model.

### DistilBERT  ![DistilBERT](https://img.shields.io/badge/DistilBERT-Compact%20%26%20Faster-yellow.svg)      [Documentation](https://huggingface.co/docs/transformers/model_doc/distilbert)

DistilBERT is a compact and faster variant of BERT, a state-of-the-art natural language processing model. It maintains BERT's performance while reducing its size and computation time, making it ideal for resource-constrained environments and faster inference on NLP tasks.  It has 40% less parameters than bert-base-uncased, runs 60% faster while preserving over 95% of BERT’s performances as measured on the GLUE language understanding benchmark.

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
docker build -t my_fastapi_app:v3 .
```

2. Run the Docker container:

```
docker run -d -p 5000:5000 my_fastapi_app:latest
```

3. Access the web app in your browser at http://localhost:5000.

### Colab File

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1NPVSyA0qf4P0-bhdyVUeTsD8B2v58VS1?usp=sharing)

[Files](https://drive.google.com/drive/folders/1wa3O1ZiLrnzPnieJBZcfeiZ6pEOxfKWz?usp=sharing)

[Deployment frontend - (Error due to large size req)](http://rushikesh220.pythonanywhere.com/)

## Some Snapshots of the project : 

![Basic EDA](https://github.com/RushikeshKothawade07/supply-chain-project/blob/main/screenshots/eda.png)

![Textual Data Analysis](https://github.com/RushikeshKothawade07/supply-chain-project/blob/main/screenshots/eda-words.png)

![Flask Web App Home Page](https://github.com/RushikeshKothawade07/supply-chain-project/blob/main/screenshots/home-input.jpg)

![Output predictions](https://github.com/RushikeshKothawade07/supply-chain-project/blob/main/screenshots/output.jpg)

![FastAPI](https://github.com/RushikeshKothawade07/supply-chain-project/blob/main/screenshots/fastapi.jpg)

![Postman JSON queries](https://github.com/RushikeshKothawade07/supply-chain-project/blob/main/screenshots/postman-json.jpg)

![Docker Deployment](https://github.com/RushikeshKothawade07/supply-chain-project/blob/main/screenshots/docker.jpg)


## Rationale for Metrics Selection

In this project, we used the `sparse_categorical_crossentropy` loss function and the `accuracy` metric for evaluating the performance of our multi-class classification model. Below is the rationale for choosing these metrics:

### Loss Function: sparse_categorical_crossentropy

The `sparse_categorical_crossentropy` loss function was chosen because it is well-suited for multi-class classification problems where each sample belongs to exactly one class, and the target labels are represented as integers (class indices). Here's why we chose this loss function:

1. **Integer Target Labels**: In our dataset, the target labels are represented as integers, with each integer corresponding to a specific class. The `sparse_categorical_crossentropy` loss handles these integer target labels directly, without the need for one-hot encoding.

2. **Softmax Activation**: The output layer of our model uses the softmax activation function, which converts the model's raw output scores into probabilities for each class. The `sparse_categorical_crossentropy` loss function works well with softmax activations and computes the cross-entropy loss between the true class labels and the predicted probabilities.

3. **Efficient Computation**: Compared to `categorical_crossentropy`, which requires one-hot encoding of target labels, `sparse_categorical_crossentropy` is computationally more efficient as it avoids the need to convert labels into one-hot format. This is particularly beneficial when dealing with large datasets and complex models.

### Evaluation Metric: Accuracy

We selected the `accuracy` metric to evaluate the performance of our classification model. Accuracy is a widely used evaluation metric for classification problems and measures the proportion of correctly classified samples over the total number of samples. Here's why accuracy was chosen:

1. **Interpretability**: Accuracy is easy to interpret and understand. It provides a clear measure of the model's overall performance by quantifying the percentage of correct predictions.

2. **Balanced Classes**: Our dataset has a relatively balanced distribution of classes. In such scenarios, accuracy is a suitable metric for assessing model performance, as it treats all classes equally.

3. **Effectiveness in Balanced Datasets**: When the classes are balanced, accuracy can provide meaningful insights into how well the model is performing across all classes.

Please note that while accuracy is a valuable metric, it might not be sufficient in cases of imbalanced datasets. In such situations, additional metrics like precision, recall, F1-score, or area under the Receiver Operating Characteristic (ROC) curve might be more informative to assess the model's performance for specific classes.

In conclusion, we chose the `sparse_categorical_crossentropy` loss function and the `accuracy` metric for their efficiency and interpretability in our multi-class classification task. It allowed us to train the model effectively and gauge its overall performance across different classes in our dataset.


## Tasks 
[Link to Tasks README](https://github.com/RushikeshKothawade07/supply-chain-project/blob/main/Tasks/README.md)


## Contributing 

Contributions are welcome! If you find any issues or want to add new features, please open an issue or submit a pull request.



