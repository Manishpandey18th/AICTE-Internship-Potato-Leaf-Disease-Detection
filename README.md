# AICTE-Internship-Potato-Leaf-Disease-Detection

# Potato Leaf Disease Detection

This project uses **deep learning** to detect diseases in potato leaves. It includes a model training pipeline and a Streamlit-based web application for real-time disease detection.

## Features

* **Image Preprocessing and Augmentation:** Images are preprocessed and augmented to improve model performance and robustness.
* **Deep Learning Model (CNN):** A **Convolutional Neural Network (CNN)** model, a type of deep learning model, is trained to classify potato leaves as healthy or diseased (specifically, early blight or late blight).
* **Streamlit Web Application:** A user-friendly web interface built with Streamlit allows for easy image uploads and disease predictions.
* **Deployment-Ready Model:** The trained model can be easily deployed for real-world applications.

## Technologies Used

* **Python:** The primary programming language.
* **TensorFlow/Keras:** Used for building and training the **deep learning (CNN)** model.
* **Streamlit:** Framework for creating the interactive web application.
* **OpenCV:** Library for image processing and manipulation.
* **NumPy & Pandas:** Libraries for numerical computation and data manipulation.
* **Matplotlib & Seaborn:** Libraries for data visualization.

## Dataset

The dataset consists of labeled images of potato leaves, categorized into "healthy," "early blight," and "late blight" classes. 
The images are preprocessed before training, which may include resizing, normalization, and augmentation. 
*(Optional: Add a link to where the dataset can be found if it's publicly available.)*

## Installation
1. **Clone the Repository:**
   ```
   git clone [https://github.com/Manishpandey18th/AICTE-Internship-Potato-Leaf-Disease-Detection.git](https://www.google.com/search?q=https://github.com/Manishpandey18th/AICTE-Internship-Potato-Leaf-Disease-Detection.git)
   cd Potato-Leaf-Disease-Detection
   ```
   
2. **Install Dependencies :**
```
 pip install -r requirements.txt
```

## Model Training (Jupyter Notebook):
•	Open and run your training script in Jupyter Notebook e.g 
``` 
 jupyter notebook Train_potato_disease.ipynb
```

•	Critical: Before running, ensure correct dataset and model save paths are set within the training script.

## Running the Streamlit App:
1.	Update the model path in web.py to the correct relative path of your saved model.
2.	Run the app: 
```
streamlit run web.py
```
3.	Upload an image for disease prediction.

## Enhancements:
•	``` web.py ``` model path updated for portability.

•	Duplicate ```'Home' ``` condition in ``` web.py ``` fixed.

•	Added instructions for Streamlit deployment.

