# 🎵 Music Genre Classification using CNNs

![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![Keras](https://img.shields.io/badge/Keras-2.x-red.svg)
![Librosa](https://img.shields.io/badge/Librosa-0.9.x-green.svg)

A deep learning project to classify music into one of 10 genres using a Convolutional Neural Network (CNN). This repository contains a Google Colab notebook that walks through the entire process, from audio processing and feature extraction to model training and real-time prediction.

---

## 🚀 Features

- **End-to-End Pipeline:** Complete code from data preprocessing to model evaluation.
- **MFCC Feature Extraction:** Uses Librosa to convert raw audio into Mel-Frequency Cepstral Coefficients (MFCCs), ideal for audio classification.
- **CNN Architecture:** Implements a robust Convolutional Neural Network built with Keras/TensorFlow to learn features from the MFCC spectrograms.
- **Audio Prediction Demo:** Includes a script to upload your own `.mp3` or `.wav` file and get an instant genre prediction with a confidence score.
- **Spectrogram Visualization:** Generates and displays a Mel spectrogram for any uploaded audio file.
- **Trained Model Included:** Comes with a pre-trained `music_genre_classifier.h5` model file, so you can run predictions immediately.

---

## 🛠️ Tech Stack

- **Programming Language:** Python
- **Deep Learning:** TensorFlow, Keras
- **Audio Processing:** Librosa
- **Data Manipulation:** NumPy, Scikit-learn
- **Visualization:** Matplotlib
- **Environment:** Google Colaboratory (recommended for GPU support)

---

## 🏁 Getting Started

Follow these steps to set up and run the project.

### Prerequisites

- A Google Account (to use Google Colab).
- A Kaggle Account (to download the dataset).

### Installation & Setup

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/your-username/your-repository-name.git](https://github.com/your-username/your-repository-name.git)
    cd your-repository-name
    ```

2.  **Get Kaggle API Token**
    - Go to your Kaggle account page: [www.kaggle.com/account](https://www.kaggle.com/account).
    - Scroll down to the "API" section.
    - Click **"Create New API Token"**. This will download a file named `kaggle.json`. Keep this file handy.

3.  **Open in Google Colab**
    - Go to [colab.research.google.com](https://colab.research.google.com).
    - Click `File` -> `Upload notebook...` and upload the `music_genre_classification.ipynb` file from this repository.

4.  **Enable GPU Acceleration**
    - In the Colab notebook, go to the menu and click `Runtime` -> `Change runtime type`.
    - Select **`GPU`** as the "Hardware accelerator" and click `Save`. This will make model training significantly faster.

---

## 🖥️ Usage

The notebook is divided into cells. Run them in order from top to bottom.

1.  **Run the Setup Cell (Cell 1):**
    - This cell will install the necessary libraries.
    - It will then prompt you to upload the `kaggle.json` file you downloaded earlier.
    - Finally, it will automatically download and unzip the GTZAN dataset from Kaggle.

2.  **Run Feature Extraction (Cell 2):**
    - This cell processes all the `.wav` files from the dataset, extracts MFCCs, and saves them to a `data.json` file. This step may take several minutes.

3.  **Run Data Loading & Splitting (Cell 3):**
    - This cell loads the data from `data.json` and splits it into training and testing sets.

4.  **Run Model Training (Cell 4):**
    - This cell builds, compiles, and trains the CNN model for 50 epochs. This is the most time-consuming step and will benefit greatly from the GPU runtime.

5.  **Run Evaluation (Cell 5):**
    - This cell plots the accuracy/loss curves and saves the trained model as `music_genre_classifier.h5`.

6.  **Run the Prediction Demo (Cell 6):**
    - This is the fun part! This cell will prompt you to upload your own audio file (`.mp3` or `.wav`).
    - After uploading, it will display the audio's spectrogram and print the predicted genre with its confidence score.

---

## 📂 Project Structure


.
├── music_genre_classification.ipynb    # The main Google Colab notebook
├── music_genre_classification.py       # .py file
├── music_genre_classifier.h5           # The pre-trained model file
├── README.md                           # This file
├── project report                      #pdf of project report
└── music_classification_video          #video depicting usage


---

## 🙏 Acknowledgments

This project uses the **GTZAN Genre Collection** dataset, collected by G. Tzanetakis. A huge thank you to the creators for making this valuable resource available to the community.

* G. Tzanetakis and P. Cook, "Musical Genre Classification of Audio Signals," IEEE Transactions on Audio and Speech Processing, 2002.

---

## 📧 Contact

Anish Kar - [anishkar3@gmail.com]
