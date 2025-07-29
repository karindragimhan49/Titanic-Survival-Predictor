# 🚢 Titanic Survival Predictor

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.1.2-orange?style=for-the-badge&logo=scikit-learn)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.12.0-red?style=for-the-badge&logo=streamlit)](https://streamlit.io/)
[![GitHub license](https://img.shields.io/badge/license-MIT-green?style=for-the-badge)](https://github.com/[your-github-username]/Titanic-Survival-Predictor/blob/main/LICENSE)

An interactive web application that predicts whether a passenger would have survived the sinking of the Titanic. This project demonstrates a complete end-to-end machine learning workflow, from data exploration and model training to deployment as a user-friendly web app.

<br>

<!-- 
    IMAGE PLACEHOLDER 1: Add a GIF or a high-quality screenshot of your running Streamlit app here.
    Follow the instructions in the previous chat to get an image URL.
-->
![Titanic Survival Predictor App Demo](https://user-images.githubusercontent.com/...) 

---

## ✨ Core Features

- **Interactive UI:** A clean and intuitive interface built with Streamlit, allowing users to input passenger details easily via a sidebar.
- **Real-time Predictions:** Leverages a trained Logistic Regression model to provide instant survival predictions.
- **Data-Driven Insights:** The model is trained on the official Kaggle Titanic dataset, capturing real historical patterns and correlations.
- **Detailed & Visual Results:** Displays not just the final verdict (Survived/Not Survived) but also the prediction probability, enhanced with progress bars and celebratory (or commiseratory) GIFs.
- **Reproducible Workflow:** The entire process is scripted and documented, ensuring that the model can be retrained and the results reproduced reliably.

---

## 🚀 Getting Started

Follow these instructions to get a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

- [Python 3.9](https://www.python.org/downloads/) or later
- `pip` and `venv` for package management

### Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/[your-github-username]/Titanic-Survival-Predictor.git
    cd Titanic-Survival-Predictor
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # For Windows
    python -m venv venv
    .\venv\Scripts\activate

    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the required packages:**
    All necessary libraries are listed in `requirements.txt`.
    ```bash
    pip install -r requirements.txt
    ```

---

## 🏃‍♀️ How to Run the Project

### 1. Model Training (Optional)

The repository includes a pre-trained model. However, if you wish to retrain it, you can run the training script. This will process the data from `data/train.csv` and save a new model file to `models/titanic_model.joblib`.

```bash
python src/train.py