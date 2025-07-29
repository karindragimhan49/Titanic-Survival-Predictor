🚢 Titanic Survival Predictor

![alt text](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python)


![alt text](https://img.shields.io/badge/Scikit--learn-1.1.2-orange?style=for-the-badge&logo=scikit-learn)


![alt text](https://img.shields.io/badge/Streamlit-1.12.0-red?style=for-the-badge&logo=streamlit)


![alt text](https://img.shields.io/badge/license-MIT-green?style=for-the-badge)

An interactive web application built with Streamlit and Scikit-learn that predicts whether a passenger would have survived the Titanic disaster based on their personal details. This project serves as a complete end-to-end demonstration of a machine learning workflow, from data exploration to model deployment.

(Note: Replace this with a screenshot of your own running app! You can drag and drop your screenshot into a GitHub issue comment to get a URL like this.)

✨ Features

Interactive UI: A user-friendly interface built with Streamlit, featuring a sidebar for inputting passenger details.

Real-time Predictions: Instantly predicts survival probability using a trained Logistic Regression model.

Data-Driven Insights: The model is trained on the official Kaggle Titanic dataset, capturing real historical patterns.

Detailed Results: Displays not just the final verdict (Survived/Not Survived) but also the prediction probability with visual aids.

Reproducible Workflow: The entire process, from data cleaning to model training, is documented in a Jupyter Notebook and scripted for easy reproduction.

🚀 Getting Started

Follow these instructions to get a copy of the project up and running on your local machine.

Prerequisites

Python 3.9 or later

pip and venv for package management

Installation & Setup

Clone the repository:

Generated bash
git clone https://github.com/[your-github-username]/Titanic-Survival-Predictor.git
cd Titanic-Survival-Predictor


Create and activate a virtual environment:

Generated bash
# For Windows
python -m venv venv
.\venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END

Install the required packages:
All necessary libraries are listed in requirements.txt.

Generated bash
pip install -r requirements.txt
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END
🏃‍♀️ How to Run

There are two main components to this project: the exploration notebook and the final web application.

1. Model Training

The model is pre-trained, but if you wish to retrain it or see the training process, you can run the training script. This will process the data from data/train.csv and save a new model file to models/titanic_model.joblib.

Generated bash
python src/train.py
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END
2. Launching the Web App

To start the interactive Streamlit application, run the following command from the project's root directory:

Generated bash
streamlit run app.py
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END

This will open a new tab in your web browser at http://localhost:8501.

📂 Project Structure

The repository is organized as follows:

Generated code
Titanic-Survival-Predictor/
├── data/
│   └── train.csv             # The raw dataset used for training
├── models/
│   └── titanic_model.joblib  # The pre-trained and saved model file
├── notebooks/
│   └── titanic-exploration.ipynb # Jupyter Notebook for data analysis and model prototyping
├── src/
│   └── train.py              # Script to train the model and save it
├── .gitignore                # Specifies files to be ignored by Git
├── app.py                    # The main Streamlit application file
├── LICENSE                   # MIT License file
├── README.md                 # You are here!
└── requirements.txt          # List of Python packages required for the project
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
IGNORE_WHEN_COPYING_END
🛠️ Tech Stack

Data Analysis & Modeling:

Pandas: For data manipulation and analysis.

NumPy: For numerical operations.

Scikit-learn: For building and evaluating the machine learning model.

Jupyter Notebook: For interactive data exploration.

Web App & Deployment:

Streamlit: For creating and serving the interactive web UI.

Tools:

Git & GitHub: For version control and project management.

VS Code: As the primary code editor.

📜 License

This project is licensed under the MIT License. See the LICENSE file for more details.