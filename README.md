# AI Smart Study Planner with Stress Prediction

## Overview

This project is a practical application of Machine Learning aimed at helping students manage their study routine more effectively. It analyzes basic daily habits such as study hours, sleep duration, screen time, and subject difficulty to estimate the student’s stress level.

The main idea behind this project comes from a common issue faced by students, especially during exams — unstructured study patterns and rising stress. This system not only predicts stress but also provides simple suggestions to improve daily habits.

---

## Objectives

* To build a system that can predict student stress levels
* To provide a simple and interactive interface for users
* To encourage better study planning and time management
* To apply Machine Learning to a real-world student problem

---

## Tech Stack

* Python
* Pandas and NumPy
* Scikit-learn
* Streamlit

---

## Project Structure

```
study-planner-ai/
│
├── data/
│   └── student_stress.csv
│
├── model/
│   └── model.pkl
│
├── app.py
├── train_model.py
└── requirements.txt
```

---

## How It Works

The project follows a simple workflow:

1. A dataset is created containing features like study hours, sleep hours, screen time, and difficulty level.
2. The stress level is used as the target variable.
3. A Machine Learning model (Logistic Regression) is trained on this data.
4. The trained model is saved and used in the main application.
5. The user enters their daily routine in the app, and the system predicts the stress level.
6. Based on the prediction, the app suggests ways to improve study habits.

---

## How to Run the Project

1. Clone the repository

```
git clone <your-repo-link>
cd study-planner-ai
```

2. Install dependencies

```
pip install -r requirements.txt
```

3. Train the model

```
python train_model.py
```

4. Run the application

```
python -m streamlit run app.py
```

5. Open the application in a browser

```
http://localhost:8501
```

---

## Features

* Simple and clean user interface
* Stress level prediction (Low, Medium, High)
* Suggestions based on user input
* Lightweight and easy to run
* Designed for real-world usability

---

## Future Scope

* Use a larger and more realistic dataset
* Improve model performance using advanced algorithms
* Add user login and progress tracking
* Extend the system to include time-table generation

---

## Conclusion

This project demonstrates how Machine Learning can be used to address everyday challenges faced by students. By combining prediction with simple guidance, it provides a basic framework for improving study habits and managing stress in a practical way.

---

## Author

Developed by Achal Ghate
