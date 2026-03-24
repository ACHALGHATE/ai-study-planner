from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# load model
with open("model/model.pkl", "rb") as file:
    model = pickle.load(file)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    # get values from form
    study_hours = int(request.form.get("study"))
    sleep_hours = int(request.form.get("sleep"))
    screen_time = int(request.form.get("screen"))
    difficulty = int(request.form.get("difficulty"))

    # prepare data
    input_data = np.array([[study_hours, sleep_hours, screen_time, difficulty]])

    # prediction
    prediction = model.predict(input_data)[0]

    # result + plan
    if prediction == 0:
        result = "Low Stress"
        plan = "You can increase study time and maintain consistency."

    elif prediction == 1:
        result = "Medium Stress"
        plan = "Balance study and breaks properly."

    else:
        result = "High Stress"
        plan = "Reduce workload and take proper rest."

    # 🔥 TERMINAL OUTPUT
    print("\nNew Prediction Request")
    print(f"Study Hours   : {study_hours}")
    print(f"Sleep Hours   : {sleep_hours}")
    print(f"Screen Time   : {screen_time}")
    print(f"Difficulty    : {difficulty}")
    print(f"Result        : {result}")
    print(f"Plan          : {plan}")
    print("-" * 40)

    return render_template(
        "index.html",
        result=result,
        plan=plan,
        show_button=True
    )


if __name__ == "__main__":
    app.run(debug=True)