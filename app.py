
from flask import Flask, request, render_template
import pickle
import pandas as pd

app = Flask(__name__)

# Load updated model
with open("addiction_model_v4.pkl", "rb") as f:
    model = pickle.load(f)

def get_feedback(level):
    if level <= 3:
        return "🎯 Low Addiction — You’re using social media mindfully!"
    elif level <= 6:
        return "⚠️ Moderate Addiction — Try balancing your usage."
    else:
        return "🔥 High Addiction — Consider a break or setting limits."

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    feedback = ""
    input_data = {}

    if request.method == "POST":
        try:
            input_data = {
                "platform": request.form["platform"],
                "total_time": int(request.form["total_time"]),
                "sessions": int(request.form["sessions"]),
                "engagement": float(request.form["engagement"]),
                "scroll_rate": float(request.form["scroll_rate"])
            }

            avg_time = input_data["total_time"] / input_data["sessions"]
            hooked_score = input_data["engagement"] * input_data["scroll_rate"]

            df_input = pd.DataFrame([{
                "Platform": input_data["platform"],
                "Total Time Spent": input_data["total_time"],
                "Number of Sessions": input_data["sessions"],
                "Engagement": input_data["engagement"],
                "Scroll Rate": input_data["scroll_rate"],
                "Avg Time per Session": avg_time,
                "Hookedness Score": hooked_score
            }])

            prediction = round(model.predict(df_input)[0], 2)
            feedback = get_feedback(prediction)

        except Exception as e:
            prediction = f"Error: {e}"

    return render_template("index.html", prediction=prediction, feedback=feedback, input_data=input_data)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)