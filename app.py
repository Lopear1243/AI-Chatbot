# app.py
from flask import Flask, render_template, request, jsonify
from chatbot import predict_class, get_response  # ✅ use chatbot, not train_chatbot!

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get", methods=["POST"])
def chatbot_response():
    user_message = request.json["message"]
    intents = predict_class(user_message)
    response = get_response(intents)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)
