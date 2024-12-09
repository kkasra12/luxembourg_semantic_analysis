import re

from dotenv import load_dotenv
from flask import Flask, render_template, request

from models import all_models

load_dotenv()


all_models_ = [model() for model in all_models]

app = Flask(__name__)


def humanize_model_name(name: str) -> str:
    """This function convert the "PascalCase" name of a class to "Pascal case" which is more readable for humans.

    Args:
        name (str): The name of the class

    Returns:
        str: The humanized name of the class
    """
    return " ".join([word.capitalize() for word in re.findall(r"[A-Z][a-z]*", name)])


@app.route("/hello/")
@app.route("/hello/<name>")
def hello(name=None):
    return render_template("hello.html", person=name)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    text = request.form["input_text"]
    predicts = []
    for model in all_models_:
        out = model.predict(text)
        if out:
            predicts.append(
                {"name": humanize_model_name(model.__class__.__name__), "results": out}
            )
    # The result is a list of dictionaries, where each dictionary has two keys: 'name' and 'results'
    # 'name' is the name of the model, and 'results' is the result of the model
    # The result of the model is a list of dictionaries, where each dictionary has two keys: 'label' and 'score'
    return render_template("analyze.html", text=text, results=predicts)


if __name__ == "__main__":
    app.run(debug=True)
