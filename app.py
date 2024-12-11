import re

from dotenv import load_dotenv
from flask import Flask, redirect, render_template, request, url_for
import pandas as pd

from models import all_models
from data import check_downloaded_data, data_downloader

load_dotenv()

# TODO: use better name for this
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


def process_the_output(
    predictions: list[dict[str, float | str]] | pd.DataFrame
) -> pd.DataFrame:
    if isinstance(predictions, list):
        predictions = pd.DataFrame(predictions)
    assert all(
        predictions.columns == ["label", "score"]
    ), f"Invalid columns: {predictions.columns}"
    predictions = predictions.sort_values("label")
    predictions["score"] /= predictions["score"].sum()

    return predictions.to_dict(orient="records")


def calculate_overall(
    all_predictions: list[dict[str, str | list[dict[str, float]]]]
) -> pd.DataFrame:
    all_predictions_df = []
    for prediction in all_predictions:
        df = pd.DataFrame(prediction["results"]).set_index("label")
        df.columns = [prediction["name"]]
        all_predictions_df.append(df)
    all_predictions_df = pd.concat(all_predictions_df, axis=1)
    assert (
        all_predictions_df.isna().sum().sum() == 0
    ), f"Missing values in the predictions: {all_predictions_df}"
    all_predictions_df["overall"] = all_predictions_df.mean(axis=1)
    return {
        "name": "overall",
        "results": process_the_output(
            all_predictions_df[["overall"]]
            .rename(columns={"overall": "score"})
            .reset_index()
            .to_dict(orient="records")
        ),
    }


def select_best_label(predictions: list[dict[str, float | str]]) -> str:
    return max(predictions, key=lambda x: x["score"])["label"]


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
                {
                    "name": humanize_model_name(model.__class__.__name__),
                    "results": process_the_output(out),
                }
            )
    # The result is a list of dictionaries, where each dictionary has two keys: 'name' and 'results'
    # 'name' is the name of the model, and 'results' is the result of the model
    # The result of the model is a list of dictionaries, where each dictionary has two keys: 'label' and 'score'
    # return render_template("analyze.html", text=text, results=predicts)
    predicts.append(calculate_overall(predicts))
    return render_template(
        "chart_analyze.html",
        text=text,
        data=predicts,
        all_models=[p["name"] for p in predicts],
        all_labels=[p["label"] for p in predicts[0]["results"]],
    )


@app.route("/show_data/<social_media>")
def show_data(social_media):
    df = check_downloaded_data(social_media)
    if df is None:
        return render_template(
            f"show_data_{social_media}.html",
            is_data_available=False,
        )
    return render_template(
        f"show_data_{social_media}.html",
        table=df.to_html(),
        data_size=len(df),
        is_data_available=True,
        columns=df.columns,
        social_media_name=social_media,
        model_names=[
            humanize_model_name(model.__class__.__name__) for model in all_models_
        ],
    )


@app.route("/download_<social_media>", methods=["POST"])
def download_data(social_media):
    func = data_downloader(social_media)
    func(**request.form)
    return redirect(url_for("show_data", social_media=social_media))


@app.route("/label_data/<social_media>", methods=["POST"])
def label_data(social_media):
    df = check_downloaded_data(social_media)
    assert df is not None, f"No data for {social_media}"
    df = df[request.form["col_name"]].dropna().to_frame()
    model = all_models_[int(request.form["model_index"])]
    model_name = humanize_model_name(model.__class__.__name__)
    predictions = []
    for (row,) in df.itertuples(index=False):
        out = model.predict(row)
        if out:
            predictions.append(select_best_label(out))
        else:
            predictions.append("N/A")
    df["predicted label"] = predictions
    return render_template(
        "label_data.html",
        table=df.to_html(),
        data_size=len(df),
        columns=df.columns,
        social_media_name=social_media,
        model_name=model_name,
    )


if __name__ == "__main__":
    app.run(debug=True)
