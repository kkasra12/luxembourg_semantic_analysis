import abc
from pyexpat import model
import warnings

import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import torch
from transformers import pipeline
from google.cloud import language_v1


class modelInerface(abc.ABC):
    def predict(self, text, *, threshold=None, label_mapper=None):
        pass

    def load(self, path):
        pass

    def save(self, path):
        pass

    def fit(self, x_train, y_train, *, x_test=None, y_test=None):
        pass

    @property
    def all_labels(self):
        pass


class DistilbertBaseUncasedEmotion(modelInerface):
    """This model is using the distilbert base uncased model to predict the emotion of the text.
    The model weights are getting from "hadresh-savani/distilbert-base-uncased-emotion" model.
    for more info check here: https://huggingface.co/bhadresh-savani/distilbert-base-uncased-emotion
    This is a sample output from the model:
    ```
    [[
    {'label': 'sadness', 'score': 0.0006792712374590337},
    {'label': 'joy', 'score': 0.9959300756454468},
    {'label': 'love', 'score': 0.0009452480007894337},
    {'label': 'anger', 'score': 0.0018055217806249857},
    {'label': 'fear', 'score': 0.00041110432357527316},
    {'label': 'surprise', 'score': 0.0002288572577526793}
    ]]
    ```

    Args:
        modelInerface (_type_): _description_
    """

    def __init__(self, device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipeline = self.load(device=device)

    def predict(self, text, *, threshold=None, label_mapper=None):
        out = self.pipeline(text)
        if len(out) > 1:
            warnings.warn(
                "The model is returning more than one output, the first one will be used.\n the input is {}".format(
                    text
                )
            )

        if len(out) == 0:
            return []

        out = out[0]

        # TODO: use label mapper
        return pd.DataFrame(out).sort_values("score", ascending=False)

    @property
    def all_labels(self):
        return [o["label"] for o in self.pipeline("This is a random text")[0]]

    def save(self, path):
        pass

    def load(self, device=None):
        return pipeline(
            "text-classification",
            model="bhadresh-savani/distilbert-base-uncased-emotion",
            top_k=None,
            device=device,
        )

    def fit(self, x_train, y_train, *, x_test=None, y_test=None):
        pass


class GoogleCloudNLP(modelInerface):
    def __init__(self, device=None):
        self.model = language_v1.LanguageServiceClient()

    def predict(self, text, *, threshold=None, label_mapper=None):
        document = language_v1.Document(
            content="I am feeling fantastic!",
            type_=language_v1.Document.Type.PLAIN_TEXT,
        )
        sentiment = self.model.analyze_sentiment(
            request={"document": document}
        ).document_sentiment


class NltkSentiment(modelInerface):
    def __init__(self):
        # nltk.download('vader_lexicon')
        self.model = SentimentIntensityAnalyzer()
        try:
            self.model.polarity_scores("This is a random text")
        except LookupError:
            nltk.download("vader_lexicon")
            self.model = SentimentIntensityAnalyzer()

    def predict(self, text, *, threshold=None, label_mapper=None):
        return (
            pd.Series(self.model.polarity_scores(text))
            .to_frame(name="score")
            .reset_index()
            .rename(columns={"index": "label"})
            .sort_values("score", ascending=False)
        )

    @property
    def all_labels(self):
        return list(self.model.polarity_scores("This is a random text").keys())


if __name__ == "__main__":
    # model = DistilbertBaseUncasedEmotion()
    model = NltkSentiment()
    print(
        model.predict(
            "I am happy, I am sad, I am angry, I am surprised, I am scared, I am in love"
        )
    )
    print(model.all_labels)
