import abc
import json
import warnings
import os

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import torch
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import login as huggingface_login
from google.cloud import language_v1

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


class modelInerface(abc.ABC):
    @abc.abstractmethod
    def predict(
        self, text: str, *, threshold: float = None, label_mapper: dict[str, str] = None
    ) -> list[dict[str, float]]:
        """
        This method should return the sentiment of the text.

        Args:
            text (str): The text to predict the sentiment.
            threshold (float, optional): The threshold to filter the results. Defaults to None.
            label_mapper (dict[str, str], optional): The mapper to map the labels to the desired labels. Defaults to None.

        Returns:
            list[dict[str, float]]: The sentiment of the text, each key should have the key "label" and "score".
        """
        pass

    @abc.abstractmethod
    def load(self, path):
        pass

    def save(self, path):
        pass

    def fit(self, x_train, y_train, *, x_test=None, y_test=None):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def all_labels(self):
        pass

    def __call__(self):
        print("This is a call method")

    # TODO: implement the __call__ mthod and move the label_mapper and assertiions there and sort the labels based on the scores.
    # assert for the output format
    # assert for the input format
    # TODO: instead of using a list of dicts for output, use a class.


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

        return out[0]

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


class BertBaseUncasedEmotion(modelInerface):
    def __init__(self):
        self.model_location = os.path.join(os.path.dirname(__file__), "bert-base-uncased-emotion")
        tokenizer, model = self.load()
        self.tokenizer = tokenizer
        self.model = model

    def load(self):
        # tokenizer = AutoTokenizer.from_pretrained("nateraw/bert-base-uncased-emotion")
        # model = AutoModelForSequenceClassification.from_pretrained("nateraw/bert-base-uncased-emotion")
        print("the __file__ is")
        print(__file__)
        tokenizer = AutoTokenizer.from_pretrained(self.model_location)
        model = AutoModelForSequenceClassification.from_pretrained(self.model_location)
        return tokenizer, model

    def predict(self, text, *, threshold=None, label_mapper=None):
        inputs = self.tokenizer(text, return_tensors="pt")
        outputs = self.model(**inputs)
        scores = (
            torch.nn.functional.softmax(outputs.logits, dim=1)
            .detach()
            .numpy()
            .tolist()[0]
        )
        return [{"label": k, "score": v} for k, v in zip(self.all_labels, scores)]

    @property
    def all_labels(self):
        with open(os.path.join(self.model_location, "config.json")) as f:
            config = json.load(f)
        return config["id2label"].values()


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

        return
        return {"sentiment": sentiment.score}
    
    @property
    def all_labels(self):
        return ["positive", "negative", "neutral"]

    def load(self, path):
        return super().load(path)


class NltkSentiment(modelInerface):
    def __init__(self):
        # nltk.download('vader_lexicon')
        try:
            self.model = SentimentIntensityAnalyzer()
            self.model.polarity_scores("This is a random text")
        except LookupError:
            self.load()
            self.model = SentimentIntensityAnalyzer()

    def predict(self, text, *, threshold=None, label_mapper=None):
        # return (
        #     pd.Series(self.model.polarity_scores(text))
        #     .to_frame(name="score")
        #     .reset_index()
        #     .rename(columns={"index": "label"})
        #     .sort_values("score", ascending=False)
        # )
        return [
            {"label": k, "score": v}
            for k, v in self.model.polarity_scores(text).items()
        ]

    @property
    def all_labels(self):
        return list(self.model.polarity_scores("This is a random text").keys())

    def load(self):
        nltk.download("vader_lexicon")


class Llama3(modelInerface):
    def __init__(self, is_offline=False):
        self.is_offline = is_offline
        if not is_offline:
            self.load()
            self.pipe = pipeline(
                "text-generation",
                model="meta-llama/Llama-3.2-3B",
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )

        else:
            raise NotImplementedError

    def predict(self, text, *, threshold=None, label_mapper=None):
        label = self.pipe(
            (message:=f"The output shoud be one of {self.all_labels}.\n{text}, Sentiment of the text is:"),
            max_new_tokens=1,
            pad_token_id=self.pipe.tokenizer.eos_token_id
        )[0]['generated_text'].replace(message, "").strip()
        out = [{"label": k, "score": 0 if label != k else 1} for k in self.all_labels]
        return out

    @property
    def all_labels(self):
        return ["positive", "negative", "neutral"]

    def load(self):
        huggingface_login(token=os.environ["HUGGING_FACE_API"])


# https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest
if __name__ == "__main__":
    # model_ = DistilbertBaseUncasedEmotion()
    # model_ = NltkSentiment()
    # model_ = GoogleCloudNLP()
    # model_ = BertBaseUncasedEmotion()
    model_ = Llama3()
    print(model_.predict("I am happy"))
    print(model_.all_labels)
