import abc

from transformers import pipeline


class modelInerface(abc.ABC):
    def predict(self, text, *, threshold=None, label_mapper=None):
        pass

    def load(self, path):
        pass

    def save(self, path):
        pass

    def fit(self, x_train, y_train, *,x_test=None, y_test=None):
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
    def __init__(self):
        self.pipeline = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion", return_all_scores=True)

    def predict(self, text):
        return self.pipeline(text)[0]
    
    @property
    def all_labels(self):
        return [o['label'] for o in self.pipeline("This is a random text")]
    
    def save(self, path):
        pass

    def load(self, path):
        pass

    def fit(self, x_train, y_train, *,x_test=None, y_test=None):
        pass



if __name__ == "__main__":
    model = DistilbertBaseUncasedEmotion()
    print(model.predict("I am happy"))
    print(model.all_labels)
    