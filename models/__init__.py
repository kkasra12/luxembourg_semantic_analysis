from .sentiment_models import (
    BertBaseUncasedEmotion,
    DistilbertBaseUncasedEmotion,
    GoogleCloudNLP,
    Llama3,
    NltkSentiment,
    modelInerface,
)

all_models = [
    DistilbertBaseUncasedEmotion,  # Entaon
    NltkSentiment,  # Yashar
    BertBaseUncasedEmotion,  # Dylan
    # GoogleCloudNLP,
    # Llama3,
]

assert all(model.__bases__[0] == modelInerface for model in all_models), (
    "All models should inherit from modelInerface, "
    f"but {[model.__name__ for model in all_models if model.__bases__[0] != modelInerface]} do not."
)

__all__ = ["DistilbertBaseUncasedEmotion", "NltkSentiment", "all_models"]
