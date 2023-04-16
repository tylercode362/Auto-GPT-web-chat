"""Base class for memory providers."""
import abc

import spacy

from autogpt.config import AbstractSingleton, Config

cfg = Config()

nlp = spacy.load("en_core_web_md")

def get_ada_embedding(text):
    text = text.replace("\n", " ")
    doc = nlp(text)
    return doc.vector

class MemoryProviderSingleton(AbstractSingleton):
    @abc.abstractmethod
    def add(self, data):
        pass

    @abc.abstractmethod
    def get(self, data):
        pass

    @abc.abstractmethod
    def clear(self):
        pass

    @abc.abstractmethod
    def get_relevant(self, data, num_relevant=5):
        pass

    @abc.abstractmethod
    def get_stats(self):
        pass
