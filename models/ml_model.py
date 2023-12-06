from abc import ABC, abstractmethod

class MLModel(ABC):

    @abstractmethod
    def init_parameters(self):
        pass

    @abstractmethod
    def forward_propogation(self):
        pass

    @abstractmethod
    def compute_cost(self):
        pass

    @abstractmethod
    def back_propogation(self):
        pass

    @abstractmethod
    def update_parameters(self):
        pass

    @abstractmethod
    def train(self, X, Y, iters):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def score(self, A, Y):
        pass

    @abstractmethod
    def get_parameters(self):
        pass


