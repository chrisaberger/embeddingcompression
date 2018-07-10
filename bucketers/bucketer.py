class Bucketer:
    def __init__(self):
        raise NotImplementedError

    def name(self):
        raise NotImplementedError

    def bucket(self, X):
        raise NotImplementedError