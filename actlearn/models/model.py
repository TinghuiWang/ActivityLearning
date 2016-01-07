from actlearn.log.logger import actlearn_logger


class Model(object):
    """
    Abstract super class that defines the interface for
    learning models
    """

    def __init__(self, model_name):
        """
        Initialization Model Object
        :param model_name: String, Model Name
        :return:
        """
        self.params = []
        self.model_name = model_name
        self.logger = actlearn_logger.get_logger(model_name)
        pass

    def cost(self, y):
        """
        Symbolic Tensor for cost calculation of the model
        :param y: minibatch target class array
        :return:
        """
        raise NotImplementedError()

    def error(self, y):
        """
        Symbolic Tensor for calculating the classification
        errors of a mini-batch input
        :param y:
        :return:
        """
        raise NotImplementedError()

    def classify(self, data):
        """
        Symbolic Tensor for running the classifier on given
        mini-batch input
        :return: array of class
        """
        raise NotImplementedError()

    # def monitor(self):
    #     """
    #     Return Monitor Statistics
    #     :return:
    #     """
    #     raise NotImplementedError()

    def save(self, filename):
        """
        Save current model parameters to file in Pickle format
        :param filename: Name of the file to save the model to
        :return:
        """
        raise NotImplementedError()

    def load(self, filename):
        """
        Load saved model parameter from file
        :param filename: Name of the file to load the model from
        :return:
        """
        raise NotImplementedError()

    def export(self, filename, type):
        """
        Export current model to file
        :param filename: Name of the file to export the model to
        :param type: export file type
        :return:
        """
        raise NotImplementedError()