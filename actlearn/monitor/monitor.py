

class Monitor(object):
    """
    Abstract super class that defines the interface for
    learning models
    """

    def __init__(self, name):
        """
        Initialize Monitor Structure by given name to it
        :param name:
        :return:
        """
        self.name = name
