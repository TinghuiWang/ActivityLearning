from actlearn.monitor.monitor import Monitor


class ModelParamMonitor(Monitor):
    """
    Model Parameter Monitor

    The monitor monitors parameter defined by the model
    """

    def __init__(self, name, param):
        Monitor.__init__(self, name)
        self.tensor = param
