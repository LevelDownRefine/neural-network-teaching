"""Common model interface, following KAIR naming conventions."""


class ModelBase:
    def feed_data(self, data):
        pass

    def optimize_parameters(self):
        pass

    def test(self):
        pass

    def current_results(self):
        pass

    def reset(self):
        pass

    def state_dict(self):
        pass

    def load_state_dict(self, state):
        pass
