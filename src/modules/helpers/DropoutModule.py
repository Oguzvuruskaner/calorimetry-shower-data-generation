class DropoutModule:

    def __init__(self, dropout_rate):
        self.dropout_rate = dropout_rate


    def decay_dropout_rate(self,gamma = 0.95):

        self.dropout_rate *= gamma
        if self.dropout_rate < 0.01:
            self.dropout_rate = 0