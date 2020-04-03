from keras import Input,Dense
from keras.models import  Model

class Encoder(Model):

    def __init__(self):
        super(Encoder,self).__init__()


class Decoder(Model):

    def __init__(self,input_size=4):

        self.input_layer = Input(4)



class AutoEncoder(Model):

    def __init__(self):

        super(AutoEncoder,self).__init__()
        self.encoder = Encoder(),
        self.decoder = Decoder()

    def createEncoder(self):

        ...
