from keras.layers import Input,Dense
from keras.models import  Model,Sequential

def create_encoder(input_size = 2,output_size=2):

    model = Sequential()


    return model



def create_decoder(input_size = 2,output_size=2):

    ...




def create_autoencoder(encoder:Model,decoder:Model):

    model = Model(inputs=[encoder],outputs=[decoder(encoder)])

    model.compile(optimizer="adam",loss=)


def train(data,version):

    encoder = create_encoder()
    decoder = create_decoder()
    autoencoder = create_autoencoder(encoder,decoder)



    ...