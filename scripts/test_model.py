from keras.models import  load_model
import os

def test_model(version):

    generator = load_model(os.path.join("..","models","gen{}_generator.h5".format(version)))
    critic =  load_model(os.path.join("..","models","gen{}critic.h5".format(version)))

    test_for_generator(generator)
    test_for_critic(critic)


def test_for_critic(critic):
    ...

def test_for_generator(generator):
    ...


