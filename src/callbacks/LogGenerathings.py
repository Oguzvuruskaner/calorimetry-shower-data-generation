from pytorch_lightning.callbacks import Callback


class LogGenerathings(Callback):


    def __init__(self,log_dir,particle_limit = 30):

        #How many particles will be generated.
        self.particle_limit = particle_limit
        self.log_dir = log_dir

    def on_epoch_end(self, trainer, pl_module):


        c = 3