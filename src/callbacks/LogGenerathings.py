from pytorch_lightning.callbacks import Callback
import os

class LogGenerathings(Callback):


    def __init__(self,log_dir,particle_limit = 30):

        #How many particles will be generated.
        self.particle_limit = particle_limit
        self.log_dir = log_dir

    def on_epoch_end(self, trainer, pl_module,*args,**kwargs):

        current_epoch = trainer.current_epoch
        with open(os.path.join(self.log_dir,"epoch_{}.csv".format(current_epoch)),"w") as fp:
            fp.write("x,y,z,e\n")
            particles = pl_module.generate(self.particle_limit)
            for particle in particles:
                fp.write("{},{},{},{}\n".format(*particle))