import pickle

import numpy as np
from keras.models import  Model
from tqdm import tqdm
from config import __MODEL_VERSION__
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os


PARTICLES_MEAN = 139356
PARTICLES_STD = 25077

def test_critic(data,critic,version=__MODEL_VERSION__):

    print("Generating critic plot.")
    critic_results = critic.predict(data,verbose=1)

    sns.distplot(critic_results,kde=False)
    plt.savefig(os.path.join("results","v_{}_critic_result.png".format(version)))
    plt.clf()

def plot_data(data:np.array,plot_title:str,filepath:str,jet = False):

    plt.title(plot_title)

    fig = plt.figure(constrained_layout = True,dpi=500)
    fig.set_size_inches(20, 50)

    if jet:
        grid_spec = fig.add_gridspec(5,2)
    else :
        grid_spec = fig.add_gridspec(3,2)

    for ind,feature in enumerate(["hit_r","hit_z","hit_e"]):
        ax1 = fig.add_subplot(grid_spec[ind,0])
        ax2 = fig.add_subplot(grid_spec[ind,1])

        sns.distplot(data[:,ind],kde=False,ax=ax1)
        ax1.set_title(feature)

        ax2.set_title("{} Stats".format(feature))
        ax2.grid(False)
        ax2.axes.xaxis.set_ticks([])
        ax2.axes.yaxis.set_ticks([])
        ax2.text(0.1,0.5,show_stats(data),clip_on=True,fontsize=24)

    if jet:
        ax = fig.add_subplot(grid_spec[3:,:])
        h = ax.hist2d(x=data[:,0],y=data[:,1],weights=data[:,2],bins=100,norm=LogNorm())
        plt.colorbar(h[3], ax=ax)
        ax.set_title("Jet image")
        ax.set_xlabel("R")
        ax.set_ylabel("Z")

    plt.savefig(filepath,dpi=500)
    plt.clf()

def plot_loss(loss_array,save_path:str):
    # loss_array is an array of triple tuple.
    # [0]: critic real data loss
    # [1]: critic fake data loss
    # [2]: generator loss

    #Taken by https://machinelearningmastery.com/how-to-code-a-wasserstein-generative-adversarial-network-wgan-from-scratch/
    plt.clf()
    plt.plot(loss_array[:,0],label="critic_real")
    plt.plot(loss_array[:,1],label="critic_fake")
    plt.plot(loss_array[:,2],label="generator")
    plt.legend()
    plt.savefig(save_path)
    plt.clf()


def show_stats(results:np.array):

    tmp = ""
    tmp += "Total Entities : {} \n".format(len(results))
    tmp += "Mean : {0:10.3f} \n".format(np.mean(results))
    tmp += "Std : {0:10.3f} \n".format(np.std(results))
    tmp += "Variance : {0:10.3f}\n".format(np.std(results)**2)
    tmp += "Min : {0:10.3f}\n".format(np.min(results))
    tmp += "Max : {0:10.3f}\n".format(np.max(results))
    tmp += ".25 Quantile : {0:10.3f}\n".format(np.quantile(results,.25))
    tmp += ".50 Quantile : {0:10.3f}\n".format(np.quantile(results,.5))
    tmp += ".75 Quantile : {0:10.3f}\n".format(np.quantile(results,.75))
    return tmp

def get_total_particles():
    #For now, generation of number of particles in experiment
    # is done manually.
    return int(np.random.normal(PARTICLES_MEAN,PARTICLES_STD,size=(1,))[0])



def generate_fake_data(generator:Model,visualized_experiments=5,generated_experiments=100,version=__MODEL_VERSION__):


    r_scaler = pickle.load(open("r_scaler.pkl","rb"))
    e_scaler = pickle.load(open("e_scaler.pkl","rb"))
    z_scaler = pickle.load(open("z_scaler.pkl","rb"))

    results = []

    print("Generating visualizations ")
    for i in tqdm(range(visualized_experiments)):
        tmp_results = generator.predict(np.random.normal(0,1,(get_total_particles(),100)))
        tmp_results[:,0] = r_scaler.transform(tmp_results[:,0].reshape(-1,1)).reshape(-1,)
        tmp_results[:,1] = z_scaler.transform(tmp_results[:,1].reshape(-1,1)).reshape(-1,)
        tmp_results[:,2] = e_scaler.transform(tmp_results[:,2].reshape(-1,1)).reshape(-1,)


        plot_data(tmp_results,"r Experiment {}".format(i+1),
                     os.path.join("results","v_{}_result_experiment_{}".format(version,i+1)),jet=True)

        results.extend(tmp_results)


    print("Generating data ")
    for _ in tqdm(range(generated_experiments)):
        tmp_results = generator.predict(np.random.normal(0,1,(get_total_particles(),100)))
        tmp_results[:, 0] = r_scaler.transform(tmp_results[:, 0].reshape(-1, 1)).reshape(-1, )
        tmp_results[:, 1] = z_scaler.transform(tmp_results[:, 1].reshape(-1, 1)).reshape(-1, )
        tmp_results[:, 2] = e_scaler.transform(tmp_results[:, 2].reshape(-1, 1)).reshape(-1, )

        results.extend(tmp_results)

    results = np.array(results)
    plot_data(results,"r Results",
                 os.path.join("results","v_{}_result_all".format(version)))

    np.save(os.path.join("results","v_{}_prediction_array.npy".format(version)),results)
