import os
import pickle
import numpy as np
import seaborn as sns
import math

from PIL import Image

from keras.models import  Model
from tqdm import tqdm, trange
from src.config import __MODEL_VERSION__, DIMENSION
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn.decomposition import PCA


PARTICLES_MEAN = 139356
PARTICLES_STD = 25077

def test_critic(data,critic,version=__MODEL_VERSION__):

    print("Generating critic plot.")
    critic_results = critic.predict(data,verbose=1)

    sns.distplot(critic_results,kde=False)
    plt.savefig(os.path.join("results","v_{}_critic_result.png".format(version)))
    plt.clf()

def plot_data(
        data:np.array,
        plot_title:str,
        filepath:str,
        jet = False,
        jet_bins=100,
        logarithmic_x = False,
        logarithmic_y = False,
        dpi=500
        ):


    fig = plt.figure(constrained_layout = True,dpi=dpi)
    fig.set_size_inches(20, 50)

    if jet:
        grid_spec = fig.add_gridspec(5,2)
    else :
        grid_spec = fig.add_gridspec(3,2)

    fig.suptitle(plot_title, fontsize=72)


    for ind,feature in enumerate(["hit_r","hit_z","hit_e"]):
        ax1 = fig.add_subplot(grid_spec[ind,0])
        ax2 = fig.add_subplot(grid_spec[ind,1])

        sns.distplot(data[:,ind],kde=False,ax=ax1)
        ax1.set_title(feature, fontsize=64)

        ax2.set_title("{} Stats".format(feature), fontsize=64)
        ax2.grid(False)
        ax2.axes.xaxis.set_ticks([])
        ax2.axes.yaxis.set_ticks([])
        ax2.text(0.1,0.5,show_stats(data[:,ind]),clip_on=True,fontsize=48)

    if jet:
        ax = fig.add_subplot(grid_spec[3:,:])
        h = ax.hist2d(x=data[:,0],y=data[:,1],weights=data[:,2],bins=jet_bins,norm=LogNorm())

        colorbar = plt.colorbar(h[3], ax=ax)
        colorbar.ax.set_title("GeV", fontsize=64)

        ax.set_title("Jet image", fontsize=64)
        ax.set_xlabel("R(cm)", fontsize=64)
        ax.set_ylabel("Z(cm)", fontsize=64)

        if logarithmic_x:
            ax.set_xscale("log")

        if logarithmic_y:
            ax.set_yscale("log")

    plt.savefig(filepath,dpi=dpi)
    plt.close(fig)

def plot_jet_generator_train_results(epoch_results:np.array,save_path:str):
    # loss_array is an array of triple tuple.
    # [0]: critic real data loss
    # [1]: critic fake data loss
    # [2]: generator loss

    plt.plot(epoch_results[:, 0], label="critic_real")
    plt.plot(epoch_results[:, 1], label="critic_fake")
    plt.plot(epoch_results[:, 2], label="generator")
    plt.plot(epoch_results[:, 3], label="critic_evaluate")

    plt.legend()
    plt.savefig(save_path)
    plt.clf()

def generate_jet_images(
        generator : Model,
        count=200,
        root_dir=os.path.join("results","jet_images_{}".format(__MODEL_VERSION__)),
        pca : PCA = None
):

    if not os.path.exists(root_dir):
        os.mkdir(root_dir)

    print("Generating jet images.")
    noise_input_size = generator.inputs[0].shape.dims[1]

    results = generator.predict(np.random.normal(size=(count,noise_input_size)))

    if pca:
        results = pca.inverse_transform(results)

    for ind,result in tqdm(enumerate(results,start=1)):
        image = result.reshape((DIMENSION,DIMENSION))

        save_jet_image(image,os.path.join(root_dir, "{}.png".format(ind)))



def save_jet_image(
        image:np.array,
        path:str
):

    plt.imshow(image,cmap="gray",vmax=1,vmin = 0)
    plt.savefig(path)
    plt.close()



def plot_loss(loss_array,save_path:str):
    # loss_array is an array of triple tuple.
    # [0]: critic real data loss
    # [1]: critic fake data loss
    # [2]: generator loss

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
    for i in trange(visualized_experiments):
        tmp_results = generator.predict(np.random.normal(0,1,(get_total_particles(),100)))
        tmp_results[:,0] = r_scaler.transform(tmp_results[:,0].reshape(-1,1)).reshape(-1,)
        tmp_results[:,1] = z_scaler.transform(tmp_results[:,1].reshape(-1,1)).reshape(-1,)
        tmp_results[:,2] = e_scaler.transform(tmp_results[:,2].reshape(-1,1)).reshape(-1,)


        plot_data(tmp_results,"r Experiment {}".format(i+1),
                     os.path.join("results","v_{}_result_experiment_{}".format(version,i+1)),jet=True)

        results.extend(tmp_results)


    print("Generating data ")
    for _ in trange(generated_experiments):
        tmp_results = generator.predict(np.random.normal(0,1,(get_total_particles(),100)))
        tmp_results[:, 0] = r_scaler.transform(tmp_results[:, 0].reshape(-1, 1)).reshape(-1, )
        tmp_results[:, 1] = z_scaler.transform(tmp_results[:, 1].reshape(-1, 1)).reshape(-1, )
        tmp_results[:, 2] = e_scaler.transform(tmp_results[:, 2].reshape(-1, 1)).reshape(-1, )

        results.extend(tmp_results)

    results = np.array(results)
    plot_data(results,"r Results",
                 os.path.join("results","v_{}_result_all".format(version)))

    np.save(os.path.join("results","v_{}_prediction_array.npy".format(version)),results)
