import pickle

import numpy as np
from keras.models import  Model
from tqdm import tqdm
from config import __MODEL_VERSION__
import seaborn as sns
from os.path import join
import matplotlib.pyplot as plt



PARTICLES_MEAN = 139356
PARTICLES_STD = 25077

def test_critic(data,critic,version=__MODEL_VERSION__):

    print("Generating critic plot.")
    critic_results = critic.predict(data)

    plot_data(critic_results, "Critic predictions",
              join("results", "v_{}_critic_result".format(version)))



def plot_data(data:np.array,plot_title:str,filepath:str):
    plt.clf()
    fig,(ax1,ax2) = plt.subplots(1,2)
    sns.distplot(data,kde=False,ax=ax1)
    ax1.set_title(plot_title)
    ax2.set_title("Stats")
    ax2.grid(False)
    ax2.axes.xaxis.set_ticks([])
    ax2.axes.yaxis.set_ticks([])

    ax2.text(0.1,0.5,show_stats(data),clip_on=True)
    plt.savefig(filepath)
    plt.clf()

def plot_loss(loss_array,save_path:str):
    # loss_array is an array of triple tuple.
    # [0]: critic real data loss
    # [1]: critic fake data loss
    # [2]: generator loss
    critic_real_loss_array = []
    critic_fake_loss_array = []
    generator_loss_array = []

    for critic_real_loss,critic_fake_loss,generator_loss in loss_array:
        critic_fake_loss_array.append(critic_fake_loss)
        critic_real_loss_array.append(critic_real_loss)
        generator_loss_array.append(generator_loss)

    #Taken by https://machinelearningmastery.com/how-to-code-a-wasserstein-generative-adversarial-network-wgan-from-scratch/
    plt.clf()
    plt.plot(critic_real_loss_array,label="critic_real")
    plt.plot(critic_fake_loss_array,label="critic_fake")
    plt.plot(generator_loss_array,label="generator")
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

    tmp_data_r = []
    tmp_data_z = []
    tmp_data_e = []

    print("Generating visualizations ")
    for i in tqdm(range(visualized_experiments)):
        results = generator.predict(np.random.normal(0,1,(get_total_particles(),100)))
        results_r = r_scaler.transform(results[:,0].reshape(-1,1)).reshape(-1,)
        results_z = z_scaler.transform(results[:,1].reshape(-1,1)).reshape(-1,)
        results_e = e_scaler.transform(results[:,2].reshape(-1,1)).reshape(-1,)

        for j in range(len(results_r)):
            tmp_data_r.append(results_r[j])
            tmp_data_e.append(results_e[j])
            tmp_data_z.append(results_z[j])


        plot_data(results_r,"r Experiment {}".format(i+1),
                     join("results","v_{}_r_result_experiment_{}".format(version,i+1)))
        plot_data(results_z, "z Experiment {}".format(i + 1),
                     join("results", "v_{}_z_result_experiment_{}".format(version,i+1)))
        plot_data(results_e, "e Experiment {}".format(i + 1),
                     join("results", "v_{}_e_result_experiment_{}".format( version,i+1)))

    print("Generating data ")
    for _ in tqdm(range(generated_experiments)):
        results = generator.predict(np.random.normal(0,1,(get_total_particles(),100)))
        results_r = r_scaler.transform(results[:,0].reshape(-1,1)).reshape(-1,)
        results_z = z_scaler.transform(results[:,1].reshape(-1,1)).reshape(-1,)
        results_e = e_scaler.transform(results[:,2].reshape(-1,1)).reshape(-1,)

        for i in range(len(results_r)):
            tmp_data_r.append(results_r[i])
            tmp_data_e.append(results_e[i])
            tmp_data_z.append(results_z[i])

    tmp_data_r = np.array(tmp_data_r)
    tmp_data_e = np.array(tmp_data_e)
    tmp_data_z = np.array(tmp_data_z)

    plot_data(tmp_data_r.reshape((-1,)),"r Results",
                 join("results","v_{}_r_result_all".format(version)))
    plot_data(tmp_data_z.reshape((-1,)), "z Results",
                 join("results", "v_{}_z_result_all".format(version)))
    plot_data(tmp_data_e.reshape((-1,)), "e Results",
                     join("results", "v_{}_e_result_all".format(version)))

    np.save(join("results","results_r_array_v{}.npy".format(version)),tmp_data_r)
    np.save(join("results","results_e_array_v{}.npy".format(version)),tmp_data_e)
    np.save(join("results","results_z_array_v{}.npy".format(version)),tmp_data_z)

