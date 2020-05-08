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

def plot_loss(loss_array,epochs:int,model_name:str,plot_title:str):
    plt.clf()
    plt.ylim((-1., 1.))
    plt.xlim((0, epochs + 1))
    plt.scatter(np.arange(epochs) + 1, loss_array)
    plt.title(plot_title)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.savefig(join("models", model_name))
    plt.clf()

def show_stats(results:np.array):

    tmp = ""
    tmp += "Total Entities : {} \n".format(len(results))
    tmp += "Mean : {0:10.3f} \n".format(results.mean())
    tmp += "Std : {0:10.3f} \n".format(results.std())
    tmp += "Variance : {0:10.3f}\n".format(results.std()**2)
    tmp += "Min : {0:10.3f}\n".format(results.min())
    tmp += "Max : {0:10.3f}\n".format(results.max())
    tmp += ".25 Quantile : {0:10.3f}\n".format(np.quantile(results,.25))
    tmp += ".50 Quantile : {0:10.3f}\n".format(np.quantile(results,.5))
    tmp += ".75 Quantile : {0:10.3f}\n".format(np.quantile(results,.75))
    return tmp

def get_total_particles():
    #For now, generation of number of particles in experiment
    # is done manually.
    return int(np.random.normal(PARTICLES_MEAN,PARTICLES_STD,size=(1,))[0])



def generate_fake_data(generator:Model,visualized_experiments=5,generated_experiments=100,version=__MODEL_VERSION__):


    tmp_data_r = []
    tmp_data_z = []
    tmp_data_e = []

    print("Generating visualizations ")
    for i in tqdm(range(visualized_experiments)):
        results = generator.predict(np.random.normal(0,1,(get_total_particles(),100)))
        results_r = results[:,0]
        results_z = results[:,1]
        results_e = results[:,2]

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
        results_r = results[:, 0]
        results_z = results[:, 1]
        results_e = results[:, 2]

        for i in range(len(results_r)):
            tmp_data_r.append(results_r[i])
            tmp_data_e.append(results_e[i])
            tmp_data_z.append(results_z[i])

    tmp_data_r = np.array(tmp_data_r)
    tmp_data_e = np.array(tmp_data_e)
    tmp_data_z = np.array(tmp_data_z)

    plot_data(tmp_data_r,"r Results",
                 join("results","v_{}_r_result_all".format(version)))
    plot_data(tmp_data_z, "z Results",
                 join("results", "v_{}_z_result_all".format(version)))
    plot_data(tmp_data_e, "e Results",
                     join("results", "v_{}_e_result_all".format(version)))

    np.save(join("results","results_r_array_v{}.npy".format(version)),tmp_data_r)
    np.save(join("results","results_e_array_v{}.npy".format(version)),tmp_data_e)
    np.save(join("results","results_z_array_v{}.npy".format(version)),tmp_data_z)

