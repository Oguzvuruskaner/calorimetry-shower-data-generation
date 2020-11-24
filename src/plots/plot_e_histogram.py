import matplotlib.pyplot as plt
import os


if __name__ == "__main__":

    csv_path = os.path.join("..","..","logs","generathings","epoch_322.csv")
    plot_path = os.path.join("..","plot2.png")

    with open(csv_path,"r") as fp:

        e_values = []

        for line in fp.readlines():
            x,y,z,e = line.split(",")

            e_values.append(e)

        plt.hist(x=e,bins=8)
        plt.xlabel("E (GeV)")
        plt.ylabel("# occurences")
        plt.plot()
        plt.savefig(plot_path)
