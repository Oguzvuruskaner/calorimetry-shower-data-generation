import numpy as np
from tqdm import tqdm
import re

def loadAndSplitArray(filepath:str,number_of_chunks):

    pattern = re.compile(r"(.*?)\.npy")

    data = np.load(filepath)
    chunks = np.split(data,number_of_chunks)

    rootFilepath = re.findall(pattern,filepath)[0]

    for index,chunk  in enumerate(tqdm(chunks)):

        np.save("{}_chunk_{}.npy".format(rootFilepath,index+1),chunk)



