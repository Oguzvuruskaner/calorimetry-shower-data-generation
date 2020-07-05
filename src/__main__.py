from src.readRoot import create_jet_images,create_jet_image_array
from src.scripts.pca import write_pca_to_csv
from src.scripts.scripts import get_root_files




if __name__ == "__main__":

    root_files = get_root_files()
    write_pca_to_csv()

