import os

CURRENT_FILE_PATH = os.path.realpath(__file__)
ROOT_FILES_DIR = os.path.join("..","root_files")

DATASETS = {
    1 : [
        {
            "path" : os.path.join(ROOT_FILES_DIR,"RezaAnalysis_1GeV.root") 
        }
    ],
    2 : [
        {
            "path" : os.path.join(ROOT_FILES_DIR,"RezaAnalysis_2GeV.root") 
        }
    ],
    5: [
        {
            "path" : os.path.join(ROOT_FILES_DIR, "RezaAnalysis_5GeV.root")
        }
    ],
    7: [
        {
            "path" : os.path.join(ROOT_FILES_DIR, "RezaAnalysis_7GeV.root")
        }
    ],
    10: [
        {
            "path" : os.path.join(ROOT_FILES_DIR, "RezaAnalysis_10GeV.root")
        }
    ],
    15: [
        {
            "path" : os.path.join(ROOT_FILES_DIR, "RezaAnalysis_15GeV.root")
        }
    ],
    20: [
        {
            "path" : os.path.join(ROOT_FILES_DIR, "RezaAnalysis_20GeV.root")
        }
    ],
    50: [
        {
            "path" : os.path.join(ROOT_FILES_DIR, "RezaAnalysis_50GeV.root")
        }
    ],
}
entries = lambda root : root["g4SimHits"]["eventTree"]