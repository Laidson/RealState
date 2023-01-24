import os 
import sys
sys.path.append(os.path.normpath(os.getcwd() + "/src"))

import shutil
from ml_settings import MlSettings

# Steps
from step_01 import MLDataInput 
from step_02 import TrainModel



class Pipeline:    

    def __init__(self) -> None:
        pass
 
    def main(self):   

        #TODO temporary just to delete everything in the folder
        # dir = f'working/{MlSettings.PROJECT_NAME}/'
        # shutil.rmtree(dir)
        

        #STEP 01 - ML data preparation
        MLDataInput().main()

        #STEP 02 - train|test
        TrainModel().main()

        


if __name__ == '__main__':    
    Pipeline().main()