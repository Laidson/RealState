# Steps
from step_01 import MLDataInput 
from step_02 import TrainModel



class Pipeline:    

    def __init__(self) -> None:
        pass
 
    def main(self):   

        #STEP 01 - ML data preparation
        MLDataInput().main()

        #STEP 02 - train|test
        TrainModel().main()

        


if __name__ == '__main__':    
    Pipeline().main()