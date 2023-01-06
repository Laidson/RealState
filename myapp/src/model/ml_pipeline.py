import os
import shutil
import pandas as pd
from step_01 import MLDataInput 



class Pipeline:    

    def __init__(self) -> None:
        pass
 
    def main(self):   

        #STEP 01
        MLDataInput().main()
        


if __name__ == '__main__':    
    Pipeline().main()