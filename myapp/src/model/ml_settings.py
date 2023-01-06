

class MlSettings:
    
    PROJECT_NAME = 'real-state-ny'
    VARIABLE_FILES = False
    #Maximum amount of rows to take
    SAMPLE_COUNT = 20000
    FASTAI_LEARNING_RATE = 1e-1
    AUTO_AJUST_LEARNING_RATE = False
    #Set to True automatically infer variables are categorical or continuous
    ENABLE_BREAKPOINT = True
    #When trying to declare a column as a continuous variable, if it fails, convert to a categorical variable
    CONVERT_CAT = False
    REFRESSOR = True
    SEP_DOLLAR = True
    SEP_PERCENT = True
    SHUFFLE_DATA = True   
    
    
    def __init__(self) -> None:
        pass