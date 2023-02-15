# define some hyper-parameter and some configurations
BATCH_SIZE = 64
NUM_EPOCHS = 20
IMG_SIZE = (112, 112)
SAVE_DIR = './saved_models_anti_face'

SAVED_EPOCH = 5 # save model's weight every 5 epochs
CASIA_DATA_DIR = './data_train_antiface.csv'

LEARNING_RATE = 0.001