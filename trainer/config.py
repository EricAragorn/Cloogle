import tensorflow as tf

USING_GCP = False

# Data
LABEL_PATH = "../parsed_data"
IMG_PATH = "../images"
IMG_DIM = (115, 250)
LABEL_SIZE = 44

COLS = ["jewelNeck", "roundNeck", "turtleNeck", "bardotNeck", "collarNeck", "vNeck", "plungingNeck", "surpliceNeck",
        "halterNeck", "squareNeck", "sweetheartNeck", "asymmetricNeck", "splitNeck", "cowlNeck", "oneshoulderSleeve",
        "offtheshoulderSleeve", "coldshoulderSleeve", "bellSleeve", "raglanSleeve", "dolmanSleeve","balloonSleeve", "capSleeve",
        "kimonoSleeve", "ruffleSleeve", "fittedCut","straightCut","flaredCut","mermaidCut","longSleeve","midSleeve",
        "shortSleeve","capSleeve","sleeveless","spaghettistrapSleeve","straplessSleeve","shortLength","kneeLength",
        "midiLength","maxiLength","asymmetricLength","highlowLength","cropLength","midLength","longLength"]

CATEGORY_COUNT = [14, 10, 4, 7, 9]

# Model
KERNEL_SIZE = (3, 3)
if USING_GCP:
    CNN_INITIALIZER = tf.initializers.truncated_normal(stddev=0.3, mean=0)
else:
    CNN_INITIALIZER = tf.initializers.he_normal()

# Training
BATCH_SIZE = 128
MAX_EPOCH = 100
INITIAL_LR = 5e-6
LR_DECAY_RATE = 0.95
LR_DECAY_EPOCH = 70
START_SAVING = 50