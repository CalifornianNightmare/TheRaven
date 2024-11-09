import os

DATA_PATH = os.path.join("data", "input")
MERGED_FILE_PATH = os.path.join("data", "processed", "merged_test.csv")
TEST_FILE_PATH = os.path.join("data", "test.csv")
OUTPUT_PATH = os.path.join("data", "output", "submission.csv")

HYPEROPT_MAX_EVALS = 50
SEED = 42
TEST_SIZE = 0.2
THRESHOLD = 0.3
