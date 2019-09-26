import csv
import os
import sys
import traceback

GT_FN = 'DL_info.csv'  # Ground truth file
DIR_IN = 'Images_png'  # input directory

TRAIN_DIR = DIR_IN + os.sep + "train"
VAL_DIR = DIR_IN + os.sep + "val"
TEST_DIR = DIR_IN + os.sep + "test"

if not os.path.exists(TRAIN_DIR):
    os.makedirs(TRAIN_DIR)

if not os.path.exists(VAL_DIR):
    os.makedirs(VAL_DIR)

if not os.path.exists(TEST_DIR):
    os.makedirs(TEST_DIR)


try:
    with open(GT_FN, 'rt') as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)  # skip the headers row
        for row in reader:
            filename = row[0]  # replace the last _ in filename with / or \
            idx = filename.rindex('_')
            file_path = filename[:idx] + os.sep + filename[idx + 1:]
            ds_name = row[17]
            if ds_name == str(1):
                if not os.path.exists(TRAIN_DIR + os.sep + filename[:idx]):
                    os.makedirs(TRAIN_DIR + os.sep + filename[:idx])
                if os.path.exists(DIR_IN + os.sep + file_path):
                    os.rename(DIR_IN + os.sep + file_path, TRAIN_DIR + os.sep + file_path)
                    if len(os.listdir(os.path.join(DIR_IN, filename[:idx]))) == 0:
                        os.rmdir(os.path.join(DIR_IN, filename[:idx]))
            elif ds_name == str(2):
                if not os.path.exists(VAL_DIR + os.sep + filename[:idx]):
                    os.makedirs(VAL_DIR + os.sep + filename[:idx])
                if os.path.exists(DIR_IN + os.sep + file_path):
                    os.rename(DIR_IN + os.sep + file_path, VAL_DIR + os.sep + file_path)
                    if len(os.listdir(os.path.join(DIR_IN, filename[:idx]))) == 0:
                        os.rmdir(os.path.join(DIR_IN, filename[:idx]))
            elif ds_name == str(3):
                if not os.path.exists(TEST_DIR + os.sep + filename[:idx]):
                    os.makedirs(TEST_DIR + os.sep + filename[:idx])
                if os.path.exists(DIR_IN + os.sep + file_path):
                    os.rename(DIR_IN + os.sep + file_path, TEST_DIR + os.sep + file_path)
                    if len(os.listdir(os.path.join(DIR_IN, filename[:idx]))) == 0:
                        os.rmdir(os.path.join(DIR_IN, filename[:idx]))
except FileNotFoundError:
    print(traceback.format_exc())
    sys.exit(1)

