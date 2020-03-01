import os, pickle
import numpy as np
import matplotlib.pyplot as plt

from constant import *


def time_taken(start, end):
    """Human readable time between `start` and `end`
    :param start: time.time()
    :param end: time.time()
    :returns: day:hour:minute:second.millisecond
    """

    my_time = end-start
    day = my_time // (24 * 3600)
    my_time = my_time % (24 * 3600)
    hour = my_time // 3600
    my_time %= 3600
    minutes = my_time // 60
    my_time %= 60
    seconds = my_time
    milliseconds = ((end - start)-int(end - start))
    day_hour_min_sec = str('%02d' % int(day))+":"+str('%02d' % int(hour))+":"+str('%02d' % int(minutes))+":"+str('%02d' % int(seconds)+"."+str('%.3f' % milliseconds)[2:])

    return day_hour_min_sec

def read_dataset_discretized(a_path, db_file_name, temporal_bins):

    bin_file = open(a_path+os.sep+db_file_name, "rb")

    examples = pickle.load(bin_file)
    labels = pickle.load(bin_file)
    bin_file.close()
    
    return examples.reshape(NUM_OF_OBJECTS, TRIALS_PER_OBJECT, CHANNELS, temporal_bins, -1), labels.reshape(NUM_OF_OBJECTS, TRIALS_PER_OBJECT)

def split_train_test(n_folds, num_of_objects):
    test_size = TRIALS_PER_OBJECT//FOLDS

    tt_splits  = {}

    for a_fold in range(FOLDS):

        train_index = []
        test_index = np.arange(test_size*a_fold, test_size*(a_fold+1))

        if test_size*a_fold > 0:
            train_index.extend(np.arange(0, test_size*a_fold))
        if test_size*(a_fold+1)-1 < TRIALS_PER_OBJECT-1:
            train_index.extend(np.arange(test_size*(a_fold+1), TRIALS_PER_OBJECT))

        tt_splits.setdefault("fold_" + str(a_fold), {}).setdefault("train", []).extend(train_index)
        tt_splits.setdefault("fold_" + str(a_fold), {}).setdefault("test", []).extend(test_index)
    
    return tt_splits

