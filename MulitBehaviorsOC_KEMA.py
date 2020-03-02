#!/usr/bin/env python

import os
import pickle
import time
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import matlab.engine

from sklearn.svm import SVC
from scipy.io import loadmat
from scipy import interpolate

from utils import read_dataset_discretized, split_train_test, time_taken
from constant import *


def classifier(my_classifier, x_train_temp, x_test_temp, y_train_temp, y_test_temp):
    """
    Train a classifier on test data and return accuracy and prediction on test data
    :param my_classifier:
    :param x_train_temp:
    :param x_test_temp:
    :param y_train_temp:
    :param y_test_temp:
    :return: accuracy, prediction
    """
    # Fit the model on the training data.
    my_classifier.fit(x_train_temp, y_train_temp)

    # See how the model performs on the test data.
    accuracy = my_classifier.score(x_test_temp, y_test_temp)
    prediction = my_classifier.predict(x_test_temp)
    probability = my_classifier.predict_proba(x_test_temp)

    return accuracy, prediction, probability


# Combine Multiple behaviors, using Latent effort features from KEMA and SVM as classifier
# 2 source robots, 1 target robot

path = r"Datasets"

CLF = SVC(gamma='auto', kernel='rbf', probability=True)
CLF_NAME = "SVM-RBF"

ALL_ROBOTS_LIST = ["baxter", "sawyer", "fetch"]

#A_TARGET_ROBOT = "baxter" #always 1 robot
A_TARGET_ROBOT = "fetch" #always 1 robot
#A_TARGET_ROBOT = "sawyer" #always 1 robot

#A_TARGET_ROBOT_DATATYPE = "discretizedmean-10"
A_TARGET_ROBOT_DATATYPE = "discretizedrange-15"

SOURCE_ROBOT_LIST = [] #always 2 robots
for a_robot in ALL_ROBOTS_LIST:
    if a_robot != A_TARGET_ROBOT:
        SOURCE_ROBOT_LIST.append(a_robot)
#SOURCE_ROBOT_DATATYPE = ["discretizedmean-10", "discretizedmean-10"]
SOURCE_ROBOT_DATATYPE = ["discretizedrange-15", "discretizedrange-15"]

# BEHAVIOR_LIST = ["pick", "place"]
BEHAVIOR_LIST = ["grasp", "pick", "place", "shake"]

# NO_OF_INTERACTIONS = [1, 40, 80]
NO_OF_INTERACTIONS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80]
#NO_OF_INTERACTIONS = range(1, len(TRAIN_TEST_SPLITS["fold_0"]["train"]))

TRAIN_TEST_SPLITS = split_train_test(FOLDS, TRIALS_PER_OBJECT)
i = 18  # for effort
new_lables = np.arange(1, NUM_OF_OBJECTS+1) # all 25 lables
NUM_OF_OBJECTS = len(new_lables)

KEMA_PARAMETERS_ROBOTS = {'baxter':{'source_per':[10, 5, 5], 'kema_fea':[1, 1, 1]},
                          'fetch':{'source_per':[30, 5, 5], 'kema_fea':[1, 1, 1]},
                          'sawyer':{'source_per':[10, 5, 5], 'kema_fea':[1, 2, 2]}
                         }

MATLAB_eng = matlab.engine.start_matlab()

data_path_KEMA = 'Results'+os.sep+"Transfer"
input_filename_KEMA = 'data_'+A_TARGET_ROBOT
output_filename_KEMA = 'projections_'+A_TARGET_ROBOT

os.makedirs(data_path_KEMA, exist_ok=True)


"""
Getting source data ready
"""

def get_source_data_resized(data, labels, sr_i, source_data_percent):
    
    data_temp = data[:, 0:source_data_percent, :, :, i:i+FEATURES_PER_MODALITY]
    data_temp = data_temp.reshape((-1, data_temp.shape[-2]*data_temp.shape[-1]))

    labels_temp = labels[:, 0:source_data_percent].reshape((-1, 1)) #+ 1 # adding 1 because in KEMA (MATLAB) labels starts from 1

    return data_temp, labels_temp


source_robots_data = {}
print("Source Robots")
for sr_i, a_source_robot in enumerate(SOURCE_ROBOT_LIST):
    # Read and save the examples for each behavior
    robot = {}
    for a_behavior in BEHAVIOR_LIST:
        db_file_name = a_source_robot+"_"+a_behavior+"_"+SOURCE_ROBOT_DATATYPE[sr_i]+".bin"

        robot_name, behavior_name, dataset_type = db_file_name.split('_')[0].capitalize(), db_file_name.split('_')[1].capitalize(), db_file_name.split('_')[2][:-4].capitalize()
        temporal_bins = int(dataset_type.split('-')[1])

        examples, labels = read_dataset_discretized(path, db_file_name, temporal_bins)

        examples_new = []
        labels_new = []
        label_count = 1
        for a_label in new_lables:
            a_label -= 1
            examples_new.append(examples[a_label])
            labels_new.append(np.repeat(label_count, len(examples[a_label])))
            label_count += 1
        examples = np.array(examples_new)
        print("db_file_name: ", db_file_name, examples.shape)
        labels = np.array(labels_new)

        robot.setdefault(a_behavior, {})
        robot[a_behavior]["examples"] = examples
        robot[a_behavior]["labels"] = labels
    
    source_robots_data[a_source_robot] = robot
    

"""
Reading target data 
"""
target_robots_data = {}
print("Target Robot")
for a_behavior in BEHAVIOR_LIST:
    db_file_name = A_TARGET_ROBOT+"_"+a_behavior+"_"+A_TARGET_ROBOT_DATATYPE+".bin"
    
    robot_name, behavior_name, dataset_type = db_file_name.split('_')[0].capitalize(), db_file_name.split('_')[1].capitalize(), db_file_name.split('_')[2][:-4].capitalize()
    temporal_bins = int(dataset_type.split('-')[1])

    examples, labels = read_dataset_discretized(path, db_file_name, temporal_bins)
    
    examples_new = []
    labels_new = []
    label_count = 1
    for a_label in new_lables:
        a_label -= 1
        examples_new.append(examples[a_label])
        labels_new.append(np.repeat(label_count, len(examples[a_label])))
        label_count += 1
    examples = np.array(examples_new)
    labels = np.array(labels_new)
        
    print("db_file_name: ", db_file_name, examples.shape)

    target_robots_data.setdefault(a_behavior, {})
    target_robots_data[a_behavior]["examples"] = np.array(examples)
    target_robots_data[a_behavior]["labels"] = np.array(labels)


f_source_per = interpolate.interp1d([1, 40, 80], KEMA_PARAMETERS_ROBOTS[A_TARGET_ROBOT]['source_per'])
f_kema_fea = interpolate.interp1d([1, 40, 80], KEMA_PARAMETERS_ROBOTS[A_TARGET_ROBOT]['kema_fea'])

# Writing log file for execution time
with open(data_path_KEMA+os.sep+A_TARGET_ROBOT+"_"+CLF_NAME+'_KEMA_time_log.txt', 'w') as file:
    file.write('Time Log\n')
    file.write("\nsource_robot_list: " + str(SOURCE_ROBOT_LIST))
    file.write("\nsource_robot_datatype: " + str(SOURCE_ROBOT_DATATYPE))
    file.write("\na_target_robot: " + A_TARGET_ROBOT)
    file.write("\na_target_robot_datatype: " + A_TARGET_ROBOT_DATATYPE)
    file.write("\nbehavior_list: " + str(BEHAVIOR_LIST))
    file.write("\nno_of_interactions: " + str(NO_OF_INTERACTIONS))
    file.write("\nCLF_NAME: " + CLF_NAME)
    main_start_time = time.time()

# For each fold
all_fold_scores = {}
for a_fold in sorted(TRAIN_TEST_SPLITS):
    print(a_fold)
    start_time = time.time()
    
    # For each no. of example
    all_behavior_scores = {}
    for example_per_objects in NO_OF_INTERACTIONS:
        # For each behavior, combine weighted probability based on its accuracy score
        behavior_proba = {}
        for a_behavior in BEHAVIOR_LIST:
            # Get test data
            X_test = []
            y_test = []
            for a_object in range(NUM_OF_OBJECTS):
                X_test.extend(target_robots_data[a_behavior]["examples"][a_object][TRAIN_TEST_SPLITS[a_fold]["test"]])
                y_test.extend(target_robots_data[a_behavior]["labels"][a_object][TRAIN_TEST_SPLITS[a_fold]["test"]])
            X_test_haptic = np.array(X_test)
            X_test_haptic = X_test_haptic[:, :, :, i:i+FEATURES_PER_MODALITY] # Examples, Channels, Temporal Bins, Features
            X_test_haptic = X_test_haptic.reshape((-1, X_test_haptic.shape[-2]*X_test_haptic.shape[-1]))
            y_test = np.array(y_test).reshape((-1, 1)) #+ 1 # adding 1 because in KEMA (MATLAB) labels starts from 1

            # Get train data
            X_train = []
            y_train = []
            for a_object in range(NUM_OF_OBJECTS):
                X_train.extend(target_robots_data[a_behavior]["examples"][a_object][TRAIN_TEST_SPLITS[a_fold]["train"][0:example_per_objects]])
                y_train.extend(target_robots_data[a_behavior]["labels"][a_object][TRAIN_TEST_SPLITS[a_fold]["train"][0:example_per_objects]])
            X_train_haptic = np.array(X_train)
            X_train_haptic = X_train_haptic[:, :, :, i:i+FEATURES_PER_MODALITY] # Examples, Channels, Temporal Bins, Features
            X_train_haptic = X_train_haptic.reshape((-1, X_train_haptic.shape[-2]*X_train_haptic.shape[-1]))
            y_train = np.array(y_train).reshape((-1, 1)) #+ 1 # adding 1 because in KEMA (MATLAB) labels starts from 1

            KEMA_data = {'X3':X_train_haptic, 'Y3':y_train}
            
            source_data_percent = round(float(f_source_per(example_per_objects)))
            num_of_kema_features = round(float(f_kema_fea(example_per_objects)))
            
            count = 1
            #for a_source_robot in source_robots_data:
            for sr_i, a_source_robot in enumerate(SOURCE_ROBOT_LIST):
                data_temp, labels_temp = get_source_data_resized(source_robots_data[a_source_robot][a_behavior]["examples"],
                                                                 source_robots_data[a_source_robot][a_behavior]["labels"],
                                                                 sr_i, source_data_percent)
                KEMA_data['X'+str(count)] = data_temp
                KEMA_data['Y'+str(count)] = labels_temp                
                count += 1
                        
            KEMA_data['X3_Test'] = X_test_haptic
            
            scipy.io.savemat(os.path.join(data_path_KEMA, input_filename_KEMA), mdict=KEMA_data)
            
            MATLAB_eng.call_project3Domains(data_path_KEMA, input_filename_KEMA, output_filename_KEMA, 1)
            
            # In case Matlab messes up, we'll load and check these immediately, then delete them so we never read in an old file
            projections = None
            if os.path.isfile(os.path.join(data_path_KEMA, output_filename_KEMA+".mat")):
                try:
                    projections = loadmat(os.path.join(data_path_KEMA, output_filename_KEMA+".mat"))
                    Z1_train, Z2_train, Z3_train, Z3_test = projections['Z1'], projections['Z2'], projections['Z3'], projections['Z3_Test']
                    os.remove(os.path.join(data_path_KEMA, input_filename_KEMA+".mat"))
                    os.remove(os.path.join(data_path_KEMA, output_filename_KEMA+".mat"))
                except TypeError as e:
                    print('loadmat failed: ' + str(e))
            else:
                print("projections.mat not exist !! - 1" )
                break

            X_train_haptic = np.concatenate((Z1_train, Z2_train, Z3_train), axis=0)
            y_train = np.concatenate((KEMA_data['Y1'], KEMA_data['Y2'], y_train), axis=0)

            # Train and Test
            if num_of_kema_features:
                y_acc, y_pred, y_proba = classifier(CLF, X_train_haptic[:, 0:num_of_kema_features], Z3_test[:, 0:num_of_kema_features], y_train.ravel(), y_test.ravel())
            else:
                y_acc, y_pred, y_proba = classifier(CLF, X_train_haptic, Z3_test, y_train.ravel(), y_test.ravel())
            
            y_proba_pred = np.argmax(y_proba, axis=1) + 1 # adding 1 because in KEMA (MATLAB) labels starts from 1
            y_prob_acc = np.mean(y_test.ravel() == y_proba_pred)

            all_behavior_scores.setdefault(a_behavior, [])
            all_behavior_scores[a_behavior].append(y_prob_acc)
            
            # For each behavior, get an accuracy score to combine weighted probability based on its accuracy score
            # Use only training data to get a score
            y_acc_train, y_pred_train, y_proba_train = classifier(CLF, X_train_haptic, X_train_haptic, y_train.ravel(), y_train.ravel())
            y_proba_pred_train = np.argmax(y_proba_train, axis=1) + 1 # In dataset v3, labels starts from 1
            y_prob_acc_train = np.mean(y_train == y_proba_pred_train)

            if y_prob_acc_train > 0:
                # Multiple the score on training data by probability of test data to combine each
                # behavior's performance accordingly
                y_proba = y_proba * y_prob_acc_train # weighted probability
            behavior_proba[a_behavior] = y_proba
            
        # Combine weighted probability of all behaviors
        y_proba_norm = np.zeros(behavior_proba[a_behavior].shape)
        for a_behavior in BEHAVIOR_LIST:
            y_proba_norm = y_proba_norm + behavior_proba[a_behavior]

        # Normalizing probability
        y_proba_norm_sum = np.sum(y_proba_norm, axis=1)
        y_proba_norm_sum = np.repeat(y_proba_norm_sum, NUM_OF_OBJECTS, axis=0).reshape(y_proba_norm.shape)
        y_proba_norm = y_proba_norm/y_proba_norm_sum

        y_proba_pred = np.argmax(y_proba_norm, axis=1) + 1 # adding 1 because in KEMA (MATLAB) labels starts from 1
        y_prob_acc = np.mean(y_test.ravel() == y_proba_pred)
        all_behavior_scores.setdefault("all_behavior", [])
        all_behavior_scores["all_behavior"].append(y_prob_acc)
        
        print("example_per_objects, source_data_percent, num_of_kema_features, y_prob_acc",
              example_per_objects, source_data_percent, num_of_kema_features, y_prob_acc)
    
    all_fold_scores[a_fold] = all_behavior_scores # Saving scores of each fold
        
    # Writing log file for execution time
    file = open(data_path_KEMA+os.sep+A_TARGET_ROBOT+"_"+CLF_NAME+'_KEMA_time_log.txt', 'a')  # append to the file created
    end_time = time.time()
    file.write("\n"+a_fold+" Time: " + time_taken(start_time, end_time))
    file.close()

# Taking average of all folds
# Saving accuracy of each folds
all_scores = {}
for a_fold in sorted(TRAIN_TEST_SPLITS):
    for a_behavior in BEHAVIOR_LIST:
        all_scores.setdefault(a_behavior, [])
        all_scores[a_behavior].append(all_fold_scores[a_fold][a_behavior])
    all_scores.setdefault("All behaviors", [])
    all_scores["All behaviors"].append(all_fold_scores[a_fold]["all_behavior"])

# Computing mean of each folds and plotting
for a_behavior in sorted(all_scores):    
    all_scores[a_behavior] = np.array(all_scores[a_behavior])
    all_scores[a_behavior] = np.mean(all_scores[a_behavior], axis=0)

    plt.plot(NO_OF_INTERACTIONS, all_scores[a_behavior], label=a_behavior.capitalize())

plt.xlabel('No. of training interaction per object')
plt.ylabel('Accuracy ('+str(NUM_OF_OBJECTS)+' Objects)')
plt.title(A_TARGET_ROBOT.capitalize()+': Accuracy Curve using latent features (KEMA)')
plt.legend()
plt.savefig(data_path_KEMA+os.sep+A_TARGET_ROBOT+"_"+CLF_NAME+"_KEMA.png", bbox_inches='tight', dpi=100)
#plt.show()
plt.close()

# Save results
db_file_name = data_path_KEMA+os.sep+A_TARGET_ROBOT+"_"+CLF_NAME+"_KEMA.bin"
output_file = open(db_file_name, "wb")
pickle.dump(all_scores, output_file)
pickle.dump(NO_OF_INTERACTIONS, output_file)
pickle.dump(A_TARGET_ROBOT, output_file)
pickle.dump(NUM_OF_OBJECTS, output_file)
pickle.dump(CLF_NAME, output_file)
output_file.close()

file = open(data_path_KEMA+os.sep+A_TARGET_ROBOT+"_"+CLF_NAME+'_KEMA_time_log.txt', 'a')  # append to the file created
end_time = time.time()
file.write("\nTotal Time: " + time_taken(main_start_time, end_time))
file.close()
