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
from matplotlib.lines import Line2D

from utils import read_dataset_discretized, split_train_test
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


path = r"Datasets"

CLF = SVC(gamma='auto', kernel='rbf', probability=True)
CLF_NAME = "SVM-RBF"


ALL_ROBOTS_LIST = ["baxter", "fetch", "sawyer"]

#A_TARGET_ROBOT = "baxter" #always 1 robot
A_TARGET_ROBOT = "fetch" #always 1 robot
#A_TARGET_ROBOT = "sawyer" #always 1 robot

A_TARGET_ROBOT_DATATYPE = "discretizedmean-10"

SOURCE_ROBOT_LIST = [] #always 2 robots
for a_robot in ALL_ROBOTS_LIST:
    if a_robot != A_TARGET_ROBOT:
        SOURCE_ROBOT_LIST.append(a_robot)
SOURCE_ROBOT_DATATYPE = ["discretizedmean-10", "discretizedmean-10"]

BEHAVIOR_LIST = ["place"]
#BEHAVIOR_LIST = ["grasp", "pick"]
#BEHAVIOR_LIST = ["grasp", "pick", "place", "shake"]

MY_COLORS = ["orangered", "olive", "beige", "black", "chartreuse", "blue", "brown", "coral", "crimson", "cyan",
             "darkblue", "darkgreen", "fuchsia", "gold", "green", "grey", "indigo", "orange", "khaki", "orchid",
             "steelblue", "chocolate", "lightblue", "magenta", "maroon"]

i = 18 #for effort

RUNS = 2 #10

if A_TARGET_ROBOT == "baxter":
    SOURCE_DATA_PERCENT = 10  # 1 to 100
elif A_TARGET_ROBOT == "fetch":
    SOURCE_DATA_PERCENT = 30  # 1 to 100
elif A_TARGET_ROBOT == "sawyer":
    SOURCE_DATA_PERCENT = 10  # 1 to 100

new_lables = np.arange(1, NUM_OF_OBJECTS+1) # all 25 lables
NUM_OF_OBJECTS = len(new_lables)

NUM_OF_NOVEL_OBJECTS = 2 # Min: 2, Max: NUM_OF_OBJECTS-1
TEST_ON_NON_TRAIN = False  # True: to test on all other non-train objects
if TEST_ON_NON_TRAIN:
    NUM_OF_NOVEL_OBJECTS = 2

num_of_KEMA_features = 1

print_plots = False

MATLAB_eng = matlab.engine.start_matlab()

data_path_KEMA = r"Results"+os.sep+"Novel"
input_filename_KEMA = 'data_'+A_TARGET_ROBOT
output_filename_KEMA = 'projections_'+A_TARGET_ROBOT

os.makedirs(data_path_KEMA, exist_ok=True)
os.makedirs(data_path_KEMA+os.sep+A_TARGET_ROBOT+"_IE", exist_ok=True)


def get_train_data_from_source_robot(Z, Y, source_data_percent, test_objects):
    """
    Get train data of source robot for novel (test) objects
    """
    Z_train_test_obj = Z.reshape(NUM_OF_OBJECTS, source_data_percent, -1)
    Y_train_test_obj = Y.reshape(NUM_OF_OBJECTS, source_data_percent, -1)
    # Get train data
    Z_train = []
    Y_train = []
    for a_object in test_objects:
        Z_train.extend(Z_train_test_obj[a_object])
        Y_train.extend(Y_train_test_obj[a_object])
    Z_train_test_obj = np.array(Z_train)
    Y_train_test_obj = np.array(Y_train).reshape((-1, 1))
    
    return Z_train_test_obj, Y_train_test_obj


"""
Getting source data ready
"""
source_robots_data = {}
new_lables_old_lables = {}
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
            new_lables_old_lables.setdefault(label_count, a_label+1)
            label_count += 1
        examples = np.array(examples_new)
        labels = np.array(labels_new)
        
        data_temp = examples[:, 0:SOURCE_DATA_PERCENT, :, :, i:i+FEATURES_PER_MODALITY]
        data_temp = data_temp.reshape((-1, data_temp.shape[-2]*data_temp.shape[-1]))
        
        print("db_file_name: ", db_file_name, data_temp.shape)
                
        labels_temp = labels[:, 0:SOURCE_DATA_PERCENT].reshape((-1, 1)) #+ 1 # adding 1 because in KEMA (MATLAB) labels starts from 1
        
        robot.setdefault(a_behavior, {})
        robot[a_behavior]["examples"] = data_temp
        robot[a_behavior]["labels"] = labels_temp
    
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


# In[ ]:


all_target_objects_scores = []
all_target_objects_scores_std = []
all_target_objects_relative_errors = []
all_target_objects_relative_errors_std = []
all_target_objects_ordinal_errors = []
all_target_objects_ordinal_errors_std = []
all_target_objects_weight_differance = []
all_target_objects_weight_differance_std = []


for no_of_target_objects in np.arange(NUM_OF_OBJECTS-NUM_OF_NOVEL_OBJECTS, 0, -1):
    print("no_of_target_objects: ", no_of_target_objects)

    all_runs_scores = []
    for a_run in range(1, RUNS + 1):
        print("a_run: ", a_run)
        train_target_objects = np.random.choice(NUM_OF_OBJECTS, size=no_of_target_objects, replace=False)
        print("train_target_objects: ", train_target_objects + 1)
        test_target_objects = np.array([xi for xi in np.arange(NUM_OF_OBJECTS) if xi not in train_target_objects])
        if not TEST_ON_NON_TRAIN:
            test_target_objects = test_target_objects[0:NUM_OF_NOVEL_OBJECTS] # Always test on random NUM_OF_NOVEL_OBJECTS objects
        print("test_target_objects: ", test_target_objects + 1)
                
        # For each behavior, combine weighted probability based on its accuracy score
        behavior_scores = {}
        behavior_proba = {}
        for a_behavior in BEHAVIOR_LIST:
            print("a_behavior: ", a_behavior)

            y_test_dict = {}
            for yi in range(len(test_target_objects)):
                y_test_dict[yi + 1] = test_target_objects[yi] + 1 # adding 1 because in KEMA (MATLAB) labels starts from 1

            # Get test data
            X_test = []
            y_test = []
            for a_object in test_target_objects:
                X_test.extend(target_robots_data[a_behavior]["examples"][a_object])
                y_test.extend(target_robots_data[a_behavior]["labels"][a_object])
            X_test_haptic = np.array(X_test)
            X_test_haptic = X_test_haptic[:, :, :, i:i+FEATURES_PER_MODALITY] # Examples, Channels, Temporal Bins, Features
            X_test_haptic = X_test_haptic.reshape((-1, X_test_haptic.shape[-2]*X_test_haptic.shape[-1]))
            y_test = np.array(y_test).reshape((-1, 1))

            # Get train data
            X_train = []
            y_train = []
            for a_object in train_target_objects:
                X_train.extend(target_robots_data[a_behavior]["examples"][a_object])
                y_train.extend(target_robots_data[a_behavior]["labels"][a_object])
            X_train_haptic = np.array(X_train)
            X_train_haptic = X_train_haptic[:, :, :, i:i+FEATURES_PER_MODALITY] # Examples, Channels, Temporal Bins, Features
            X_train_haptic = X_train_haptic.reshape((-1, X_train_haptic.shape[-2]*X_train_haptic.shape[-1]))
            y_train = np.array(y_train).reshape((-1, 1))

            KEMA_data = {'X3':X_train_haptic, 'Y3':y_train}

            count = 1
            for a_source_robot in source_robots_data:
                KEMA_data['X'+str(count)] = source_robots_data[a_source_robot][a_behavior]["examples"]
                KEMA_data['Y'+str(count)] = source_robots_data[a_source_robot][a_behavior]["labels"]
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

            # Classifier should be trained only on source robot's examples of the novel objects
            Z1_train_test_obj, Y1_train_test_obj = get_train_data_from_source_robot(Z1_train, KEMA_data['Y1'], SOURCE_DATA_PERCENT, test_target_objects)
            Z2_train_test_obj, Y2_train_test_obj = get_train_data_from_source_robot(Z2_train, KEMA_data['Y2'], SOURCE_DATA_PERCENT, test_target_objects)

            Z1_Z2_train_haptic = np.concatenate((Z1_train_test_obj, Z2_train_test_obj), axis=0)
            y1_y2_train = np.concatenate((Y1_train_test_obj, Y2_train_test_obj), axis=0)
            
            # Train and Test
            if num_of_KEMA_features:
                y_acc, y_pred, y_proba = classifier(CLF, Z1_Z2_train_haptic[:, 0:num_of_KEMA_features], Z3_test[:, 0:num_of_KEMA_features], y1_y2_train.ravel(), y_test.ravel())
            else:
                y_acc, y_pred, y_proba = classifier(CLF, Z1_Z2_train_haptic, Z3_test, y1_y2_train.ravel(), y_test.ravel())
            
            y_proba_pred = np.argmax(y_proba, axis=1) + 1 # adding 1 because in KEMA (MATLAB) labels starts from 1

            y_proba_pred_corrected = []
            for a_y in y_proba_pred:
                y_proba_pred_corrected.append(y_test_dict[a_y])

            y_prob_acc = np.mean(y_test.ravel() == y_proba_pred_corrected)

            behavior_scores[a_behavior] = y_prob_acc # Save accuracy
            behavior_proba[a_behavior] = y_proba * y_prob_acc # weighted probability
            
            #################################################################
            if print_plots:
                fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))
                fig.suptitle(a_behavior+', Accuracy: '+str(round(y_prob_acc*100, 1))+'%', fontsize='20')
                
                legend_elements = []
                legend_elements.append(Line2D([0], [0], marker='s', color='w', label='Source (Train)', markerfacecolor='k', markersize=12))
                legend_elements.append(Line2D([0], [0], marker='o', color='w', label='Target (Test: True Labels)', markerfacecolor='k', markersize=12))

                for obj_lab in sorted(test_target_objects):
                    obj_lab += 1
                    obj_name = OBJECT_LABELS_WEIGHTS[new_lables_old_lables[obj_lab]]
                    indices = np.where(Y1_train_test_obj[:, 0] == obj_lab)
                    
                    ax.scatter(Z1_train_test_obj[indices, 0], Z1_train_test_obj[indices, 1],
                               c=np.repeat(MY_COLORS[obj_lab-1], SOURCE_DATA_PERCENT), s=100, edgecolor='red', marker='s')
                    ax.scatter(Z2_train_test_obj[indices, 0], Z2_train_test_obj[indices, 1],
                               c=np.repeat(MY_COLORS[obj_lab-1], SOURCE_DATA_PERCENT), s=100, edgecolor='blue', marker='s')
                    indices = np.where(y_test[:, 0] == obj_lab)
                    ax.scatter(Z3_test[indices, 0], Z3_test[indices, 1],
                               c=np.repeat(MY_COLORS[obj_lab-1], TRIALS_PER_OBJECT), s=100, edgecolor='black', marker='o', label=str(obj_lab)+" ("+str(obj_name)+" kg)")
                
                legend1 = ax.legend(loc='upper left', title="Object Class (weight)", fontsize=14, bbox_to_anchor=(1, 1))
                legend2 = ax.legend(handles=legend_elements, fontsize=14, loc='upper right')
                ax.add_artist(legend1)
                ax.add_artist(legend2)
                xmin, xmax, ymin, ymax = ax.axis()
                file_name = data_path_KEMA+os.sep+A_TARGET_ROBOT+"_IE/"+a_behavior+"_"+str(len(train_target_objects))+"Train_"+str(len(test_target_objects))+"Test_"+str(a_run)+"Run_"+str(round(y_prob_acc*100, 1))+"%.png"
                plt.savefig(file_name, bbox_inches='tight', dpi=100)
                plt.show(block=True)
                

                fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))
                fig.suptitle(a_behavior+', Accuracy: '+str(round(y_prob_acc*100, 1))+'%', fontsize='20')
                
                legend_elements = []
                legend_elements.append(Line2D([0], [0], marker='o', color='w', label='Target (Test: Predicted Labels)', markerfacecolor='k', markersize=12))

                for obj_lab in sorted(test_target_objects):
                    obj_lab += 1
                    obj_name = OBJECT_LABELS_WEIGHTS[new_lables_old_lables[obj_lab]]
                    indices = np.where(y_test[:, 0] == obj_lab)
                    
                    color_codes = []
                    for indx in indices[0]:
                        color_codes.append(MY_COLORS[y_proba_pred_corrected[indx]-1])
                    
                    ax.scatter(Z3_test[indices, 0], Z3_test[indices, 1],
                               c=color_codes, s=100, edgecolor='black', marker='o', label=str(obj_lab)+" ("+str(obj_name)+" kg)")
                    
                    
                legend1 = ax.legend(loc='upper left', title="Object Class (weight)", fontsize=14, bbox_to_anchor=(1, 1))
                legend2 = ax.legend(handles=legend_elements, fontsize=14, loc='upper right')
                ax.add_artist(legend1)
                ax.add_artist(legend2)
                ax.set_ylim([ymin, ymax])
                ax.set_xlim([xmin, xmax])
                file_name = data_path_KEMA+os.sep+A_TARGET_ROBOT+"_IE/"+a_behavior+"_"+str(len(train_target_objects))+"Train_"+str(len(test_target_objects))+"Test_"+str(a_run)+"Run_"+str(round(y_prob_acc*100, 1))+"%_prediction.png"
                plt.savefig(file_name, bbox_inches='tight', dpi=100)
                plt.show(block=True)
            #################################################################


        # Combine weighted probability of all behaviors
        y_proba_norm = np.zeros(behavior_proba[a_behavior].shape)
        for a_behavior in BEHAVIOR_LIST:
            y_proba_norm = y_proba_norm + behavior_proba[a_behavior]

        # Normalizing probability
        y_proba_norm_sum = np.sum(y_proba_norm, axis=1)
        y_proba_norm_sum = np.repeat(y_proba_norm_sum, len(test_target_objects), axis=0).reshape(y_proba_norm.shape)
        y_proba_norm = y_proba_norm/y_proba_norm_sum

        y_proba_pred = np.argmax(y_proba_norm, axis=1) + 1 # adding 1 because in KEMA (MATLAB) labels starts from 1

        y_proba_pred_corrected = []
        for a_y in y_proba_pred:
            y_proba_pred_corrected.append(y_test_dict[a_y])
        y_proba_pred_corrected = np.array(y_proba_pred_corrected)

        y_prob_acc = np.mean(y_test.ravel() == y_proba_pred_corrected)
        behavior_scores["All behaviors"] = y_prob_acc
        print("All behaviors Accuracy: ", y_prob_acc)
        all_runs_scores.append(y_prob_acc)
        print("")

    print("Mean Accuracy: ", np.mean(all_runs_scores), np.std(all_runs_scores))
    
    all_target_objects_scores.append(np.mean(all_runs_scores))
    all_target_objects_scores_std.append(np.std(all_runs_scores))
    
    print("")

# Save results
db_file_name = data_path_KEMA+os.sep+A_TARGET_ROBOT+"_"+CLF_NAME+"_Novel.bin"
output_file = open(db_file_name, "wb")
pickle.dump(A_TARGET_ROBOT, output_file)
pickle.dump(all_target_objects_scores, output_file)
pickle.dump(all_target_objects_scores_std, output_file)
pickle.dump(NUM_OF_OBJECTS, output_file)
pickle.dump(NUM_OF_NOVEL_OBJECTS, output_file)
pickle.dump(TEST_ON_NON_TRAIN, output_file)
output_file.close()


# Read results
db_file_name = data_path_KEMA+os.sep+A_TARGET_ROBOT+"_"+CLF_NAME+"_novel.bin"

bin_file = open(db_file_name, "rb")
A_TARGET_ROBOT = pickle.load(bin_file)
all_target_objects_scores = pickle.load(bin_file)
all_target_objects_scores_std = pickle.load(bin_file)
NUM_OF_OBJECTS = pickle.load(bin_file)
NUM_OF_NOVEL_OBJECTS = pickle.load(bin_file)
TEST_ON_NON_TRAIN = pickle.load(bin_file)
bin_file.close()


target_objects = np.arange(1, NUM_OF_OBJECTS-NUM_OF_NOVEL_OBJECTS+1) # 1 to 10

if not TEST_ON_NON_TRAIN:
    novel_objects = np.repeat(NUM_OF_NOVEL_OBJECTS, len(target_objects))
    novel_objects_baseline = 100 / novel_objects
    print("novel_objects_baseline: ", novel_objects_baseline)
else:
    novel_objects = np.arange(NUM_OF_OBJECTS-1, 1, -1) # 11 to 2
    novel_objects_baseline = 100 / novel_objects

all_target_objects_scores = np.flip(all_target_objects_scores) * 100
all_target_objects_scores_std = np.flip(all_target_objects_scores_std) * 100

plt.errorbar(x=target_objects, y=all_target_objects_scores, yerr=all_target_objects_scores_std, fmt='-o', label="Accuracy", color='blue')
plt.plot(target_objects, novel_objects_baseline, '--', label="Chance Accuracy (baseline)", color='pink')
plt.xticks(target_objects)
plt.ylim(0, 110)
plt.yticks(np.arange(0, 110, 10))
plt.legend(loc='upper left')
# plt.title(str(NUM_OF_NOVEL_OBJECTS)+" novel objects accuracy curve", fontsize=15)
plt.ylabel("% Recognition Accuracy", fontsize=14)
plt.xlabel("No. of objects explored by target robot", fontsize=14)
file_name = data_path_KEMA+os.sep+A_TARGET_ROBOT+"_accuracy_novel"
plt.savefig(file_name, bbox_inches='tight', dpi=100)
#plt.show()
