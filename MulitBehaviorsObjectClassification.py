#!/usr/bin/env python

import os, pickle, csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

from utils import split_train_test, read_dataset_discretized
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


# Combine Multiple behaviors, use only effors features and SVM as classifier 

path = r"Datasets"

CLF = SVC(gamma='auto', kernel='rbf', probability=True)
CLF_NAME = "SVM-RBF"

robot_list = ["baxter", "fetch", "sawyer"]
# robot_datatype = ["discretizedmean-10", "discretizedmean-10", "discretizedmean-10"]
robot_datatype = ["discretizedrange-15", "discretizedrange-15", "discretizedrange-15"]

#behavior_list = ["pick", "place"]
behavior_list = ["grasp", "pick", "place", "shake"]

#no_of_interactions = [1, 40, 80]
no_of_interactions = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80]
#no_of_interactions = range(1, len(train_test_splits["fold_0"]["train"]))

train_test_splits = split_train_test(FOLDS, TRIALS_PER_OBJECT)
i = 18 #for effort
new_lables = np.arange(1, NUM_OF_OBJECTS+1) # all 25 lables
NUM_OF_OBJECTS = len(new_lables)

results_path = 'Results'+os.sep+"Baseline"


for r_i, a_robot in enumerate(robot_list):
    # Read and save the examples for each behavior
    robot = {}
    for a_behavior in behavior_list:
        db_file_name = a_robot+"_"+a_behavior+"_"+robot_datatype[r_i]+".bin"
        
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
                
        robot.setdefault(a_behavior, {})
        robot[a_behavior]["examples"] = np.array(examples)
        robot[a_behavior]["labels"] = np.array(labels)
        
    # For each fold
    all_fold_scores = {}
    for a_fold in sorted(train_test_splits):
        print(a_fold)
        
        # For each no. of example
        all_behavior_scores = {}
        for example_per_objects in no_of_interactions:
            # For each behavior, combine weighted probability based on its accuracy score
            behavior_proba = {}
            for a_behavior in behavior_list:
                # Get test data
                X_test = []
                y_test = []
                for a_object in range(NUM_OF_OBJECTS):
                    X_test.extend(robot[a_behavior]["examples"][a_object][train_test_splits[a_fold]["test"]])
                    y_test.extend(robot[a_behavior]["labels"][a_object][train_test_splits[a_fold]["test"]])
                X_test_haptic = np.array(X_test)
                X_test_haptic = X_test_haptic[:, :, :, i:i+FEATURES_PER_MODALITY] # Examples, Channels, Temporal Bins, Features
                X_test_haptic = X_test_haptic.reshape((-1, X_test_haptic.shape[-3]*X_test_haptic.shape[-2]*X_test_haptic.shape[-1]))
                
                # Get train data
                X_train = []
                y_train = []
                for a_object in range(NUM_OF_OBJECTS):
                    X_train.extend(robot[a_behavior]["examples"][a_object][train_test_splits[a_fold]["train"][0:example_per_objects]])
                    y_train.extend(robot[a_behavior]["labels"][a_object][train_test_splits[a_fold]["train"][0:example_per_objects]])
                X_train_haptic = np.array(X_train)
                X_train_haptic = X_train_haptic[:, :, :, i:i+FEATURES_PER_MODALITY] # Examples, Channels, Temporal Bins, Features
                X_train_haptic = X_train_haptic.reshape((-1, X_train_haptic.shape[-3]*X_train_haptic.shape[-2]*X_train_haptic.shape[-1]))
                
                # Train and Test
                y_acc, y_pred, y_proba = classifier(CLF, X_train_haptic, X_test_haptic, y_train, y_test)
                
                y_proba_pred = np.argmax(y_proba, axis=1) + 1 # In dataset, labels starts from 1
                y_prob_acc = np.mean(y_test == y_proba_pred)
                
                all_behavior_scores.setdefault(a_behavior, [])
                all_behavior_scores[a_behavior].append(y_prob_acc)
                
                # For each behavior, get an accuracy score to combine weighted probability based on its accuracy score
                # Use only training data to get a score
                y_acc_train, y_pred_train, y_proba_train = classifier(CLF, X_train_haptic, X_train_haptic, y_train, y_train)
                y_proba_pred_train = np.argmax(y_proba_train, axis=1) + 1 # In dataset v3, labels starts from 1
                y_prob_acc_train = np.mean(y_train == y_proba_pred_train)
                                
                if y_prob_acc_train > 0:
                    # Multiple the score on training data by probability of test data to combine each
                    # behavior's performance accordingly
                    y_proba = y_proba * y_prob_acc_train # weighted probability
                behavior_proba[a_behavior] = y_proba

            # Combine weighted probability of all behaviors
            y_proba_norm = np.zeros(behavior_proba[a_behavior].shape)
            for a_behavior in behavior_list:
                y_proba_norm = y_proba_norm + behavior_proba[a_behavior]
            
            # Normalizing probability
            y_proba_norm_sum = np.sum(y_proba_norm, axis=1) # sum of weighted probability
            y_proba_norm_sum = np.repeat(y_proba_norm_sum, NUM_OF_OBJECTS, axis=0).reshape(y_proba_norm.shape)
            y_proba_norm = y_proba_norm/y_proba_norm_sum
            
            y_proba_pred = np.argmax(y_proba_norm, axis=1) + 1 # In dataset v3, labels starts from 1
            y_prob_acc = np.mean(y_test == y_proba_pred)
            all_behavior_scores.setdefault("all_behavior", [])
            all_behavior_scores["all_behavior"].append(y_prob_acc)
        
        all_fold_scores[a_fold] = all_behavior_scores # Saving scores of each fold
        
    # Taking average of all folds
    # Saving accuracy of each folds
    all_scores = {}
    for a_fold in sorted(train_test_splits):
        for a_behavior in behavior_list:
            all_scores.setdefault(a_behavior, [])
            all_scores[a_behavior].append(all_fold_scores[a_fold][a_behavior])
        all_scores.setdefault("All behaviors", [])
        all_scores["All behaviors"].append(all_fold_scores[a_fold]["all_behavior"])

    # Computing mean of each folds and plotting
    for a_behavior in sorted(all_scores):
        all_scores[a_behavior] = np.array(all_scores[a_behavior])
        all_scores[a_behavior] = np.mean(all_scores[a_behavior], axis=0)

        plt.plot(no_of_interactions, all_scores[a_behavior], label=a_behavior.capitalize())

    plt.xlabel('No. of training interaction per object')
    plt.ylabel('Accuracy ('+str(NUM_OF_OBJECTS)+' Objects)')
    plt.title(a_robot.capitalize()+': Accuracy Curve using '+robot_datatype[r_i]+' features')
    plt.legend()
    plt.savefig(results_path+os.sep+a_robot+"_"+CLF_NAME+"_"+robot_datatype[r_i]+".png", bbox_inches='tight', dpi=100)
    #plt.show()
    plt.close()
    
    # Save results
    db_file_name = results_path+os.sep+a_robot+"_"+CLF_NAME+"_"+robot_datatype[r_i]+".bin"
    output_file = open(db_file_name, "wb")
    pickle.dump(a_robot, output_file)
    pickle.dump(robot_datatype[r_i], output_file)
    pickle.dump(all_scores, output_file)
    pickle.dump(no_of_interactions, output_file)
    pickle.dump(NUM_OF_OBJECTS, output_file)
    pickle.dump(CLF_NAME, output_file)
    output_file.close()
