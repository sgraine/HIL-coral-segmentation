# imports
import numpy as np

# *************************************************************************************
#                                   METRICS
# *************************************************************************************

# This script provides helper functions to manually calculate the global accuracy, 
# per class mean accuracy and per class mean iou from a confusion matrix.

# Expects the confusion matrix to be numpy array on the cpu
# The ignore class should be None if not used, or an integer otherwise
# These functions will not include classes that have not been included in the test set

# *************************************************************************************

def calculate_iou(confusion_matrix, ignore_class=None):
    num_classes = confusion_matrix.shape[0]
    iou_scores = []

    for i in range(num_classes):
        if np.sum(confusion_matrix[i, :]) == 0:
            continue  # Skip if there are no examples from this class

        if i == ignore_class:
            continue
        
        intersection = confusion_matrix[i, i]
        
        if ignore_class is not None:
            # Create a mask to exclude the row
            mask = np.ones(confusion_matrix.shape[0], dtype=bool)
            mask[ignore_class] = False

            # Sum along the specified axis while ignoring the specified row
            union = intersection + (np.sum(confusion_matrix[i, :])- intersection) + (np.sum(confusion_matrix[:, i][mask]) - intersection)
        else:
            union = intersection + (np.sum(confusion_matrix[i, :])- intersection) + (np.sum(confusion_matrix[:, i]) - intersection)
        
        if union == 0:
            iou = 0  # If Union is 0, IoU is undefined, set to 0
        else:
            iou = intersection / union
        
        iou_scores.append(iou)

    mean_iou = np.mean(iou_scores)
    
    return iou_scores, mean_iou

def calculate_pixel_accuracy(confusion_matrix, ignore_class=None):
    num_classes = confusion_matrix.shape[0]
    class_accuracies = []
    
    for i in range(num_classes):
        if np.sum(confusion_matrix[i, :]) == 0:
            continue  # Skip if there are no examples from this class
        
        if i == ignore_class:
            continue
        
        true_positives = confusion_matrix[i, i]
        total_samples = np.sum(confusion_matrix[i, :])
        
        class_accuracy = true_positives / total_samples
        class_accuracies.append(class_accuracy)
    
    mean_pixel_accuracy = np.mean(class_accuracies)
    
    return class_accuracies, mean_pixel_accuracy

def calculate_global_accuracy(confusion_matrix, ignore_class=None):
    num_classes = confusion_matrix.shape[0]
    correct_predictions = 0
    total_samples = 0
    
    for i in range(num_classes):
        if np.sum(confusion_matrix[i, :]) == 0:
            continue  # Skip if there are no examples from this class
        
        if i == ignore_class:
            continue
        
        correct_predictions += confusion_matrix[i, i]
        total_samples += np.sum(confusion_matrix[i, :])
    
    global_accuracy = correct_predictions / total_samples if total_samples != 0 else 0
    
    return global_accuracy
