import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

#################
    # ID3 Algorithm
#################

def entropy(target_col):
    total_samples = len(target_col);
    entropy = 0
    
    # Entropy of target column is calculated using formula => E = sum(probability of the class * log(probability of the same class))
    for class_ in np.unique(target_col) :
        p = len(target_col[target_col == class_])/total_samples
        entropy = entropy - (p*np.log2(p))
    return entropy 

def InfoGain(data, split_attribute_name, target_name="class"):
    E = entropy(data[target_name])

    split_col = data[split_attribute_name]
    len_split_col = len(split_col)
    
    # Maximum info gain is iniated with negative infinite to find the best maximum after iterations
    max_info_gain = -np.inf
    best_split = None
    
    # As the data is continuos, finding the splitting points to find the best possible split at the chosen split_attribute
    split_points = get_split_points(split_col)
    for split_val in split_points :
        p = len(split_col[split_col <= split_val])/len_split_col
        
        data_less_than_split_point = data[data[split_attribute_name] <= split_val]
        data_more_than_split_point = data[data[split_attribute_name] > split_val]
        
        entropy_split = (p*entropy(data_less_than_split_point[target_name]) + (1-p)*entropy(data_more_than_split_point[target_name]))

        # Total Information Gain = Total entropy of data - Entropy of the split feature
        info_gain_split = E - entropy_split

        # Best possible split is obtained based on maximum information gain
        if(max_info_gain < info_gain_split) :
            max_info_gain = info_gain_split
            best_split = split_val
    
    return (max_info_gain, best_split)

def ID3(data, originaldata, features, target_attribute_name="class", parent_node_class = None):
    
    empty_data = len(data) == 0
    num_features = len(features)
    unique_classes = np.unique(data[target_attribute_name])
    
    # Stopping conditions
    if empty_data :
        return get_mode_target(originaldata, target_attribute_name)
      
    elif num_features == 0 :
        return parent_node_class
                  
    else :
        # Returning the class when there the all the samples in the set belong to same class
        if len(unique_classes) <= 1 :
            return unique_classes[0]
        else :
            # Finding the best split feature based on maximum information gain
            IG_info = [InfoGain(data, f, target_attribute_name) for f in features]
            IG_list = [info[0] for info in IG_info]
            
            IG_max_feature_index = np.argmax(IG_list)
            
            split_feature_name = features[IG_max_feature_index]

            split_val = IG_info[IG_max_feature_index][1]

            # Removing the feature on which the tree got split already
            new_feature_set = [f for f in features if f != split_feature_name]
            
            # Structure of the decision tree where val -> value on which it is split, left -> dataset which is less the split value for that feature,
            # right -> dataset which is greater than the split val for the feature
            tree = {split_feature_name : {'val': {}, 'left' : {}, 'right' : {}}}
            tree[split_feature_name]['val'] = split_val
            
            left_sub_data = data[data[split_feature_name] <= split_val]
            left_sub_tree = ID3(left_sub_data, originaldata, new_feature_set, "class", get_mode_target(data, target_attribute_name))

            tree[split_feature_name]['left'] = left_sub_tree
            
            right_sub_data = data[data[split_feature_name] > split_val]
            right_sub_tree = ID3(right_sub_data, originaldata, new_feature_set, "class", get_mode_target(data, target_attribute_name))

            tree[split_feature_name]['right'] = right_sub_tree
                
            return tree
 
# Calculating the generic class to return based on the probability of the class distribution on the data
def get_mode_target(data, target_attribute_name) :
    tot_samples_per_class = np.unique(data[target_attribute_name], return_counts=True)[1]
    return np.unique(data[target_attribute_name])[np.argmax(tot_samples_per_class[1])]

# Find all the midpoints for every adjacent values in the unique list of a particular column to increasing the correct prediction for continious data
def get_split_points(split_feature_col) :
    split_points = []
    unique_split_vals = np.sort(np.unique(split_feature_col))
    for i in range(len(unique_split_vals)-1) :
        split_points.append((unique_split_vals[i] + unique_split_vals[i+1])/2)
    return split_points
   
# Predict the class of samples using the trained ID3 model
def predict_con(x_sample, decision_tree) :
    for key in list(x_sample.keys()):
        if key in list(decision_tree.keys()):
            if x_sample[key] <= decision_tree[key]['val'] :
                result = decision_tree[key]['left']
            else :
                result = decision_tree[key]['right']
            
            if isinstance(result, dict):
                return predict_con(x_sample, result)
            else:
                return result 




#################
    # C 4.5 Algorithm
#################

def entropy(target_col):
    total_samples = len(target_col);
    entropy = 0
    
    # Entropy of target column is calculated using formula => E = sum(probability of the class * log(probability of the same class))
    for class_ in np.unique(target_col) :
        p = len(target_col[target_col == class_])/total_samples
        entropy = entropy - (p*np.log2(p))
    return entropy

def info_gain(data, split_attribute_name, target_name="class"):
    E = entropy(data[target_name])

    split_col = data[split_attribute_name]
    len_split_col = len(split_col)

    # Maximum info gain is iniated with negative infinite to find the best maximum after iterations
    max_info_gain = -np.inf
    best_split = None

    # As the data is continuos, finding the splitting points to find the best possible split at the chosen split_attribute
    split_points = get_split_points(split_col)
    for split_val in split_points :
        p = len(split_col[split_col <= split_val])/len_split_col
        
        data_less_than_split_point = data[data[split_attribute_name] <= split_val]
        data_more_than_split_point = data[data[split_attribute_name] > split_val]
        
        entropy_split = (p*entropy(data_less_than_split_point[target_name]) + (1-p)*entropy(data_more_than_split_point[target_name]))

        # Total Information Gain = Total entropy of data - Entropy of the split feature
        info_gain_split = E - entropy_split
        
        # Best possible split is obtained based on maximum information gain
        if(max_info_gain < info_gain_split) :
            max_info_gain = info_gain_split
            best_split = split_val
    
    return (max_info_gain, best_split) 

# Gain ratio is the key feature to split the data in C4.5
def gain_ratio(data, split_attribute_name, target_name="class"):
    split_col = data[split_attribute_name]
    len_split_col = len(split_col)
    max_gain_ratio = -np.inf

    IG, split_val = info_gain(data, split_attribute_name)
    p = len(split_col[split_col <= split_val])/len_split_col
    
    # Split Info 
    split_info = -(p) * np.log2(p) - (1-p)*np.log2(1-p)
    
    # Gain ratio = Information Gain / Split Info
    gain_ratio = IG/split_info
    
    return (gain_ratio, split_val)

def best_split(data, features, target_name="class"):
    # Best split is found based on high gain ratio after iterating through all the features
    gain_ratio_list = [gain_ratio(data, feature) for feature in features]
    best_feature_index = np.argmax([gr[0] for gr in gain_ratio_list])
    best_split_feature = features[best_feature_index]
    best_split_val = [gr[1] for gr in gain_ratio_list][best_feature_index]

    # Return the best split feature along with the split value as it is continuous data
    return (best_split_feature, best_split_val)

def C45(data, originaldata, features, target_attribute_name="class", parent_node_class=None):
    empty_data = len(data) == 0
    num_features = len(features)
    unique_classes = np.unique(data[target_attribute_name])
    
    # Stopping conditions
    if empty_data :
        return get_mode_target(originaldata, target_attribute_name)
      
    elif num_features == 0 :
        return parent_node_class
                  
    else :
        # Returning the class when there the all the samples in the set belong to same class
        if len(unique_classes) <= 1 :
            return unique_classes[0]
        else :
            split_feature_name , split_val = best_split(data, features)

            # Removing the feature on which the tree got split already
            new_feature_set = [f for f in features if f != split_feature_name]
            
            # Structure of the decision tree where val -> value on which it is split, left -> dataset which is less the split value for that feature,
            # right -> dataset which is greater than the split val for the feature
            tree = {split_feature_name : {'val': {}, 'left' : {}, 'right' : {}}}
            tree[split_feature_name]['val'] = split_val
            
            left_sub_data = data[data[split_feature_name] <= split_val]
            left_sub_tree = C45(left_sub_data, originaldata, new_feature_set, "class", get_mode_target(data, target_attribute_name))

            tree[split_feature_name]['left'] = left_sub_tree
            
            right_sub_data = data[data[split_feature_name] > split_val]
            right_sub_tree = C45(right_sub_data, originaldata, new_feature_set, "class", get_mode_target(data, target_attribute_name))

            tree[split_feature_name]['right'] = right_sub_tree 

            return tree

# Calculating the generic class to return based on the probability of the class distribution on the data
def get_mode_target(data, target_attribute_name) :
    tot_samples_per_class = np.unique(data[target_attribute_name], return_counts=True)[1]
    return np.unique(data[target_attribute_name])[np.argmax(tot_samples_per_class[1])]

# Find all the midpoints for every adjacent values in the unique list of a particular column to increasing the correct prediction for continious data
def get_split_points(split_feature_col) :
    split_points = []
    unique_split_vals = np.sort(np.unique(split_feature_col))
    for i in range(len(unique_split_vals)-1) :
        split_points.append((unique_split_vals[i] + unique_split_vals[i+1])/2)
    return split_points

# Predict the class of samples using the trained C4.5 model
def predict(tree, instance):
    for col in list(instance.keys()):
        if col in list(tree.keys()):
            if instance[col] <= tree[col]['val'] :
                child = tree[col]['left']
            else :
                child = tree[col]['right']
            if isinstance(child, dict):
                return predict(child, instance)
            else:
                return child  

def main():
    # Load and prepare the dataset
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Split the dataset for training and testing for ID3
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=42)

    # Prepare training data
    training_data = pd.DataFrame(X_train, columns=iris["feature_names"])
    training_data['class'] = y_train

    # Define features
    features = iris["feature_names"]

    # Train the ID3 model
    print("Training ID3 model")
    id3_tree = ID3(training_data, training_data, features, "class")
    print("ID3 decision tree:")
    print(id3_tree)

    # Prepare test data for ID3 prediction
    test_data = pd.DataFrame(X_test, columns=iris["feature_names"])
    test_data['class'] = y_test
    test_data_query = test_data.to_dict('records')

    # Predict and evaluate the ID3 model
    print("\nEvaluating ID3 model on test data")
    y_predicted_test_id3 = [predict_con(test_instance, id3_tree) for test_instance in test_data_query]
    evaluate_model(y_test, y_predicted_test_id3)

    # Split the dataset for training and testing for C4.5
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Prepare training data
    training_data = pd.DataFrame(X_train, columns=iris["feature_names"])
    training_data['class'] = y_train
    
    # Train the C4.5 model
    print("\nTraining C4.5 model")
    c45_tree = C45(training_data, training_data, features, "class")
    print("C4.5 decision tree:")
    print(c45_tree)
    
    # Prepare test data for C4.5 prediction
    test_data = pd.DataFrame(X_test, columns=iris["feature_names"])
    test_data['class'] = y_test
    test_data_query = test_data.to_dict('records')
                                       
    # Predict and evaluate the C4.5 model
    print("\nEvaluating C4.5 model on test data...")
    y_predicted_test_c45 = [predict(c45_tree, test_instance) for test_instance in test_data_query]
    evaluate_model(y_test, y_predicted_test_c45)

def evaluate_model(y_test, y_predicted_test):
    cm = metrics.confusion_matrix(y_test, y_predicted_test)
    precision = metrics.precision_score(y_test, y_predicted_test, average='weighted')
    accuracy = metrics.accuracy_score(y_test, y_predicted_test)
    recall = metrics.recall_score(y_test, y_predicted_test, average='weighted')
    f1_score = metrics.f1_score(y_test, y_predicted_test, average='weighted')
    print(cm)
    print("Precision:", precision)
    print("Accuracy:", accuracy)
    print("Recall:", recall)
    print("F1_score:", f1_score)



if __name__ == "__main__":
    main()
