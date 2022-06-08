#import libraries
import numpy as np
from decisionTree import Decision_tree

# read in data.
# training data
train_data = np.genfromtxt("train_data.csv",delimiter=",")
train_x = train_data[:,:-1]
train_y = train_data[:,-1]

# validation data
valid_data = np.genfromtxt("validate_data.csv",delimiter=",")
valid_x = valid_data[:,:-1]
valid_y = valid_data[:,-1]

# test data
test_data = np.genfromtxt("test_data.csv",delimiter=",")
test_x = test_data[:,:-1]
test_y = test_data[:,-1]

# experiment with different settings of minimum node entropy
candidate_min_entropy = [0.01,0.05,0.1,0.2,0.5,0.8,1,2.0]
valid_accuracy = []
for i, min_entropy in enumerate(candidate_min_entropy):
    # initialize the model
    clf = Decision_tree(min_entropy=min_entropy)
    # update the model based on training data, and record the best validation accuracy
    clf.fit(train_x,train_y)
    predictions_train = clf.predict(train_x)
    predictions_val = clf.predict(valid_x)
    cur_train_accuracy = np.count_nonzero(predictions_train.reshape(-1)==train_y.reshape(-1))/len(train_x)
    cur_valid_accuracy = np.count_nonzero(predictions_val.reshape(-1)==valid_y.reshape(-1))/len(valid_x)
    valid_accuracy.append(cur_valid_accuracy)
    # print('Training/validation accuracy for minimum node entropy %f is %.3f / %.3f' %(candidate_min_entropy[i],cur_train_accuracy,cur_valid_accuracy))

# select the best minimum node entropy and use it to train the model
best_entropy = candidate_min_entropy[np.argmax(valid_accuracy)]
clf = Decision_tree(min_entropy=best_entropy)
clf.fit(train_x,train_y)

# evaluate on test data
predictions = clf.predict(test_x)
accuracy = np.count_nonzero(predictions.reshape(-1)==test_y.reshape(-1))/len(test_x)

print('Test accuracy with minimum node entropy %f is %.3f' %(best_entropy,accuracy))


### test passwords

def generate_features(password):
    '''
    1. 8+ characters
    2. has capital-case letter
    3. has lower-case letter
    4. has number
    5. has symbol
    '''
    features = [0, 0, 0, 0, 0]
    for char in password:
        if char.isupper():
            features[1] = 1
        if char.islower():
            features[2] = 1
        if type(char) == int or type(char) == float:
            features[3] = 1
        if (ord(char) >= 33 and ord(char) <= 47) or (ord(char) >= 58 and ord(char) <= 64) or (ord(char) >= 91 and ord(char) <= 96) or (ord(char) >= 123 and ord(char) <= 126):
            features[4] = 1
    if len(password) >= 8:
        features[0] = 1
    return features

inp = ''

while inp != '.':
    inp = input('input = ')
    features = generate_features(inp)
    print("features are ", features)

    # pre-process and filter
    if len(inp) < 8:
        prediction = [0]
    else:
        prediction = clf.predict(np.array([features]))
    
    if prediction[0] == 0:
        print("This is most likely not a password")
    else:
        print("This is most likely a password")
    print('\n')