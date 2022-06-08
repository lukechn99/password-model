import csv

files = ['test', 'train', 'validate']

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

for f in files:
    r = csv.reader(open(f + '_words.csv'))
    lines = list(r)
    features = []
    for l in lines:
        features.append(generate_features(l[1]) + [l[0]])
    writer = csv.writer(open(f + '_data.csv', 'w'))
    writer.writerows(features)
