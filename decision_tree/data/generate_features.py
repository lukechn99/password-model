import csv

def generate_features(password):
    '''
    1. 8+ characters
    2. has capital-case letter
    3. has lower-case letter
    4. has number
    5. has symbol
    6. only the first letter is capitalized
    '''
    features = [0, 0, 0, 0, 0, 0, 0]
    for i, char in enumerate(password):
        if char.isupper():
            features[1] = 1
        if char.islower():
            features[2] = 1
        if type(char) == int or type(char) == float:
            features[3] = 1
        if (ord(char) >= 33 and ord(char) <= 47) or (ord(char) >= 58 and ord(char) <= 64) or (ord(char) >= 91 and ord(char) <= 96) or (ord(char) >= 123 and ord(char) <= 126):
            features[4] = 1
        if i > 0 and char.isupper():
            features[6] = 1

    if password[-1] == '!':
        features[5] = 1
    if len(password) >= 8:
        features[0] = 1
    return features

def generate_data():
    files = ['test', 'train', 'validate', 'all']

    for f in files:
        r = csv.reader(open(f + '_words.csv'))
        lines = list(r)
        features = []
        for l in lines:
            features.append(generate_features(l[1]) + [l[0]])
        writer = csv.writer(open(f + '_data.csv', 'w'))
        writer.writerows(features)

def generate_all_words():
    r = csv.reader(open('all_words.csv'))
    lines = list(r)
    nlines = []
    for l in lines:
        if l[0] == '0':
            nlines.append(['0', l[1][1].upper() + l[1][2:]])
        nlines.append([l[0], l[1][1:]])
    writer = csv.writer(open('all_data.csv', 'w'))
    writer.writerows(nlines)

generate_data()