import pandas as pd
import re
sms_spam = pd.read_csv('SMSSpamCollection', sep='\t',
                       header=None, names=['Label', 'SMS'])
# print("data set = ", sms_spam.shape)
# print(sms_spam.head())
sms_spam['Label'].value_counts(normalize=True)

# Randomize the dataset
data_randomized = sms_spam.sample(frac=1, random_state=1)

# Calculate index for split
training_test_index = int(round(len(data_randomized) * 0.8))

# Split into training and test sets
training_set = data_randomized[:training_test_index].reset_index(drop=True)
test_set = data_randomized[training_test_index:].reset_index(drop=True)

# print("training set = ", training_set.shape)
# print("test set = ", test_set.shape)

# print(training_set['Label'].value_counts(normalize=True))
# print(test_set['Label'].value_counts(normalize=True))

# ----- data cleaning -----
# before
# print("before", training_set.head(3))

# After
training_set['SMS'] = training_set['SMS'].str.replace('\W', ' ')  # Removes punctuation
training_set['SMS'] = training_set['SMS'].str.lower()  # lower upper case characters
# print("after", training_set.head(3))

# -----vocabulary creation-----
training_set['SMS'] = training_set['SMS'].str.split()

vocabulary = []
for sms in training_set['SMS']:
    for word in sms:
        vocabulary.append(word)

vocabulary = list(set(vocabulary))  # remove duplicates
# print(len(vocabulary))

#  -----final training set-----
# build dictionary
word_counts_per_sms = {unique_word: [0] * len(training_set['SMS']) for unique_word in vocabulary}

for index, sms in enumerate(training_set['SMS']):
    for word in sms:
        word_counts_per_sms[word][index] += 1

word_counts = pd.DataFrame(word_counts_per_sms)
word_counts.head()
training_set_clean = pd.concat([training_set, word_counts], axis=1)
training_set_clean.head()
# print (word_counts.head())

# Isolating spam and ham messages first
spam_messages = training_set_clean[training_set_clean['Label'] == 'spam']
ham_messages = training_set_clean[training_set_clean['Label'] == 'ham']

# P(Spam) and P(Ham)
p_spam = float(len(spam_messages)) / float(len(training_set_clean))
p_ham = float(len(ham_messages)) / float(len(training_set_clean))
# N_Spam
n_words_per_spam_message = spam_messages['SMS'].apply(len)
n_spam = n_words_per_spam_message.sum()

# N_Ham
n_words_per_ham_message = ham_messages['SMS'].apply(len)
n_ham = n_words_per_ham_message.sum()

# N_Vocabulary
n_vocabulary = len(vocabulary)

# Laplace smoothing
alpha = 1

# Initiate parameters
parameters_spam = {unique_word: 0 for unique_word in vocabulary}
parameters_ham = {unique_word: 0 for unique_word in vocabulary}

# Calculate parameters
for word in vocabulary:
    n_word_given_spam = spam_messages[word].sum()  # spam_messages already defined
    p_word_given_spam = float((n_word_given_spam + alpha)) / float((n_spam + alpha*n_vocabulary))
    parameters_spam[word] = p_word_given_spam

    n_word_given_ham = ham_messages[word].sum()  # ham_messages already defined
    p_word_given_ham = float((n_word_given_ham + alpha)) / float((n_ham + alpha*n_vocabulary))
    parameters_ham[word] = p_word_given_ham


def classify(message):    # message is a string

    message = re.sub('\W', ' ', message)
    message = message.lower().split()

    p_spam_given_message = p_spam
    p_ham_given_message = p_ham

    for words in message:
        if words in parameters_spam:
            p_spam_given_message *= parameters_spam[words]

        if words in parameters_ham:
            p_ham_given_message *= parameters_ham[words]

    print('P(Spam|message):', p_spam_given_message)
    print('P(Ham|message):', p_ham_given_message)

    if p_ham_given_message > p_spam_given_message:
        print('Label: Ham')
    elif p_ham_given_message < p_spam_given_message:
        print('Label: Spam')
    else:
        print('Equal probabilities, have a human classify this!')


# classify('WINNER!! This is the secret code to unlock the money: C3421.')
# classify("Sounds good, Tom, then see u there")

def classify_test_set(message):   # message is a string

    message = re.sub('\W', ' ', message)
    message = message.lower().split()

    p_spam_given_message = p_spam
    p_ham_given_message = p_ham

    for words in message:
        if words in parameters_spam:
            p_spam_given_message *= parameters_spam[words]

        if words in parameters_ham:
            p_ham_given_message *= parameters_ham[words]

    if p_ham_given_message > p_spam_given_message:
        return 'ham'
    elif p_spam_given_message > p_ham_given_message:
        return 'spam'
    else:
        return 'needs human classification'


test_set['predicted'] = test_set['SMS'].apply(classify_test_set)
# print(test_set.head())

correct = 0
total = test_set.shape[0]

for row in test_set.iterrows():
    row = row[1]
    if row['Label'] == row['predicted']:
        correct += 1

print('Correct:', correct)
print('Incorrect:', total - correct)
print('Accuracy:', float(correct)/total)
