import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk import word_tokenize
import string

# Model building

def preprocess_emails(df):
    """
    Preprocesses email data from a DataFrame.

    Parameters:
    - df (pandas.DataFrame): The input DataFrame containing email data with 'text' and 'spam' columns.

    Returns:
    - tuple: A tuple containing two elements:
        1. X (numpy.array): An array containing email content after removing the "Subject:" prefix.
        2. Y (numpy.array): An array indicating whether each email is spam (1) or ham (0).

    The function shuffles the input DataFrame to avoid biased results in train/test splits.
    It then extracts email content and spam labels, removing the "Subject:" prefix from each email.

    """
    # Shuffles the dataset
    df = df.sample(frac = 1, ignore_index = True, random_state = 420)
    # Removes the "Subject:" string, which comprises the first 9 characters of each email. Also, convert it to a numpy array.
    X = df.text.apply(lambda x: x[9:]).to_numpy()
    # Convert the labels to numpy array
    Y = df.spam.to_numpy()
    return X, Y

def preprocess_text(X):
    """
    Preprocesses a collection of text data by removing stopwords and punctuation.

    Parameters:
    - X (str or array-like): The input text data to be processed. If a single string is provided,
      it will be converted into a one-element numpy array.

    Returns:
    - numpy.array: An array of preprocessed text data, where each element represents a document
      with stopwords and punctuation removed.

    Note:
    - The function uses the Natural Language Toolkit (nltk) library for tokenization and stopword removal.
    - If the input is a single string, it is converted into a one-element numpy array.
    """
    # Make a set with the stopwords and punctuation
    stop = set(stopwords.words('english') + list(string.punctuation))

    # The next lines will handle the case where a single email is passed instead of an array of emails.
    if isinstance(X, str):
        X = np.array([X])

    # The result will be stored in a list
    X_preprocessed = []

    for email in X:
        email = np.array([i.lower() for i in word_tokenize(email) if i.lower() not in stop]).astype(X.dtype)
        X_preprocessed.append(email)
        
    if len(X) == 1:
        return X_preprocessed[0]
    return X_preprocessed

def get_word_frequency(X,Y):
    """
    Calculate the frequency of each word in a set of emails categorized as spam (1) or not spam (0).

    Parameters:
    - X (numpy.array): Array of emails, where each email is represented as a list of words.
    - Y (numpy.array): Array of labels corresponding to each email in X. 1 indicates spam, 0 indicates ham.

    Returns:
    - word_dict (dict): A dictionary where keys are unique words found in the emails, and values
      are dictionaries containing the frequency of each word for spam (1) and not spam (0) emails.
    """
    # Creates an empty dictionary
    word_dict = {}

    num_emails = len(X)

    # Iterates over every processed email and its label
    for i in range(num_emails):
        # Get the i-th email
        email = X[i] 
        # Get the i-th label. This indicates whether the email is spam or not. 1 = None
        # The variable name cls is an abbreviation for class, a reserved word in Python.
        cls = Y[i] 
        # To avoid counting the same word twice in an email, remove duplicates by casting the email as a set
        email = set(email) 
        # Iterates over every distinct word in the email
        for word in email:
            # If the word is not already in the dictionary, manually add it. Remember that you will start every word count as 1 both in spam and ham
            if word not in word_dict.keys():
                word_dict[word] = {"spam": 1, "ham": 1}
            # Add one occurrence for that specific word in the key ham if cls == 0 and spam if cls == 1. 
            if cls == 0:    
                word_dict[word]["ham"] += 1
            if cls == 1:
                word_dict[word]["spam"] += 1
    
    return word_dict

def prob_word_given_class(word, cls, word_frequency, class_frequency):
    """
    Calculate the conditional probability of a given word occurring in a specific class.

    Parameters:
    - word (str): The target word for which the probability is calculated.
    - cls (str): The class for which the probability is calculated, it may be 'spam' or 'ham'
    - word_frequency (dict): The dictionary containing the words frequency.
    - class_frequency (dict): The dictionary containing the class frequency.

    Returns:
    - float: The conditional probability of the given word occurring in the specified class.
    """
    
    # Get the amount of times the word appears with the given class (class is stored in spam variable)
    amount_word_and_class = word_frequency[word][cls]
    p_word_given_class = amount_word_and_class/class_frequency[cls]

    return p_word_given_class

def prob_email_given_class(treated_email, cls, word_frequency, class_frequency):
    """
    Calculate the probability of an email being of a certain class (e.g., spam or ham) based on treated email content.

    Parameters:
    - treated_email (list): A list of treated words in the email.
    - cls (str): The class label for the email. It can be either 'spam' or 'ham'
    - word_frequency (dict): The dictionary containing the words frequency.
    - class_frequency (dict): The dictionary containing the class frequency.

    Returns:
    - float: The probability of the given email belonging to the specified class.
    """

    # prob starts at 1 because it will be updated by multiplying it with the current P(word | class) in every iteration
    prob = 1

    for word in treated_email:
        # Only perform the computation for words that exist in the word frequency dictionary
        if word in word_frequency.keys(): 
            # Update the prob by multiplying it with P(word | class). Don't forget to add the word_frequency and class_frequency parameters!
            prob *= prob_word_given_class(word, cls, word_frequency, class_frequency)

    return prob

def naive_bayes(treated_email, word_frequency, class_frequency, return_likelihood = False):    
    """
    Naive Bayes classifier for spam detection.

    This function calculates the probability of an email being spam (1) or ham (0)
    based on the Naive Bayes algorithm. It uses the conditional probabilities of the
    treated_email given spam and ham, as well as the prior probabilities of spam and ham
    classes. The final decision is made by comparing the calculated probabilities.

    Parameters:
    - treated_email (list): A preprocessed representation of the input email.
    - word_frequency (dict): The dictionary containing the words frequency.
    - class_frequency (dict): The dictionary containing the class frequency.
        - return_likelihood (bool): If true, it returns the likelihood of both spam and ham.

    Returns:
    If return_likelihood = False:
        - int: 1 if the email is classified as spam, 0 if classified as ham.
    If return_likelihood = True:
        - tuple: A tuple with the format (spam_likelihood, ham_likelihood)
    """

    # Compute P(email | spam) with the function you defined just above. Don't forget to add the word_frequency and class_frequency parameters!
    prob_email_given_spam = prob_email_given_class(treated_email, "spam", word_frequency, class_frequency)

    # Compute P(email | ham) with the function you defined just above. Don't forget to add the word_frequency and class_frequency parameters!
    prob_email_given_ham = prob_email_given_class(treated_email, "ham", word_frequency, class_frequency)

    # Compute P(spam) using the class_frequency dictionary and using the formula #spam emails / #total emails
    p_spam = class_frequency['spam']/(class_frequency['ham'] + class_frequency['spam'])

    # Compute P(ham) using the class_frequency dictionary and using the formula #ham emails / #total emails
    p_ham = class_frequency['ham']/(class_frequency['ham'] + class_frequency['spam'])

    # Compute the quantity P(spam) * P(email | spam), let's call it spam_likelihood
    spam_likelihood = prob_email_given_spam * p_spam

    # Compute the quantity P(ham) * P(email | ham), let's call it ham_likelihood
    ham_likelihood = prob_email_given_ham * p_ham


    # In case of passing return_likelihood = True, then return the desired tuple
    if return_likelihood == True:
        return (spam_likelihood, ham_likelihood)
    
    # Compares both values and choose the class corresponding to the higher value
    elif spam_likelihood >= ham_likelihood:
        return 1
    else:
        return 0
    
# Model performance
def get_true_positives(Y_true, Y_pred):
    """
    Calculate the number of true positive instances in binary classification.

    Parameters:
    - Y_true (list): List of true labels (0 or 1) for each instance.
    - Y_pred (list): List of predicted labels (0 or 1) for each instance.

    Returns:
    - int: Number of true positives, where true label and predicted label are both 1.
    """
    # Both Y_true and Y_pred must match in length.
    if len(Y_true) != len(Y_pred):
        return "Number of true labels and predict labels must match!"
    n = len(Y_true)
    true_positives = 0
    # Iterate over the number of elements in the list
    for i in range(n):
        # Get the true label for the considered email
        true_label_i = Y_true[i]
        # Get the predicted (model output) for the considered email
        predicted_label_i = Y_pred[i]
        # Increase the counter by 1 only if true_label_i = 1 and predicted_label_i = 1 (true positives)
        if true_label_i == 1 and predicted_label_i == 1:
            true_positives += 1
    return true_positives
        
def get_true_negatives(Y_true, Y_pred):
    """
    Calculate the number of true negative instances in binary classification.

    Parameters:
    - Y_true (list): List of true labels (0 or 1) for each instance.
    - Y_pred (list): List of predicted labels (0 or 1) for each instance.

    Returns:
    - int: Number of true negatives, where true label and predicted label are both 0.
    """
    
    # Both Y_true and Y_pred must match in length.
    if len(Y_true) != len(Y_pred):
        return "Number of true labels and predict labels must match!"
    n = len(Y_true)
    true_negatives = 0
    # Iterate over the number of elements in the list
    for i in range(n):
        # Get the true label for the considered email
        true_label_i = Y_true[i]
        # Get the predicted (model output) for the considered email
        predicted_label_i = Y_pred[i]
        # Increase the counter by 1 only if true_label_i = 0 and predicted_label_i = 0 (true negatives)
        if true_label_i == 0 and predicted_label_i == 0:
            true_negatives += 1
    return true_negatives

# Making it better by using the log function to preserve precision
def log_prob_email_given_class(treated_email, cls, word_frequency, class_frequency):
    """
    Calculate the log probability of an email being of a certain class (e.g., spam or ham) based on treated email content.

    Parameters:
    - treated_email (list): A list of treated words in the email.
    - cls (str): The class label ('spam' or 'ham')
    

    Returns:
    - float: The log probability of the given email belonging to the specified class.
    """

    # prob starts at 0 because it will be updated by summing it with the current log(P(word | class)) in every iteration
    prob = 0

    for word in treated_email: 
        # Only perform the computation for words that exist in the word frequency dictionary
        if word in word_frequency.keys(): 
            # Update the prob by summing it with log(P(word | class))
            prob += np.log(prob_word_given_class(word, cls,word_frequency, class_frequency))

    return prob

def log_naive_bayes(treated_email, word_frequency, class_frequency, return_likelihood = False):    
    """
    Naive Bayes classifier for spam detection, comparing the log probabilities instead of the actual probabilities.

    This function calculates the log probability of an email being spam (1) or ham (0)
    based on the Naive Bayes algorithm. It uses the conditional probabilities of the
    treated_email given spam and ham, as well as the prior probabilities of spam and ham
    classes. The final decision is made by comparing the calculated probabilities.

    Parameters:
    - treated_email (list): A preprocessed representation of the input email.
    - return_likelihood (bool): If true, it returns the log_likelihood of both spam and ham.

    Returns:
    - int: 1 if the email is classified as spam, 0 if classified as ham.
    """
    
    # Compute P(email | spam) with the new log function
    log_prob_email_given_spam = log_prob_email_given_class(treated_email, cls = 'spam',word_frequency = word_frequency, class_frequency = class_frequency) 

    # Compute P(email | ham) with the function you defined just above
    log_prob_email_given_ham = log_prob_email_given_class(treated_email, cls = 'ham',word_frequency = word_frequency, class_frequency = class_frequency) 

    # Compute P(spam) using the class_frequency dictionary and using the formula #spam emails / #total emails
    p_spam = class_frequency['spam']/(class_frequency['ham'] + class_frequency['spam']) 

    # Compute P(ham) using the class_frequency dictionary and using the formula #ham emails / #total emails
    p_ham = class_frequency['ham']/(class_frequency['ham'] + class_frequency['spam']) 

    # Compute the quantity log(P(spam)) + log(P(email | spam)), let's call it log_spam_likelihood
    log_spam_likelihood = np.log(p_spam) + log_prob_email_given_spam 

    # Compute the quantity P(ham) * P(email | ham), let's call it ham_likelihood
    log_ham_likelihood = np.log(p_ham) + log_prob_email_given_ham 

    # In case of passing return_likelihood = True, then return the desired tuple
    if return_likelihood == True:
        return (log_spam_likelihood, log_ham_likelihood)
    
    # Compares both values and choose the class corresponding to the higher value. 
    # As the logarithm is an increasing function, the class with the higher value retains this property.
    if log_spam_likelihood >= log_ham_likelihood:
        return 1
    else:
        return 0
    
# Enhancing model performance: Practical implementation with Naive Bayes
def get_recall(Y_true, Y_pred):
    """
    Calculate the recall for a binary classification task.

    Parameters:
    - Y_true (array-like): Ground truth labels.
    - Y_pred (array-like): Predicted labels.

    Returns:
    - recall (float): The recall score, which is the ratio of true positives to the total number of actual positives.
    """
    # Get the total number of spam emails. Since they are 1 in the data, it suffices summing all the values in the array Y.
    total_number_spams = Y_test.sum()
    # Get the true positives
    true_positives = get_true_positives(Y_true, Y_pred)
    
    # Compute the recall
    recall = true_positives/total_number_spams
    return recall

def get_false_positives(Y_true, Y_pred):
    """
    Calculate the number of false positives instances in binary classification.

    Parameters:
    - Y_true (list): List of true labels (0 or 1) for each instance.
    - Y_pred (list): List of predicted labels (0 or 1) for each instance.

    Returns:
    - int: Number of false positives, where true label is 0 and predicted label is 1.
    """
    
    # Both Y_true and Y_pred must match in length.
    if len(Y_true) != len(Y_pred):
        return "Number of true labels and predict labels must match!"
    n = len(Y_true)

    false_positives = 0
    # Iterate over the number of elements in the list
    for i in range(n):
        # Get the true label for the considered email
        true_label_i = Y_true[i]
        # Get the predicted (model output) for the considered email
        predicted_label_i = Y_pred[i]
        # Increase the counter by 1 only if true_label_i = 0 and predicted_label_i = 0 (false positive)
        if true_label_i == 0 and predicted_label_i == 1:
            false_positives += 1
    return false_positives

def get_precision(Y_true, Y_pred):
    """
    Calculate precision, a metric for the performance of a classification model,
    by computing the ratio of true positives to the sum of true positives and false positives.

    Parameters:
    - Y_true (list): True labels.
    - Y_pred (list): Predicted labels.

    Returns:
    - precision (float): Precision score.
    """
    # Get the true positives
    true_positives = get_true_positives(Y_true, Y_pred)
    false_positives = get_false_positives(Y_true, Y_pred)
    precision = true_positives/(true_positives + false_positives)
    return precision

dataframe_emails = pd.read_csv('/Users/truthixify/dev/ai/python/math/prob/labs/bayes/emails.csv')
X, Y = preprocess_emails(dataframe_emails)
X_treated = preprocess_text(X)

TRAIN_SIZE = int(0.80*len(X_treated)) # 80% of the samples will be used to train.

X_train = X_treated[:TRAIN_SIZE]
Y_train = Y[:TRAIN_SIZE]
X_test = X_treated[TRAIN_SIZE:]
Y_test = Y[TRAIN_SIZE:]

# This will build the word_frequency dictionary using the training set. 
word_frequency = get_word_frequency(X_train,Y_train)
# To count the spam and ham emails, you may just sum the respective 1 and 0 values in the training dataset, since the convention is spam = 1 and ham = 0.
class_frequency = {'ham': sum(Y_train == 0), 'spam': sum(Y_train == 1)}

# Use the Naive Bayes model (standard and log versions) to classify every email in the test dataset
Y_pred_naive_bayes = []
Y_pred_log_naive_bayes = []

for email in X_test:
  prediction = naive_bayes(email,word_frequency = word_frequency, class_frequency = class_frequency)
  log_prediction = log_naive_bayes(email,word_frequency = word_frequency, class_frequency = class_frequency)
  Y_pred_naive_bayes.append(prediction)
  Y_pred_log_naive_bayes.append(log_prediction)

print(f"Precision of the standard Naive Bayes model: {get_precision(Y_test, Y_pred_naive_bayes):.4f}")
print(f"Precision of the log Naive Bayes model: {get_precision(Y_test, Y_pred_log_naive_bayes):.4f}")