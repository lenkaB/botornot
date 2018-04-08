import csv, re
from langdetect import detect
import io
import spacy
import matplotlib.pyplot as plt
import preprocessor as p
import nltk
import random
from nltk import word_tokenize
import collections
import enchant
import math

dictionary = enchant.Dict('en')
nlp = spacy.load('en')
relationships_file_path = '/Users/lenka/Downloads/human_relationships_words.txt'
with open(relationships_file_path, 'r') as relationships_file:
    relationships_list = relationships_file.readlines()


def human_relationships_identifier(tweet):
    '''Function to check for relationship words in tweet'''
    found_list = []
    # create set of relationship words because is faster than list
    relationships_set = set()
    for word in relationships_list:
        # get rid of the \n
        clean_word = word.replace('\n', "")
        relationships_set.add(clean_word)

    # check for relationship word in tweet
    tokens_tweet = word_tokenize(tweet)
    for token in tokens_tweet:
        for rel in relationships_set:
            if rel == (token.lower().strip()):
                found_list.append(token)
                # print(f'tweet has {token}')
    #if len(found_list):
    #    print(str(found_list))
    return found_list


def extract_stats(clean_tweet, tweet_id, human_or_bot):
    found = human_relationships_identifier(clean_tweet)
    # print(clean_tweet,found)

    pos_list = []
    pos_counter = collections.Counter()
    spacy_stats = nlp(clean_tweet)
    for token in spacy_stats:
        pos_counter[token.pos_] += 1
        pos_list.append(token.pos_)



    entity_list = []
    entity_label_list = []
    for ent in spacy_stats.ents:
        entity_label_list.append(ent.label_)
        entity_list.append(ent.text)
    char_count = len(spacy_stats.text)

    unique_words = []
    processed_tweet = p.tokenize(clean_tweet)

    curve = ttr_curve(pos_list)
    processed_word_count = 0
    spell_error_count = 0
    for word in processed_tweet.split():
        # clean_word = re.sub(r'[^A-Za-z]', "", word)
        # if dictionary.check(clean_word) is False and not word.startswith('$'):
        #    print(word)
        #    spell_error_count += 1
        processed_word_count += 1
        if word not in unique_words:
            unique_words.append(word)

    if processed_word_count > 0:
        ttr = float(len(unique_words)) / float(processed_word_count)
    else:
        ttr = 0

    tweet_dict = {'index': tweet_id,
                  'raw': clean_tweet,
                  'preprocessed tweet': processed_tweet,
                  'char count': char_count,
                  'pos': pos_list,
                  'pronouns': pos_counter['PRON'],
                  'nouns': pos_counter['NOUN'],
                  'verbs': pos_counter['VERB'],
                  'adverbs': pos_counter['ADV'],
                  'adjectives': pos_counter['ADJ'],
                  'symbols': pos_counter['SYM'],
                  'punctuation': pos_counter['PUNCT'],
                  'proper nouns': pos_counter['PROPN'],
                  'entity label': entity_label_list,
                  'word count': processed_word_count,
                  'unique word count': len(unique_words),
                  'TTR': ttr,
                  'entity raw text': entity_list,
                  'mentions': clean_tweet.count('@'),
                  'hashtags': clean_tweet.count('#'),
                  'urls': clean_tweet.count('$URL$'),
                  'class': human_or_bot, 'relationship words': found,
                  'count(rel words)': len(found),
                  'ttr_curve': curve}

    return tweet_dict


def calculate_precision_and_recall(classifier, test_set):
    #TODO: store truth and predictions in output file
    cl = {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0}
    for test_tweet in test_set:
        truth = test_tweet[1]
        prediction = classifier.classify(test_tweet[0])
        if truth == 'bot':
            if prediction == 'bot':
                cl['tp'] += 1
            else:
                cl['fn'] += 1
        else:
            if prediction == 'bot':
                cl['fp'] += 1
            else:
                cl['tn'] += 1
    precision = cl['tp'] / (cl['tp'] + cl['fp'])
    recall = cl['tp'] / (cl['tp'] + cl['fn'])
    f1 = (precision + recall) / 2
    dic = {'precision':precision, 'recall':recall, 'f1':f1}
    return dic


def english_tweet_cleaner(raw_tweet):
    '''Function to clean tweet and check if it is identified as English.
    Returns a clean_tweet string'''
    # identify a clean_tweet by removing unwanted characters from tweet
    # remove anything that is not A-Za-z0-9 :;()@#*,.?!\\\/\-\_
    clean_tweet = re.sub(r'[^A-Za-z0-9 :;()@#*,.?!\\\/\-\_]', "", raw_tweet)
    # if the clean tweet is not just white space and not just 'text'(an outlier problem)
    if str.isspace(clean_tweet) or clean_tweet == 'text':
        return ''
    try:
        if detect(clean_tweet) != 'en':
            return ''
        else:
            return clean_tweet
    except:
        pass


def load_dataset_and_print_stats(human_or_bot, max):
    tweet_id = 0
    # based on the argument of the function human_or_bot, we set the parameters for loading and storing
    if human_or_bot is 'human':
        filename = '/Users/lenka/Desktop/humans2017.csv'
        output = '/Users/lenka/Desktop/output_human2017' + str(max) + '.csv'
        tweet_column = 1
    elif human_or_bot is 'bot':
        #filename = '/Users/lenka/Desktop/bot/datasets/cresci-2015.csv/INT.csv/tweets.csv'
        filename = '/Users/lenka/Desktop/bot/datasets/bot.csv'
        output = '/Users/lenka/Desktop/output_bot_fakefol15' + str(max) + '.csv'
        tweet_column = 2

    with io.open(filename, "r", encoding="ISO-8859-1") as my_file:
        reader = csv.reader(my_file)
        stored_info = list()
        with open(output, 'w', encoding="ISO-8859-1") as out:
            # header for our output = all the features we extract
            fieldnames = ['index', 'raw', 'preprocessed tweet', 'char count', 'pos',
                          'pronouns', 'nouns', 'verbs', 'adverbs', 'adjectives',
                          'symbols', 'punctuation', 'proper nouns', 'entity label',
                          'word count', 'unique word count', 'TTR', 'entity raw text', 'mentions', 'hashtags','urls',
                          'count(rel words)',
                          'relationship words', 'class', 'ttr_curve'] # URLs
            writer = csv.DictWriter(out, fieldnames=fieldnames)
            writer.writeheader()
            for row in reader:
                if tweet_id >= max:
                    break
                tweet = row[tweet_column]  # human 19, bot 2
                clean_tweet = english_tweet_cleaner(tweet)
                if clean_tweet is not '':
                    tweet_id += 1
                    tweet_dict = extract_stats(tweet, tweet_id, human_or_bot)
                    stored_info.append(tweet_dict)
                    writer.writerow(tweet_dict)
        return stored_info


def draw_scatterplot(statistics):
    X = []
    Y = []
    for line in statistics:
        # print(line)
        if line['TTR'] < 2:
            X.append(line['char count'])
            Y.append(line['TTR'])
    plt.scatter(X, Y)
    plt.title('char length VS TTR')

def ttr_curve(tweet_tokens):
    unique_tokens = []
    curve = []
    for token in tweet_tokens:
        if token not in unique_tokens:
            unique_tokens.append(token)
            if len(curve) == 0:
                curve.append(1)
            else:
                curve.append(curve[len(curve)-1]+1)
        else: curve.append(curve[len(curve)-1])
    return curve

def average_ttr_curve(stats):
    sum_curve = []
    avg_curve = []
    for el in stats:
        i = 0
        for c in el['ttr_curve']:
            if len(sum_curve) > i:
                sum_curve[i] += c
            else:
                sum_curve.append(c)
            i += 1

    for el in sum_curve:
        avg_curve.append(el/len(stats))

    return avg_curve

# this function extracts the features we want to use for classfication
def unpack_stats(stats, featurelist):
    featureset = []
    for tweet in stats:
        features = {}
        for key in tweet.keys():
            if key in featurelist:
                features[key] = tweet[key]
        featureset.append(features)
    # print(featureset)
    return featureset


def train_and_test_classifiers(human_stats, bot_stats, feature_list):
    human_featureset = unpack_stats(human_stats, feature_list)
    bot_featureset = unpack_stats(bot_stats, feature_list)

    labeled_tweets = ([(tweet, 'human') for tweet in human_featureset] + [(tweet, 'bot') for tweet in bot_featureset])
    random.shuffle(labeled_tweets)

    # training set is the first 80% tweets and the test set is the last 20%
    n = len(labeled_tweets)*20/100
    train_set = labeled_tweets[int(n):]
    test_set = labeled_tweets[:int(n)]
    print('\nTraining(' + str(len(train_set)) + ') and testing(' + str(len(test_set)) + ') set prepared!')

    experiment = {'Features used': str(feature_list)}

    classifier = nltk.NaiveBayesClassifier.train(train_set)
    # print('Accuracy with Naive Bayes is: ' + str(nltk.classify.accuracy(classifier, test_set)))
    # classifier.show_most_informative_features(10)

    classifier2 = nltk.MaxentClassifier.train(train_set)
    # print('\nAccuracy with MaxEnt is: ' + str(nltk.classify.accuracy(classifier2, test_set)))
    # classifier2.show_most_informative_features(10)
    experiment['Accuracy for NB classifier'] = nltk.classify.accuracy(classifier, test_set)

    dicNB = calculate_precision_and_recall(classifier, test_set)
    experiment['Precision for NB classifier'] = round(dicNB['precision'],4)
    experiment['Recall for NB classifier'] = round(dicNB['recall'],4)
    experiment['F1 for NB classifier'] = round(dicNB['f1'], 4)

    dicME = calculate_precision_and_recall(classifier2, test_set)
    experiment['Accuracy for ME classifier'] = nltk.classify.accuracy(classifier, test_set)
    experiment['Precision for ME classifier'] = round(dicME['precision'], 4)
    experiment['Recall for ME classifier'] = round(dicME['recall'], 4)
    experiment['F1 for ME classifier'] = round(dicME['f1'],4)

    return experiment

