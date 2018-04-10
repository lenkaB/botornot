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
from scipy.stats import linregress

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

    i = 1
    arr = []
    for pos in pos_list:
       arr.append(i)
       i += 1

    if len(arr):
        ttr_slope = linregress(arr, curve)[0]
    else:
        ttr_slope = 0

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
                  'ttr curve': curve,
                  'ttr slope': ttr_slope}

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

def conduct_experiments(feature_list, human_stats, bot_stats, deepen):
    feature_list_copy = feature_list.copy()
    experiments = []
    previous_feature = None

    print("CONDUCTING EXPERIMENTS WITH:"+str(feature_list))

    experiments.append(train_and_test_classifiers(human_stats, bot_stats, feature_list))

    curr_max = {'Accuracy for NB classifier': 0, 'Precision for NB classifier': 0,
                'Recall for NB classifier': 0, 'F1 for NB classifier': 0, 'Accuracy for ME classifier': 0,
                'Precision for ME classifier': 0,
                'Recall for ME classifier': 0, 'F1 for ME classifier': 0}

    if deepen:
        for feature in feature_list_copy:
            feature_list.remove(feature)
            if previous_feature is not None:
                feature_list.append(previous_feature)

            print("INDIVIDUAL EXPERIMENT FOR:"+str(feature_list))
            experiment = train_and_test_classifiers(human_stats, bot_stats, feature_list)

            for key in curr_max:
                if experiment[key] > curr_max[key]:
                    curr_max[key] = experiment[key]

            experiments.append(experiment)
            previous_feature = feature

    return experiments, curr_max


def plot_ttr_curve(avg_human_curve, avg_bot_curve, datasets):
        plt.plot(avg_human_curve,color='red')
        plt.plot(avg_bot_curve)
        import matplotlib.patches as mpatches
        red_patch = mpatches.Patch(color='red', label=datasets[0])
        blue_patch = mpatches.Patch(color='blue', label=datasets[1])
        plt.legend(handles=[red_patch, blue_patch])
        plt.show()


def load_dataset_and_print_stats(human_or_bot, max):
    tweet_id = 0
    # based on the argument of the function human_or_bot, we set the parameters for loading and storing
    if human_or_bot is 'human17':
        filename = '/Users/lenka/Desktop/humans2017.csv'
        output = '/Users/lenka/Desktop/output_human2017' + str(max) + '.csv'
        tweet_column = 1
    elif human_or_bot is 'bot_traditional':
        filename = '/Users/lenka/Desktop/traditional1.csv'
        output = '/Users/lenka/Desktop/output_bot_traditional1' + str(max) + '.csv'
        tweet_column = 1
    elif human_or_bot is 'bot_social2':
        filename = '/Users/lenka/Desktop/social2.csv'
        output = '/Users/lenka/Desktop/output_bot_social2' + str(max) + '.csv'
        tweet_column = 1
    elif human_or_bot is 'bot_social3':
        filename = '/Users/lenka/Desktop/social3.csv'
        output = '/Users/lenka/Desktop/output_bot_social3' + str(max) + '.csv'
        tweet_column = 1
    elif human_or_bot is 'bot_fakefol':
        filename = '/Users/lenka/Desktop/fakefol.csv'
        output = '/Users/lenka/Desktop/output_bot_fakefol' + str(max) + '.csv'
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
                          'relationship words', 'class', 'ttr curve', 'ttr slope'] # URLs
            writer = csv.DictWriter(out, fieldnames=fieldnames)
            writer.writeheader()
            for row in reader:
                if tweet_id >= max:
                    break
                tweet = row[tweet_column]  # human 19, bot 2
                clean_tweet = english_tweet_cleaner(tweet)
                if clean_tweet is not '' and clean_tweet and len(clean_tweet.split())>1:
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
        for c in el['ttr curve']:
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
                if type(tweet[key]) == list:
                    features[key] = str(tweet[key])
                else:
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

    print(feature_list)
    experiment = {'Features used': feature_list.copy()}

    classifier = nltk.NaiveBayesClassifier.train(train_set)
    classifier2 = nltk.MaxentClassifier.train(train_set, max_iter= 50)

    dicNB = calculate_precision_and_recall(classifier, test_set)
    experiment['Accuracy for NB classifier'] = nltk.classify.accuracy(classifier, test_set)
    experiment['Precision for NB classifier'] = round(dicNB['precision'],4)
    experiment['Recall for NB classifier'] = round(dicNB['recall'],4)
    experiment['F1 for NB classifier'] = round(dicNB['f1'], 4)

    dicME = calculate_precision_and_recall(classifier2, test_set)
    experiment['Accuracy for ME classifier'] = nltk.classify.accuracy(classifier, test_set)
    experiment['Precision for ME classifier'] = round(dicME['precision'], 4)
    experiment['Recall for ME classifier'] = round(dicME['recall'], 4)
    experiment['F1 for ME classifier'] = round(dicME['f1'],4)

    print(classifier.show_most_informative_features(n = 5))
    print(classifier2.show_most_informative_features(n = 5))

    return experiment

def print_experiments_and_return_suggestions(writer, experiments, max, complete_feature_list):
    explore =[]
    for exp in experiments:
        for el in max.keys():
            if exp[el] == max[el]:
                exp[el] = str(exp[el]) + ' *'
                if exp['Features used'] not in explore:
                    explore.append(exp['Features used'])

        for feature in complete_feature_list:
            if feature in exp['Features used']:
                exp[feature]=1
            else:
                exp[feature]=0
        writer.writerow(exp)
    return explore