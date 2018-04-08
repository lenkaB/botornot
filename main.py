from util import *

MAX = 1000

if __name__ == '__main__':
    print('Loading human tweets...')
    human_stats = load_dataset_and_print_stats(human_or_bot = 'human', max = MAX)
    print('Human tweets loaded and stats printed!')
    print('Loading bot tweets...')
    bot_stats = load_dataset_and_print_stats(human_or_bot = 'bot', max = MAX)
    print('Bot tweets loaded and stats printed!')
    #draw_scatterplot(human_stats)
    #draw_scatterplot(bot_stats)
    #plt.show()

    avg_human_curve = average_ttr_curve(human_stats)
    avg_bot_curve = average_ttr_curve(bot_stats)

    print(avg_human_curve)
    print(avg_bot_curve)

    plt.plot(avg_human_curve,color='red')
    plt.plot(avg_bot_curve)
    import matplotlib.patches as mpatches
    red_patch = mpatches.Patch(color='red', label='Humans')
    blue_patch = mpatches.Patch(color='blue', label='Bots')
    plt.legend(handles=[red_patch, blue_patch])

    plt.show()

    feature_list = ['char count', 'pronouns', 'nouns', 'verbs', 'adjectives', 'punctuation',
                    'symbols', 'adverbs', 'proper nouns', 'word count', 'unique word count', 'TTR',
                    'mentions', 'hashtags', 'urls', 'count(rel words)']
    feature_list_copy = feature_list.copy()
    experiments = []
    previous_feature = None

    experiments.append(train_and_test_classifiers(human_stats,bot_stats,feature_list))


    for feature in feature_list_copy:
        feature_list.remove(feature)
        if previous_feature is not None:
            feature_list.append(previous_feature)
        experiments.append(train_and_test_classifiers(human_stats,bot_stats,feature_list))
        previous_feature=feature



    fieldnames = ['Features used','Accuracy for NB classifier','Precision for NB classifier',
    'Recall for NB classifier','F1 for NB classifier','Accuracy for ME classifier','Precision for ME classifier',
    'Recall for ME classifier','F1 for ME classifier']


    with open('human2017fakefol15.csv', 'a', encoding="ISO-8859-1") as out:

        writer = csv.DictWriter(out, fieldnames=fieldnames)
        writer.writeheader()

        for el in experiments:
            writer.writerow(el)
