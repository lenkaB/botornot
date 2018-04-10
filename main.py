from util import *

MAX = 500

if __name__ == '__main__':
    print('Loading human tweets...')
    datasets = ['human17', 'bot_social3']
    human_stats = load_dataset_and_print_stats(human_or_bot = datasets[0], max = MAX)
    print('Human tweets loaded and stats printed!')
    print('Loading bot tweets...')
    bot_stats = load_dataset_and_print_stats(human_or_bot = datasets[1], max = MAX)
    print('Bot tweets loaded and stats printed!')

    #draw_scatterplot(human_stats)
    #draw_scatterplot(bot_stats)
    #plt.show()

    avg_human_curve = average_ttr_curve(human_stats)
    avg_bot_curve = average_ttr_curve(bot_stats)

    #print(avg_human_curve)
    #print(avg_bot_curve)

    plot_ttr_curve(avg_human_curve,avg_bot_curve, datasets)



    complete_feature_list = ['char count', 'pos','pronouns', 'nouns', 'verbs', 'adverbs', 'adjectives',
                          'symbols', 'punctuation', 'proper nouns', 'entity label',
                          'word count', 'unique word count', 'TTR', 'mentions', 'hashtags','urls',
                          'count(rel words)','relationship words', 'ttr curve', 'ttr slope']

    current_feature_list = ['char count', 'pos','pronouns', 'nouns', 'verbs', 'adverbs', 'adjectives',
                          'symbols', 'punctuation', 'proper nouns',
                          'word count', 'unique word count', 'TTR', 'mentions', 'hashtags','urls',
                          'count(rel words)','relationship words', 'ttr curve', 'ttr slope']


    experiments, max = conduct_experiments(current_feature_list, human_stats, bot_stats, deepen = 1)

    #print('\n\n\n--------------FIRST SET OF EXPERIMENTS OVER -------------------- \n\n\n\n\n')

    fieldnames = ['Features used','char count', 'pos','pronouns', 'nouns', 'verbs', 'adverbs', 'adjectives',
                          'symbols', 'punctuation', 'proper nouns', 'entity label',
                          'word count', 'unique word count', 'TTR', 'mentions', 'hashtags','urls',
                          'count(rel words)','relationship words', 'ttr curve', 'ttr slope'
        ,'Accuracy for NB classifier','Precision for NB classifier',
    'Recall for NB classifier','F1 for NB classifier','Accuracy for ME classifier','Precision for ME classifier',
    'Recall for ME classifier','F1 for ME classifier']

    res = str(datasets[0]) + '-' + str(datasets[1]) + '[' + str(MAX) + ']LVL2.csv'

    with open(res, 'a', encoding="ISO-8859-1") as out:
        writer = csv.DictWriter(out, fieldnames=fieldnames)
        writer.writeheader()

        explore = print_experiments_and_return_suggestions(writer, experiments, max, complete_feature_list)

        for el in explore:
            experiments, max = conduct_experiments(el, human_stats, bot_stats, deepen = 1)
            new_explore = print_experiments_and_return_suggestions(writer, experiments, max, complete_feature_list)
            for new_el in new_explore:
                experiments, max = conduct_experiments(new_el, human_stats, bot_stats, deepen=1)
                print_experiments_and_return_suggestions(writer, experiments, max, complete_feature_list)
                


