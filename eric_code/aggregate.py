#!/usr/bin/env python2
import csv

from collections import Counter
from constants import VAL_DATA_DIR, VAL_IMG_LABELS, output

output_file = 'outputs/aggregate.csv'


def aggregate_predictions():
    fr_preds = read_csv('outputs/fr_cnn_51knn.csv')
    google_preds = read_csv('outputs/google.csv')
    facenet_preds = read_csv('outputs/facenet_model-20180402-114759.csv')

    predictions = []
    unknowns = []

    for l in VAL_IMG_LABELS:
        preds = [fr_preds[l], google_preds[l], facenet_preds[l]]

        filtered = filter(lambda x: 'unknown' not in x, preds)
        if len(filtered) == 0:
            print 'No label found accross all classifiers for {}\n'.format(VAL_DATA_DIR + l)
            unknowns.append((l, 'Unknown'))
        elif len(set(filtered)) == 1:
            predictions.append((l, filtered[0]))
        else:
            print 'Labels different for {}'.format(VAL_DATA_DIR + l)
            print '{} (fr) vs {} (google) vs {} (facenet)'.format(*preds)

            c = Counter(filtered)
            majority_label = c.most_common()[0][0]
            print 'Majority label: {}\n'.format(majority_label)
            predictions.append((l, majority_label))

    return predictions + unknowns


def read_csv(csv_file):
    output_dict = {}
    with open(csv_file, 'rb') as f:
        reader = csv.reader(f, delimiter=',', quotechar='|')
        for row in reader:
            # print row[0], row[1]
            output_dict[row[0]] = row[1]
    return output_dict


def diff(f1, f2):
    preds1 = read_csv(f1)
    preds2 = read_csv(f2)
    for l in VAL_IMG_LABELS:
        if preds1[l] != preds2[l]:
            print 'Labels different for {}'.format(VAL_DATA_DIR + l)
            print '{} vs {}\n'.format(preds1[l], preds2[l])


def main():
    preds = aggregate_predictions()
    # for i in preds:
    #     print i
    output(output_file, preds, header=True)


def test():
    diff('outputs/aggregate.csv', '/Users/ericmdai/Downloads/facenet3.csv')


if __name__ == '__main__':
    main()
    # test()
