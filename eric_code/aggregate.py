#!/usr/bin/env python2
import csv

from collections import Counter
from constants import VAL_DATA_DIR, VAL_IMG_LABELS, output

output_file = 'outputs/aggregate.csv'


def aggregate_predictions():
    # fr_preds = read_csv('outputs/fr_cnn_51knn.csv')
    google_preds = read_csv('outputs/google.csv')
    facenet_preds = read_csv('outputs/facenet_model-20180402-114759.csv')

    predictions = []
    unknowns = []

    for l in VAL_IMG_LABELS:
        preds = [google_preds[l], facenet_preds[l]]
        if len(set(preds)) == 1:
            predictions.append((l, preds[0]))
        else:
            print 'Labels different for {}'.format(VAL_DATA_DIR + l)
            print '{} (google) vs {} (facenet)'.format(*preds)

            filtered = filter(lambda x: 'unknown' not in x, preds)
            if len(filtered) == 0:
                print 'No label found accross all classifiers\n'
                unknowns.append((l, 'Unknown'))
                continue

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


def main():
    preds = aggregate_predictions()
    # for i in preds:
    #     print i
    output(output_file, preds, header=True)


if __name__ == '__main__':
    main()
