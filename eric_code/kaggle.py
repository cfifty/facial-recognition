#!/usr/bin/env python2
import csv
import face_recognition as fr
import numpy as np
import os
import sqlite3

db_file = 'labed_encodings.db'

train_data_dir = './images-train'
val_data_dir = './images-val-pub'

output_file = 'submission.csv'


def setup_db(f):
    conn = sqlite3.connect(f)
    c = conn.cursor()
    c.execute('CREATE TABLE IF NOT EXISTS labed_encodings (label TEXT PRIMARY KEY, encodings TEXT);')
    conn.commit()


def train():
    labels = sorted(os.listdir(train_data_dir))
    for l in labels:
        print 'Getting encodings for %s' % l
        encodings = get_encodings('{}/{}'.format(train_data_dir, l))
        save_encoding(l, encodings)


def get_encodings(label_dir, num_jitters=1):
    images = map(lambda f: '{}/{}'.format(label_dir, f), os.listdir(label_dir))
    loaded_images = map(lambda f: fr.api.load_image_file(f), images)
    encodings = map(lambda img: fr.api.face_encodings(img, num_jitters), loaded_images)
    filtered_encodings = map(lambda x: x[0], filter(lambda x: len(x) > 0, encodings))
    return filtered_encodings


def save_encoding(label, encodings):
    print 'Saving encodings for %s' % label
    encodings_str = repr(map(lambda x: x.tostring(), encodings))
    conn = sqlite3.connect(db_file)
    c = conn.cursor()
    c.execute('INSERT INTO labed_encodings values (?,?)', (label, encodings_str))
    conn.commit()


def load_encoding(row):
    label, encoding_str = row
    encoding = map(np.fromstring, eval(encoding_str))
    return label, encoding


def predict_all(writer=None):
    files = sorted(os.listdir(val_data_dir))
    for f in files:
        pred = predict('{}/{}'.format(val_data_dir, f))
        print "Precition for %s: %s" % (f, pred)
        if writer is not None:
            writer.writerow([f, pred])


def predict(img_file, num_jitters=1):
    img = fr.api.load_image_file(img_file)
    try:
        unknown_encoding = fr.api.face_encodings(img, num_jitters)[0]
    except Exception as e:
        print "Failed to get face encoding: %s" % str(e)
        return

    pred_label = ''
    pred_dist = np.inf

    conn = sqlite3.connect(db_file)
    c = conn.cursor()
    for row in c.execute('SELECT * FROM labed_encodings ORDER BY label'):
        label, encodings = load_encoding(row)
        face_distances = fr.api.face_distance(encodings, unknown_encoding)
        if min(face_distances) < pred_dist:
            pred_label = label
            pred_dist = min(face_distances)

    return pred_label


def main():
    # Training only needs to be done once
    # setup_db(db_file)
    # train()

    with open(output_file, 'wb') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(['image_label', 'celebrity_name'])
        predict_all(writer)


if __name__ == '__main__':
    main()
