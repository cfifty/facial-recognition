#!/usr/bin/env python2
import io

from constants import CLASS_LABELS, VAL_IMG_LABELS, VAL_IMG_PATHS, output
from google.cloud import vision

output_file = 'outputs/google-test.csv'


def predict_all():
    preds = []
    for label, path in zip(VAL_IMG_LABELS, VAL_IMG_PATHS):
        preds.append((label, predict(path)))
    return preds


def predict(path):
    label = 'google-unknown'

    web_entities = detect_web_entities(path)
    for entity in web_entities:
        guess = entity.description.replace(' ', '_').lower()
        if guess in CLASS_LABELS:
            label = guess
            break

    print 'Predicted label: {}'.format(label)
    return label


def detect_web_entities(path):
    print 'Detecting web entities for %s...' % path
    client = vision.ImageAnnotatorClient()

    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    image = vision.types.Image(content=content)

    response = client.web_detection(image=image)
    annotations = response.web_detection

    return annotations.web_entities


def main():
    preds = predict_all()
    output(output_file, preds)


if __name__ == '__main__':
    main()
