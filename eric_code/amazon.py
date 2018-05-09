#!/usr/bin/env python2
import boto3
import json
import os

from constants import VAL_IMG_LABELS

bucket = 'cs4780-kaggle'
client = boto3.client('rekognition', 'us-east-2')


def log(x):
    print json.dumps(x, sort_keys=True, indent=4)


def celebrity(filename):
    return client.recognize_celebrities(Image={
        'S3Object': {
            'Bucket': bucket,
            'Name': filename,
        }
    })


def main():
    for img in VAL_IMG_LABELS:
        path = 'images-val-pub/' + img
        print path
        try:
            response = celebrity(path)
            faces = response['CelebrityFaces']
            if len(faces) == 0:
                # print '{},amazon_unknown'.format(img)
                print 'amazon_unknown'
            else:
                for f in faces:
                    # print '{},{},{}'.format(img, f['Name'], f['MatchConfidence'])
                    print '{},{}'.format(f['Name'], f['MatchConfidence'])
            # log(response)
        except Exception as e:
            print e
            # print '{},amazon_unknown'.format(img)
            print 'amazon_unknown'
        # break

    # print('Detected labels for ' + filename)
    # for label in response['Labels']:
    #     print(label['Name'] + ' : ' + str(label['Confidence']))

if __name__ == '__main__':
    main()
