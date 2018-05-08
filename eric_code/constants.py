import os

TRAIN_DATA_DIR = '/Users/ericmdai/Downloads/images-train/'
VAL_DATA_DIR = '/Users/ericmdai/Downloads/images-val-pub/'

CLASS_LABELS = sorted(os.listdir(TRAIN_DATA_DIR))
VAL_IMG_LABELS = sorted(os.listdir(VAL_DATA_DIR))
VAL_IMG_PATHS = map(lambda f: VAL_DATA_DIR + f, VAL_IMG_LABELS)


def output(preds):
    with open(output_file, 'wb') as f:
        writer = csv.writer(f, delimiter=',')
        # writer.writerow(['image_label', 'celebrity_name'])
        for img_label, pred in preds:
            writer.writerow([img_label, pred])
