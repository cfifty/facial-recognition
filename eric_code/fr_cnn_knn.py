#!/usr/bin/env python2
import face_recognition as fr
import os
import pickle

from constants import TRAIN_DATA_DIR, CLASS_LABELS, VAL_IMG_LABELS, VAL_IMG_PATHS, output
from face_recognition.face_recognition_cli import image_files_in_folder
from sklearn import neighbors

n_neighbors = 51
fr_model_path = 'models/fr_cnn_{}knn.clf'.format(n_neighbors)
output_file = 'outputs/fr_cnn_{}knn.csv'.format(n_neighbors)


def train(train_dir, model_save_path=None, n_neighbors=None, knn_algo='ball_tree', verbose=True):
    """
    Trains a k-nearest neighbors classifier for face recognition.
    :param train_dir: directory that contains a sub-directory for each known person, with its name.
     (View in source code to see train_dir example tree structure)
     Structure:

    :param model_save_path: (optional) path to save model on disk
    :param n_neighbors: (optional) number of neighbors to weigh in classification. Chosen automatically if not specified
    :param knn_algo: (optional) underlying data structure to support knn.default is ball_tree
    :param verbose: verbosity of training
    :return: returns knn classifier that was trained on the given data.
    """
    X = []
    y = []

    # Loop through each person in the training set
    for class_dir in os.listdir(train_dir):
        if not os.path.isdir(os.path.join(train_dir, class_dir)):
            continue

        # Loop through each training image for the current person
        for img_path in image_files_in_folder(os.path.join(train_dir, class_dir)):
            image = fr.load_image_file(img_path)
            face_bounding_boxes = fr.face_locations(image, model='cnn')

            if len(face_bounding_boxes) != 1:
                # If there are no people (or too many people) in a training image, skip the image.
                if verbose:
                    print("Image {} not suitable for training: {}".format(img_path,
                                                                          "Didn't find a face" if len(face_bounding_boxes) < 1 else "Found more than one face"))
            else:
                # Add face encoding for current image to the training set
                X.append(fr.face_encodings(image, known_face_locations=face_bounding_boxes)[0])
                y.append(class_dir)

    # Determine how many neighbors to use for weighting in the KNN classifier
    if n_neighbors is None:
        n_neighbors = int(round(math.sqrt(len(X))))
        if verbose:
            print("Chose n_neighbors automatically:", n_neighbors)

    # Create and train the KNN classifier
    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
    knn_clf.fit(X, y)

    # Save the trained KNN classifier
    if model_save_path is not None:
        with open(model_save_path, 'wb') as f:
            pickle.dump(knn_clf, f)

    return knn_clf


def predict_all():
    preds = []
    for label, path in zip(VAL_IMG_LABELS, VAL_IMG_PATHS):
        print 'Predicting {}:'.format(path)
        pred = predict_knn(path, model_path=fr_model_path)
        print 'Predicted label: {}\n'.format(pred)
        preds.append((label, pred))
    return preds


def predict_knn(X_img_path, knn_clf=None, model_path=None, distance_threshold=0.6):
    """
    Recognizes faces in given image using a trained KNN classifier
    :param X_img_path: path to image to be recognized
    :param knn_clf: (optional) a knn classifier object. if not specified, model_save_path must be specified.
    :param model_path: (optional) path to a pickled knn classifier. if not specified, model_save_path must be knn_clf.
    :param distance_threshold: (optional) distance threshold for face classification. the larger it is, the more chance
           of mis-classifying an unknown person as a known one.
    :return: a list of names and face locations for the recognized faces in the image: [(name, bounding box), ...].
        For faces of unrecognized persons, the name 'unknown' will be returned.
    """
    if not os.path.isfile(X_img_path) or os.path.splitext(X_img_path)[1][1:] not in {'png', 'jpg', 'jpeg'}:
        raise Exception("Invalid image path: {}".format(X_img_path))

    if knn_clf is None and model_path is None:
        raise Exception("Must supply knn classifier either thourgh knn_clf or model_path")

    # Load a trained KNN model (if one was passed in)
    if knn_clf is None:
        with open(model_path, 'rb') as f:
            knn_clf = pickle.load(f)

    # Load image file and find face locations
    X_img = fr.load_image_file(X_img_path)
    X_face_locations = fr.face_locations(X_img, model="cnn")

    # If no faces are found in the image, return an empty result.
    if len(X_face_locations) == 0:
        return 'fr_unknown'

    # Find encodings for faces in the test iamge
    faces_encodings = fr.face_encodings(X_img, known_face_locations=X_face_locations)

    # Use the KNN model to find the best matches for the test face
    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
    are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]

    # Predict classes and remove classifications that aren't within the threshold
    preds = [(pred, loc) if rec else ("unknown", loc)
             for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]
    return preds[0][0]


def main():
    # Training only needs to be done once
    # print 'Training knn classifier with n={}'.format(n_neighbors)
    # classifier = train(TRAIN_DATA_DIR, model_save_path=fr_model_path, n_neighbors=n_neighbors)
    # print 'Training complete!'

    preds = predict_all()
    output(output_file, preds)


if __name__ == '__main__':
    main()
