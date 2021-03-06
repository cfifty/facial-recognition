# Cornell CS4780 SP 2018 Final Project
*Link:* https://www.kaggle.com/c/cs4780sp2018finalproject#description

Welcome to the CS4780 final project! Unlike the other projects you have completed this semester, you will be given free reign to approach this problem however you like. You are free to use anything you learned in this class as well as anything outside of the scope of the course.

## Dataset
The dataset you are working with is comprised of images of 98 different celebrities. You are tasked with making a model that can properly classify the celebrity that is in the image. The data are given as a zipped file of jpeg images and a JSON that maps the image's filename to its label. We have also provided a 2048 dimensional feature vector for each image in the form of a JSON with its respective label, this feature vector is generated from the convolutional layers of a pre-trained residual neural network. These features are purely optional to use but may be helpful in your implementation. We have provided resources about convolutional layers and residual network architectures under the Resources section if you want to learn more about them.

## Submission
Team: You could work on a team with size up to 5 students.

Submissions: To get full credits on the competition, you are required to submit all of the following on cms:

A prediction for both the validation set and the testing set (submit through Kaggle)
A one-page short report describing your approach
A zip file containing your codes and a README with instructions to reproduce your competition results.

## Academic Dishonesty
We will be checking your code against other submissions in the class for logical redundancy. If you copy someone else's code and submit it with minor changes, we will know. These cheat detectors are quite hard to fool, so please don't try! If you do, we will pursue the strongest consequences available to us.

*IMPORTANT:* Given the small size of the testing dataset, trying to hand-label the test dataset and claim credits for such hand-labeled results is NOT ALLOWED. Almost everyone could be a good annotator for this task, and we are not testing your ability to recognize faces here. Your submission has to be outputs of your algorithm.

## Getting Help
You are not alone! If you find yourself stuck on something, contact the course staff for help. Office hours, section, and the Piazza are there for your support; please use them. If you can't make our office hours, let us know and we will schedule more. We want these projects to be rewarding and instructional, not frustrating and demoralizing. But, we don't know when or how to help unless you ask.

## Resources
Residual Networks Paper

Convolutional Layers

The pretrained ResNet used for feature extraction

## Competition Score
The evaluation metrics for this competition is classification accuracy. Your score depends on which range of accuracy your final submission gets into. The cut off of the range for accuracy will be determined by several benchmarks created by the course staff team, these benchmarks are basic algorithms that we've learned in class with little hyer-parameter tuning. To get a score in the higher A range you need to beat all the TA benchmarks. The TA teams are still working to bring out more benchmarks in the A range.

The dataset is split into three sets, 50% of it is the training set, 25% of it is the validation, and another 25% is the test set. The training set and the validation set are provided in the Data page. We will release the test set one day before the end of the competition. You will submit to the leaderboard predictions of the images in the validation set, and the leaderboard will show accuracy computed with 50% of the validation set data. Note that we plan to evaluate your model's performance with a large amount of held-out data. Please try not to overfit with one specific score.

## Project Score
The project score will be your competition score (computed as how many benchmarks have your submission managed to beat), provided that your short report and codes consistently reflect the competition results. In the short report, please include what algorithm you use, how you preprocess the data, and how you train the classifier. In README of your CMS submission, please provide the dependency of your codes, the instruction to train the classifier, and the instruction to produce solution file you submit in Kaggle.

This project score will count as an extra credit in your course score.

## Kaggle Submission Format
The submission file is a csv file (i.e. delimiter is ",") with two the following columns: image_label and celebrity_name. The image_label column contains the filename of the corresponding image in the testing dataset. There are approximately 2500 images in the given validation and test set. The celebrity_name column contains the prediction of the image with corresponding id in the image column. The prediction should be the name representing the predicted celebrity. The file should contain a header and have the following format:

```
image_label,celebrity_name
e4ef6c98fabf3a3759a66612a63cc04c.jpg,adam_sandler
d1634dfa0076e5ba357c6c60f10e8ba1.jpg,albert_einstein
6be6e0a3439ab8c15af6115e94fa74f3.jpg,kanye_west
34021274896307af51da411a9444fa10.jpg,charlie_chaplin
...
```

