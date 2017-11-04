#!/usr/bin/python

""" Complete the code in ClassifyNB.py with the sklearn
    Naive Bayes classifier to classify the terrain data.

    The objective of this exercise is to recreate the decision
    boundary found in the lesson video, and make a plot that
    visually shows the decision boundary """

if __name__=="__main__":
    from prep_terrain_data import makeTerrainData
    from class_vis import prettyPicture, output_image
    from ClassifyNB import classify

    import numpy as np
    import pylab as pl

    features_train, labels_train, features_test, labels_test = makeTerrainData()

    ### the training data (features_train, labels_train) have both "fast" and "slow" points mixed
    ### in together--separate them so we can give them different colors in the scatterplot,
    ### and visually identify them
    # grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii] == 0]
    # bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii] == 0]
    # grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii] == 1]
    # bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii] == 1]

    # You will need to complete this function imported from the ClassifyNB script.
    # Be sure to change to that code tab to complete this quiz.
    from time import time
    t0 = time()
    clf = classify(features_train, labels_train)
    print "traning time:", round(time()-t0, 3), "s"
    print 'Accuracy : %.3f' % clf.score(features_test, labels_test)

    t1 = time()
    pred = clf.predict(features_test)
    print 'predict time:', round(time()-t1, 3), 's'

    ### draw the decision boundary with the text points overlaid
    prettyPicture(clf, features_test, labels_test)
    output_image("test.png", "png", open("test.png", "rb").read())