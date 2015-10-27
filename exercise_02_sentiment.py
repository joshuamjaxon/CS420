"""Build a sentiment analysis / polarity model

Sentiment analysis can be casted as a binary text classification problem,
that is fitting a linear classifier on features extracted from the text
of the user messages so as to guess wether the opinion of the author is
positive or negative.

In this examples we will use a movie review dataset.

"""
# Author: Olivier Grisel <olivier.grisel@ensta.org>
# License: Simplified BSD

import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.datasets import load_files
from sklearn.cross_validation import train_test_split
from sklearn import metrics


if __name__ == "__main__":
    # NOTE: we put the following in a 'if __name__ == "__main__"' protected
    # block to be able to use a multi-core grid search that also works under
    # Windows, see: http://docs.python.org/library/multiprocessing.html#windows
    # The multiprocessing module is used as the backend of joblib.Parallel
    # that is used when n_jobs != 1 in GridSearchCV

    # the training data folder must be passed as first argument
    movie_reviews_data_folder = sys.argv[1]
    dataset = load_files(movie_reviews_data_folder, shuffle=False)
    print("n_samples: %d" % len(dataset.data))

    # split the dataset in training and test set:
    docs_train, docs_test, y_train, y_test = train_test_split(
        dataset.data, dataset.target, test_size=0.25, random_state=None)

    # TASK: Build a vectorizer / classifier pipeline that filters out tokens
    # that are too rare or too frequent
    clf = Pipeline([('vect', TfidfVectorizer()), ('clf', LinearSVC())])
    
    # TASK: Build a grid search to find out whether unigrams or bigrams are
    # more useful.
    # Fit the pipeline on the training set using grid search for the parameters
    parameters = {'vect__analyzer': ['char', 'word'],
                  'vect__ngram_range': [(1, 1), (2, 2)],
                  #'vect__min_df': [0, 1, 0.1, 0.2, 0.3],  These values, when checked, show that 0 is the best.
                  #'vect__max_df': [0.7, 0.8, 0.9, 1.0],   These values, when checked, show 0.7 or 0.9 is the best.
                  'vect__min_df': [0],
                  'vect__max_df': [0.9]
                 }
    # While repeated testing suggests that 0 is the best min_df value, 0.7 and 0.9 seem to be the best
    # values for max_df depending on the set. 0.9 seems to be higher more frequently, so we'll use that as default.
    gs_clf = GridSearchCV(clf, parameters, n_jobs=3)
    gs_clf.fit(docs_train, y_train)

    # TASK: print the cross-validated scores for the each parameters set
    # explored by the grid search
    best_parameters, scores, _ = max(gs_clf.grid_scores_, key=lambda x: x[1])
    for param in sorted(parameters.keys()):
        print("{0}: {1}".format(param, best_parameters[param]))

    # TASK: Predict the outcome on the testing set and store it in a variable
    # named y_predicted
    y_predicted = gs_clf.predict(docs_test)

    # Print the classification report
    print(metrics.classification_report(y_test, y_predicted,
                                        target_names=dataset.target_names))

    # Print and plot the confusion matrix
    cm = metrics.confusion_matrix(y_test, y_predicted)
    print(cm)
    
    # Three reviews from Rotten Tomatoes:
    print("Martian Review: {0}".format(dataset.target_names[gs_clf.predict(["Smart, thrilling, and surprisingly funny, The Martian offers a faithful adaptation of the bestselling book that brings out the best in leading man Matt Damon and director Ridley Scott."])]))

    print("Hot Tub Time Machine: {0}".format(dataset.target_names[gs_clf.predict(["Its flagrantly silly script -- and immensely likable cast -- make up for most of its flaws."])]))
    
    print("Terminator Genisys: {0}".format(dataset.target_names[gs_clf.predict(["Mired in its muddled mythology, Terminator: Genisys is a lurching retread that lacks the thematic depth, conceptual intelligence, or visual thrills that launched this once-mighty franchise."])]))
    
    
    # import matplotlib.pyplot as plt
    # plt.matshow(cm)
    # plt.show()
