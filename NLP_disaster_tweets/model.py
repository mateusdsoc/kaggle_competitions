import os
import pandas as pd
from sklearn import feature_extraction, linear_model

script_dir = os.path.dirname(os.path.abspath(__file__))

train_df = pd.read_csv(os.path.join(script_dir, 'data/train.csv'))
test_df = pd.read_csv(os.path.join(script_dir, 'data/test.csv'))
sample_submission = pd.read_csv(os.path.join(script_dir, 'data/sample_submission.csv'))


count_vectorizer = feature_extraction.text.CountVectorizer()

train_vectors = count_vectorizer.fit_transform(train_df["text"])
test_vectors = count_vectorizer.transform(test_df["text"])

clf = linear_model.RidgeClassifier()
clf.fit(train_vectors, train_df["target"])


sample_submission["target"] = clf.predict(test_vectors)
sample_submission.to_csv(os.path.join(script_dir, "submission.csv"), index=False)