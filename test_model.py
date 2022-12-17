from joblib import dump, load
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
import os, csv
from sklearn.metrics import accuracy_score, precision_score
from sklearn.model_selection import train_test_split





model = load("randomforest_model.joblib")

reviews_df_ML = pd.read_csv("sampled_data_tsv_1.csv", sep="\t", header=0, on_bad_lines="skip")
# reviews_df_ML.rename(columns=["marketplace", "customer_id", "review_id", "product_id", "product_parent", "product_title", "product_category", "star_rating", "helpful_votes", "total_votes", "vine", "verified_purchase", "review_headline", "review_body", "review_date"])
print(reviews_df_ML["review_body"])

X_test = pd.read_csv("X_test.csv")
y_test = pd.read_csv("y_test.csv").to_numpy()

# Extract labels
print("Extracting Labels...")
y_test = reviews_df_ML['star_rating'].values
# reviews_df_ML.drop(['star_rating'], axis=1, inplace=True)

X_test = X_test.iloc[:,0]


# Bag of words
vec = load("count_vectorizer.joblib")
X_test_BoW = vec.transform(X_test)

# Get model predictions
y_pred = model.predict(X_test_BoW)

# Create confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 3))
sns.heatmap(cm, annot=True, fmt='d')
plt.title(f'Confusion Matrix for Random Forest Classifier + TF-IDF')
plt.show()

# Get binary classification statistics (acc, precision, recall)
tn, fp, fn, tp = cm.ravel()
tn, fp, fn, tp
accuracy = accuracy_score(y_test, y_pred)
precision = tp/(tp+fp)
precision = precision_score(y_test, y_pred, average='micro')
recall = tp/(tp+fn)
print(f"=========== Random Forest Classifier + TF-IDF ===========")
print(f"Accuraccy: {accuracy}")
print(f"Precision Score: {precision}")
print(f"Recall Score: {recall}", end="\n\n")

# Write results to csv
flag = "a"
if not os.path.exists("output/model_data.csv"): flag = "w"
    
with open("output/model_data.csv", flag) as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["Random Forest", "TF-IDF", precision, accuracy])