# Text-Classification-on-Consumer-Complaint-Dataset

We utilize Python and the pandas library for data manipulation and analysis and scikit-learn for machine learning tasks, specifically for text classification using the Multinomial Naive Bayes algorithm.
We extract unique product categories from the 'Product' column and print them.
For Data Preprocessing , A new column 'Category' is created based on keywords in the 'Product' column. This helps in grouping similar complaints.
Rows with missing values in the 'Category' and 'Consumer complaint narrative' columns are dropped and Missing values in the 'Consumer complaint narrative' column are filled with empty strings.

Now , The dataset is split into training and testing sets using the train_test_split function from scikit-learn. The features (X_train and X_test) consist of the 'Consumer complaint narrative', and the target labels (y_train and y_test) are from the 'Category' column.

Vectorization of the data  - A TfidfVectorizer is used to convert text data into numerical features. The vectorizer is fit on the training set and transforms both the training and testing sets into TF-IDF features. A Multinomial Naive Bayes classifier (clf) is created and trained on the TF-IDF features of the training set.


Now for evaluating the model , The script predicts categories for the testing set and evaluates the model's performance using classification metrics such as precision, recall, and accuracy.
The classification report, accuracy, and confusion matrix are printed to assess the model's effectiveness.


New Complaint Prediction - A new consumer complaint is created as an example. The TfidfVectorizer is used to transform the new complaint, and the Multinomial Naive Bayes classifier predicts its category. The predicted category is printed.

Two Methods:

naive bayes :
new_complaint = ["I have an issue with my mortgage payment."]
new_complaint_tfidf = vectorizer.transform(new_complaint)
predicted_category = clf.predict(new_complaint_tfidf)
print("\nPredicted Category for New Complaint:", predicted_category[0])

Random forest:
new_complaint = ["I have an issue with my mortgage payment."]
new_complaint_tfidf = vectorizer.transform(new_complaint)
predicted_category = rf_classifier.predict(new_complaint_tfidf)
print("\nPredicted Category for New Complaint:", predicted_category[0])
