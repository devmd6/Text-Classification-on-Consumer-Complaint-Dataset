# Text-Classification-on-Consumer-Complaint-Dataset


## Consumer Complaints Text Classification

This repository contains Python code for text classification of consumer complaints into predefined categories. The classification is based on the product mentioned in the complaint. The primary purpose is to categorize consumer complaints into four main categories ( Credit reporting, repair, or other; Debt collection; Consumer Loan; and Mortgage) amd to classify new complaints into the predefined categories based on their narratives.

We use two different models: Multinomial Naive Bayes and Linear Support Vector Classifier (Linear SVC). The dataset, loaded from a CSV file, contains consumer complaints along with their respective product categories. 
We utilize Python and the pandas library for data manipulation and analysis and scikit-learn for machine learning tasks, specifically for text classification using the Multinomial Naive Bayes algorithm and Linear Support Vector Classification Algorithm . Also Evaluate and Compare their Performance.
We Start of by extracting unique product categories from the 'Product' column and print them.
For Data Preprocessing , A new column 'Category' is created based on keywords in the 'Product' column. This helps in grouping similar complaints.
Rows with missing values in the 'Category' and 'Consumer complaint narrative' columns are dropped and Missing values in the 'Consumer complaint narrative' column are filled with empty strings.

Now , The dataset is split into training and testing sets using the train_test_split function from scikit-learn. The features (X_train and X_test) consist of the 'Consumer complaint narrative', and the target labels (y_train and y_test) are from the 'Category' column.

Vectorization of the data  - A TfidfVectorizer is used to convert text data into numerical features. The vectorizer is fit on the training set and transforms both the training and testing sets into TF-IDF features. A Multinomial Naive Bayes classifier (clf) and A Linear SVC classifier (linear_svc) are created and trained on the TF-IDF features of the training set .

Now for evaluating the models , The script predicts categories for the testing set and evaluates the model's performance using classification metrics such as precision, recall, and accuracy.
The classification report, accuracy, and confusion matrix are printed to assess the model's effectiveness.


New Complaint Prediction - A new consumer complaint is created as an example. The TfidfVectorizer is used to transform the new complaint, and the Multinomial Naive Bayes classifier and Linear SVC Classifier predicts its category. The predicted category is printed.

# Two Methods:

## Naive Bayes :

Naive Bayes is a probabilistic classification algorithm that relies on the assumption of independence between features. In text classification tasks, it has been widely used due to its simplicity, efficiency, and ease of interpretation. Naive Bayes works well when the independence assumption holds, making it particularly suited for applications like spam detection and sentiment analysis. However, it may struggle with imbalanced datasets and is sensitive to irrelevant features, limiting its effectiveness in certain scenarios.

## Linear Support Vector Classification(linear svc):

Linear Support Vector Classification is a robust and versatile algorithm widely applied in text classification tasks. Unlike Naive Bayes, LinearSVC does not assume feature independence and is capable of handling imbalanced datasets more effectively. In the context of text classification, where high-dimensional feature spaces are common, LinearSVC often performs well. It is computationally efficient and exhibits good generalization, making it suitable for large-scale text classification problems. LinearSVC's ability to handle irrelevant features and its robustness make it a compelling choice for tasks where feature interdependencies are crucial.

## Conclusion:

LinearSVC demonstrates superior performance based on the provided classification reports. LinearSVC exhibits higher precision, recall, and F1-score across all categories, resulting in an overall accuracy of approximately 89%, compared to Naive Bayes with an accuracy of around 85%. Notably, LinearSVC outperforms Naive Bayes in handling imbalanced datasets, as evidenced by its consistently strong performance across diverse categories.


The Dataset is available in the link - " https://catalog.data.gov/dataset/consumer-complaint-database "

The Classifier Code is available in the "Text_Classification.ipynb" file

# Output ScreenShots:

## Text Pre-Processing:

![Screenshot (232)](https://github.com/devmd6/Text-Classification-on-Consumer-Complaint-Dataset/assets/85011993/0434b7db-ad33-4c59-b1c8-1945ebffb540)

## Multinomial Naive Bayes Classifier Output:

![Screenshot (233)](https://github.com/devmd6/Text-Classification-on-Consumer-Complaint-Dataset/assets/85011993/86cdb888-496a-42d7-a02b-c8d6fad9a33a)

![image](https://github.com/devmd6/Text-Classification-on-Consumer-Complaint-Dataset/assets/85011993/65204b2d-1cf6-4a7f-9db0-a30b19b8fa40)

## Predicted Category of New Complaint using Multinomial Naive Bayes Classifier:

![Screenshot (234)](https://github.com/devmd6/Text-Classification-on-Consumer-Complaint-Dataset/assets/85011993/3b6c382e-ecee-481a-9584-d704275c12ec)

![image](https://github.com/devmd6/Text-Classification-on-Consumer-Complaint-Dataset/assets/85011993/89ed36b6-f5fa-473b-904a-ef8154b9d27f)

## Linear SVC classifier Output:

![Screenshot (235)](https://github.com/devmd6/Text-Classification-on-Consumer-Complaint-Dataset/assets/85011993/33004201-66fe-4983-842d-868e7d004124)

![image](https://github.com/devmd6/Text-Classification-on-Consumer-Complaint-Dataset/assets/85011993/9d9a3095-15d5-4495-b295-1c0362c19739)

## Predicted Category of New Complaint using Linear SVC Classifier:

![image](https://github.com/devmd6/Text-Classification-on-Consumer-Complaint-Dataset/assets/85011993/a6973b27-d08b-4ed5-abd8-605047cb5c67)







