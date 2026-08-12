# EDP AI/ML Internship - Week 3

## Project Title
Spam Message Classifier

## Week 3 Topics
- Text Preprocessing
- TF-IDF
- Naive Bayes

## Objective
The objective of this project is to build a machine learning model that classifies text messages as Spam or Ham (Not Spam).

## Dataset
A sample dataset containing SMS messages was created.

The dataset contains two categories:
- Spam
- Ham

## Steps Performed

1. Created a dataset containing spam and ham messages.
2. Converted the dataset into a Pandas DataFrame.
3. Selected messages as the input feature.
4. Selected spam/ham labels as the target.
5. Split the dataset into training and testing data.
6. Performed text preprocessing using lowercase conversion and English stop word removal.
7. Converted text data into numerical features using TF-IDF Vectorization.
8. Created a Multinomial Naive Bayes model.
9. Trained the model using the training data.
10. Predicted spam and ham messages using the testing data.
11. Evaluated the model using accuracy score and classification report.
12. Tested the trained model with new messages.

## Model Used

Multinomial Naive Bayes

## Text Processing Technique

TF-IDF Vectorization was used to convert text messages into numerical features that can be used by the machine learning model.

## Model Evaluation

The model achieved:

- Accuracy: 1.00
- Test messages correctly classified as Spam and Ham.

## Technologies Used

- Python
- Pandas
- Scikit-learn
- TF-IDF Vectorizer
- Multinomial Naive Bayes
- Visual Studio Code

## Technical Skills Learned

- Text Preprocessing
- TF-IDF Vectorization
- Natural Language Processing Basics
- Naive Bayes Classification
- Multinomial Naive Bayes
- Train/Test Split
- Machine Learning Model Training
- Text Classification
- Spam Detection
- Model Prediction
- Accuracy Score
- Classification Report
- Python
- Pandas
- Scikit-learn

## Conclusion

Successfully built and evaluated a Spam Message Classifier using TF-IDF Vectorization and Multinomial Naive Bayes. The trained model was able to classify new messages as Spam or Ham.