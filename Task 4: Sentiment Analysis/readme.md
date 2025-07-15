# Twitter Sentiment Analysis with Naive Bayes Classifier
This project applies Natural Language Processing (NLP) and machine learning to classify the sentiment of tweets using a Naive Bayes model. The analysis includes preprocessing, feature extraction, model training, evaluation, and visualization of sentiments via bar charts and word clouds.

### Dataset
Filename: Twitter_Data.csv   https://www.kaggle.com/datasets/saurabhshahane/twitter-sentiment-dataset

Columns Used:

clean_text: Preprocessed tweet text

category: Sentiment label (e.g., positive, negative, neutral)

Install all dependencies using:

```bash
pip install -r requirements.txt
```

### Workflow Summary
1. Data Cleaning
Dropped rows with missing clean_text or category
Verified category distribution using .value_counts()

2. Text Vectorization
Used TfidfVectorizer to convert tweet text into numerical features
Limited to top 5000 features to avoid overfitting

3. Model Training – Naive Bayes
Split dataset into 80% training and 20% testing
Trained MultinomialNB() classifier
Made predictions on the test set

4. Evaluation
Classification Report showing:
Precision
Recall
F1-score for each sentiment class
Confusion Matrix heatmap to visualize misclassifications

```python
print(classification_report(y_test, y_pred))
sns.heatmap(confusion_matrix(...))
```

5. Visualizations
Sentiment Class Distribution
Bar chart showing counts for each sentiment

Confusion Matrix
Heatmap to evaluate model accuracy visually

Word Clouds by Sentiment
WordClouds generated for:
Positive, Negative, and Neutral tweets
Highlights most frequent words in each sentiment category

### Results Snapshot
Example output from classification report:
```markdown
              precision    recall  f1-score   support

    negative       0.89      0.87      0.88       500
     neutral       0.78      0.82      0.80       400
    positive       0.91      0.89      0.90       600
```

### Recommendations
Improve model accuracy by:

Using more advanced models (e.g., Logistic Regression, SVM, or BERT)

Adding stopword removal, stemming/lemmatization

Visualize keyword importance using feature weights

Deploy via Streamlit or Flask as a simple sentiment classifier app
