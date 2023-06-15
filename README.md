# Fake-News-Detection
___

## Problem Statement
Develop an effective and efficient fake news detection model that can identify and flag misleading or false information.

The primary objective of the model is to capture as many instances of fake news as possible, aiming to prevent their spreading before undergoing fact-checking. This requires minimizing the number of fake news articles incorrectly predicted as real. To evaluate the model's performance in achieving this objective, we will prioritize the use of the recall metric.

I also deployed the model on Huggingface.

Deployment: https://huggingface.co/spaces/aliaufa/Fake-News-Detection

The dataset used in this project comes from Kaggle which can be seen in the link below

Dataset : https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

## Methods used
1. Data Visualization
2. Feature Engineering
3. Natural Language Processing
4. Artificial Neural Network
5. Model Inference

## Technologies
1. Python
2. Pandas
3. Matplotlib
4. Wordcloud
5. nltk
6. Gensim's Word2Vec
7. Tensorflow
8. Streamlit
9. Huggingface

---

## Result

The model has recall value of 1.00 on class 1, indicating that almost 100% of the actual class 1 samples or fake news are correctly identified. From the confusion matrix we can see that 0.09% of prediction are false negatives. This means the improved model performed better than the previous model.

Comparing the top words from false positive and false negative data, we can observe the following insights:

- Overlapping Words: Both false positive and false negative data contain common words such as "said," "Trump," and "president." These words indicate the challenges in accurately classifying news articles related to political figures and statements.

- Misleading Context: False positive data includes words like "tax," "Republican," and "bill," which may indicate that certain political or legislative topics were incorrectly classified as fake news. On the other hand, false negative data includes words like "FBI," "nuclear," and "attack," which suggests that articles discussing sensitive topics were misclassified as real news.

- Speculative Language: False positive data includes words like "would," "state," and "year," which might indicate speculative or conditional statements that resemble fake news. False negative data includes words like "could" and "also," which suggest the presence of uncertain or speculative information.

These insights highlight the challenges of accurately detecting fake news, especially in the context of political content and sensitive topics. The misclassifications can occur due to the use of similar words, misleading context, or speculative language. To improve the accuracy of the model, it is important to incorporate additional features and context to better differentiate between genuine and fake news articles.
