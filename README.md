# Toxic Comment Classification
**Author**: *[Shifat Ali]*  
*Course Project – DA 623: Multimodal Learning (Winter 2025)*  

---

## 📌 Motivation

Toxic and abusive comments are pervasive across online platforms. They not only create a hostile environment for users but also pose serious moderation challenges for social media platforms. Manual filtering is unscalable and prone to subjectivity, making automated toxicity detection a critical task in natural language processing.

I chose this topic because I wanted to work on a problem that has real-world societal impact. The task also allowed me to explore classical and modern NLP pipelines while experimenting with model interpretability, class imbalance, and evaluation metrics.

---

## 🔍 Connecting with Multimodal Learning

While this project focuses on **textual comments**, toxic content on the internet is inherently **multimodal** — hateful messages can occur via **voice, video, memes, or even in multi-language code-switching conversations**.

The techniques used here — such as word embeddings, sequence modeling, and classification — are foundational to many multimodal models. For example:
- **Visual-text fusion** (memes with offensive text)
- **Audio-text alignment** (hate speech in speech transcripts)
- **Multimodal moderation systems** (combining user profiles, audio, and text for better detection)

This project thus serves as the **textual foundation** for a future multimodal toxicity detection system.

---

## 📘 Learnings from This Project

Throughout this project, I gained a deeper understanding of:
- Preprocessing pipelines (removing punctuation, stopwords, stemming)
- Using TF-IDF vectorization and CountVectorizer to convert text into features
- Training classification models like Logistic Regression and Multinomial Naive Bayes
- Handling **multi-label classification**, as each comment can belong to multiple toxicity classes
- Evaluating models with confusion matrices, classification reports, and accuracy/F1 scores
- Visualizing class imbalance and exploring correlation between labels

I also learned the importance of **data preprocessing and representation** in NLP — raw text data rarely performs well without thoughtful transformation.

---

## 🧪 Notebook Structure & Code Overview

The main Jupyter notebook (`toxic-comment-classification.ipynb`) walks through all stages of the project:

### ✅ Sections Covered:

1. **Dataset Loading & Exploration**
   - Uses Jigsaw Toxic Comment dataset from Kaggle
   - Multi-label targets: toxic, severe_toxic, obscene, threat, insult, identity_hate

2. **Data Preprocessing**
   - Text cleaning: lowercase, remove punctuation, stopwords
   - Tokenization and stemming using NLTK

3. **EDA**
   - Label distributions
   - Correlation heatmap across toxicity types
   - Word clouds per class

4. **Vectorization**
   - TF-IDF and CountVectorizer transformations

5. **Model Building**
   - Multinomial Naive Bayes
   - Logistic Regression (one-vs-rest for multilabel)

6. **Evaluation**
   - Classification reports
   - Accuracy, Precision, Recall, F1-score
   - Confusion Matrix visualization

7. **Results & Observations**
   - Logistic Regression outperformed Naive Bayes in most categories
   - Threat and severe_toxic were hard to detect due to class imbalance

8. **Bonus Ideas**
   - BERT-based classifier can be explored in future
   - Multimodal expansion using metadata, audio, or visual content

---

## 🤔 Reflections

### 💭 What Surprised Me
- Even basic models like Logistic Regression perform decently with TF-IDF representations.
- The high correlation between certain labels (e.g., *toxic* and *insult*) was unexpected.
- Data imbalance is a major challenge in multi-label classification — some labels have <1% samples.

### 🚀 Scope for Improvement
- Use BERT embeddings or fine-tuned transformer models (e.g., `bert-base-uncased`)
- Apply class imbalance techniques like SMOTE or focal loss
- Extend to **cross-lingual** or **code-switched** toxic comments
- Incorporate audio and visual modalities to create a full-spectrum toxicity detection system
- Deploy as an API to filter toxic comments in real time

---

## 📚 References

- [Jigsaw Toxic Comment Classification Dataset – Kaggle](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [NLTK – Natural Language Toolkit](https://www.nltk.org/)
- [Matplotlib](https://matplotlib.org/) & [Seaborn](https://seaborn.pydata.org/)
- HuggingFace Transformers (for potential future BERT models)
- Tools: Python, Jupyter Notebook, Google Colab, GitHub
- Assistance from ChatGPT for documentation polishing and reflection ideas

---

## 💻 Running the Notebook

To run the notebook:

1. Clone the repo:
   ```bash
   git clone https://github.com/your-username/toxic-comment-classification-blog.git
   cd toxic-comment-classification-blog
