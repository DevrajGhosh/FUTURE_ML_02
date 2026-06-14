# Support Ticket Classification & Prioritization

A machine learning project that automatically classifies IT helpdesk tickets into categories and assigns priority levels using text classification techniques.

Built as part of **Machine Learning Task 2 (2026) — Future Interns**.

---

## Project Overview

IT helpdesks get a lot of tickets every day. Someone has to read each one, figure out which team should handle it, and decide how urgent it is. That's a lot of manual work that slows things down.

In this project I tried to automate two things:
1. **Category prediction** — which team should handle this ticket? (e.g. Active Directory team, O365 team, etc.)
2. **Priority prediction** — how urgent is it? (High / Medium / Low)

The final output is a `predict_ticket()` function that takes any raw ticket text and returns the predicted category, priority, and which team to route it to — all in a few milliseconds.

---

## Problem Statement

Support teams waste a lot of time just *sorting* tickets before they can start *solving* them. When tickets pile up, urgent ones can get buried under low-priority requests. This project explores whether ML can handle the sorting automatically so agents can focus on the actual work.

---

## Dataset

**Source:** [Classification of IT Support Tickets — Zenodo](https://zenodo.org/records/7648117)

Real IT helpdesk ticket data, already split into train and test sets across 4 CSV files:

| File | What's in it |
|------|-------------|
| `X_train.csv` | Ticket ID + raw text (1,572 tickets) |
| `X_test.csv` | Ticket ID + raw text (657 tickets) |
| `y_train.csv` | Ticket ID + category label |
| `y_test.csv` | Ticket ID + category label |

**7 categories:** Active Directory, Computer-Services, EOL, Fileservice, O365, Software, Support general

> **One thing to note:** the dataset doesn't have a priority column. I created priority labels myself using a rule-based approach — more on this below.

---

## Project Workflow

### 1. Data Loading

The X and y files are separate but share an `id` column, so I merged them to get text and labels in one dataframe. After loading I checked for missing values (there were none in the key columns) and looked at the category distribution.

The data is quite unbalanced — `Fileservice` and `Support general` together make up about 70% of the training data, while `EOL` has only 45 tickets. This became a recurring challenge throughout the project.

### 2. Exploratory Data Analysis

I plotted the category distribution for both train and test sets to understand the imbalance. I also looked at a few raw ticket examples to understand what kind of text I was dealing with — which revealed some useful things like placeholder tokens (`[NAME]`, `[SERVER]`) and some non-English tickets.

### 3. Creating Priority Labels

Since there's no priority column, I built my own labels using helpdesk logic:

| Priority | Category | Reason |
|----------|----------|--------|
| High | EOL, Active Directory | Blocks employees or is a security risk |
| Medium | O365, Software, Fileservice | Important but usually has workarounds |
| Low | Computer-Services, Support general | Not urgent, can wait |

On top of the category rules, I also added a keyword check — if the ticket text contains words like `"urgent"`, `"down"`, `"blocked"`, or `"access denied"`, I upgrade the priority. I never downgrade, only upgrade.

These are approximate labels, not ground truth. The priority numbers look reasonable but a proper evaluation would need labels from actual helpdesk staff.

### 4. Text Cleaning

Raw ticket text has a lot of noise that doesn't help the model. My cleaning pipeline:

1. Lowercase everything
2. Remove `[PLACEHOLDER]` tokens like `[NAME]`, `[SERVER]` — same in every ticket, useless
3. Remove URLs, emails, numbers, and punctuation
4. Tokenize (split into individual words)
5. Remove stopwords + domain-specific noise words like `"ticket"`, `"support"`, `"please"`
6. Lemmatize — so `"installing"` and `"installed"` both become `"install"`

**Before cleaning:**
```
A Support Ticket was forwarded to your role. New request from [NAME] — [SERVER] is showing errors and nobody can log in.
```

**After cleaning:**
```
request server showing error nobody log
```

### 5. Feature Engineering — TF-IDF

ML models need numbers, not words. TF-IDF converts each ticket into a vector of numbers.

**How it works (simply):**
- A word gets a **high score** if it appears a lot in *this* ticket but rarely across all other tickets
- A word gets a **low score** if it appears in almost every ticket (like `"the"` or `"please"`) — these words don't help tell tickets apart

For example, the word `"nexthink"` appears frequently in EOL tickets but almost nowhere else, so it gets a high TF-IDF score for that category. This is exactly the kind of signal the model learns from.

**Settings I used:**
- `max_features=5000` — only the top 5,000 most useful words
- `ngram_range=(1, 2)` — considers single words AND word pairs like `"active directory"` or `"file share"`
- `sublinear_tf=True` — log scaling so very frequent words don't dominate
- `min_df=2` — ignore words appearing in fewer than 2 tickets
- `max_df=0.90` — ignore words appearing in more than 90% of tickets

**Important:** I only called `fit_transform()` on the training data. For test data I used `transform()` only. Fitting on test data too would be a data leakage mistake (I actually made this mistake the first time — more in the challenges section).

### 6. Model Training

I trained three different classifiers and compared them:

| Model | Why I tried it |
|-------|---------------|
| **Logistic Regression** | Simple, interpretable, solid baseline for text classification |
| **Multinomial Naive Bayes** | Built for word frequency data, extremely fast |
| **Linear SVM** | Usually strong on high-dimensional sparse data like TF-IDF |

For Logistic Regression and Linear SVM I used `class_weight='balanced'` to compensate for the class imbalance — this makes the model pay more attention to small categories like EOL instead of just optimising for the big ones.

I also ran 5-fold cross-validation alongside the test set evaluation. CV gives a more reliable accuracy estimate because it tests the model on 5 different chunks of training data instead of relying on one split.

### 7. Model Evaluation

**Metrics I used:**

- **Accuracy** — overall percentage of correct predictions
- **Precision** — of all tickets predicted as class X, how many were actually X? (avoids false alarms)
- **Recall** — of all tickets that really are class X, how many did the model catch? (avoids missing things)
- **F1-Score** — combines precision and recall into one number. More useful than accuracy alone when classes are imbalanced

I looked at confusion matrices to understand *where* the model was making mistakes, not just how many mistakes it made overall.

---

## Visualizations Used

- **Bar charts** — category and priority distribution for train and test sets
- **Model comparison chart** — test accuracy vs 5-fold CV mean for all three models
- **Confusion matrix (category)** — 7×7 heatmap showing which categories get confused with each other
- **Per-class precision / recall / F1 bar chart** — shows how each category performs individually
- **Confusion matrix (priority)** — 3×3 heatmap for High / Medium / Low predictions

---

## Models Compared

### Category Classification

| Model | Test Accuracy | CV Mean |
|-------|-------------|---------|
| Logistic Regression | ~77.5% | ~78.6% |
| Multinomial Naive Bayes | ~75.2% | ~81.4% |
| **Linear SVM** | **~81.9%** | **~82.8%** |

### Priority Classification

| Model | Test Accuracy | CV Mean |
|-------|-------------|---------|
| Logistic Regression | ~81.0% | ~84.7% |
| Multinomial Naive Bayes | ~77.9% | ~85.4% |
| **Linear SVM** | **~83.4%** | **~85.4%** |

**Linear SVM won both tasks.** For comparison, randomly guessing a category would give 14.3% accuracy (1 in 7), so 82% is a meaningful result.

---

## Results and Findings

**From the confusion matrix:**
- **EOL** is classified almost perfectly — it has very specific vocabulary like `"nexthink"` that appears nowhere else
- **Software** and **Support general** get mixed up a lot — both can have very generic IT language with no clear distinguishing words
- **Active Directory** and **Computer-Services** have lower recall because they have fewer training examples (84 and 65 tickets respectively), so the model hasn't seen enough examples to learn them well

**Priority numbers look reasonable** — 83.4% accuracy — but since the labels were created by rules rather than actual helpdesk staff, this number should be interpreted carefully.

---

## Why Linear SVM Performed Best

TF-IDF creates thousands of features (columns), but for any single ticket most of those features are zero — this is called sparse data. SVM is specifically designed to find the best boundary between classes in this kind of high-dimensional sparse space. It focuses on the most decisive words (the "support vectors") and ignores the rest, which makes it very effective here.

Naive Bayes assumes all words are independent of each other, which isn't true in real text — words often appear together and carry related meaning. This is why it can show different CV and test scores, and why it falls short of SVM in this task.

---

## Challenges Faced

**1. No priority column in the dataset**
The biggest issue was that the dataset only had categories, not priority. I had to come up with a way to create priority labels myself. I wasn't sure what logic to use at first — I thought about it from a helpdesk perspective (like which issues actually block someone from working vs which ones can wait) and came up with the rules in Step 2. A proper solution would be to get real labels from actual helpdesk staff.

**2. Class imbalance**
Fileservice and Support general are huge, while EOL has only 45 tickets. I used `class_weight='balanced'` which helped, but EOL and Computer-Services still have lower scores because there just isn't enough data for the model to learn them well.

**3. Some tickets are in Portuguese**
A few tickets in Active Directory and Computer-Services are not in English. My cleaning pipeline uses English stopwords and tokenizer so those tickets don't get cleaned as well. That's probably one reason those categories have lower F1.

**4. I made a data leakage mistake**
When I first wrote the TF-IDF part, I did `fit_transform` on both train and test. That's wrong because the vectorizer was learning from the test data too. My mentor caught this and I fixed it — now I only do `fit_transform` on training and `transform` on test.

**5. Accuracy isn't everything**
I initially just compared accuracy and thought Naive Bayes was worse. But then I noticed its CV score was higher than its test accuracy, which confused me. Looking at confusion matrices and per-class F1 helped me understand what was actually going on — overall accuracy hides a lot when classes are imbalanced.

---

## What I Learned From This Project

- Text data needs a lot of cleaning before it's useful — raw ticket text has way more noise than I expected
- You can't always rely on a dataset to have everything you need. Sometimes you have to create labels yourself and document your reasoning
- Accuracy alone is a bad metric when classes are different sizes. F1-score and confusion matrices tell a much more complete story
- Data leakage is a real and easy mistake to make. Fitting the vectorizer on test data gives misleadingly good results that won't hold up in the real world
- Cross-validation is more trustworthy than a single train/test split, especially when the dataset is small

---

## Future Improvements

- Get actual priority labels from helpdesk staff instead of using my rules — that would make the priority model more reliable
- Handle the Portuguese tickets better, maybe with language detection and multilingual stopwords
- Try oversampling (SMOTE) for the tiny EOL category to give the model more examples to learn from
- Build a simple Flask API so `predict_ticket()` is accessible from any application
- Try a pre-trained model like DistilBERT, which understands context and word relationships better than TF-IDF

---

## How to Run the Project

**1. Clone the repository**
```bash
git clone https://github.com/your-username/support-ticket-classifier.git
cd support-ticket-classifier
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Download the dataset**

Download the 4 CSV files from [Zenodo](https://zenodo.org/records/7648117) and place them in the project root:
- `X_train.csv`
- `X_test.csv`
- `y_train.csv`
- `y_test.csv`

**4. Open and run the notebook**
```bash
jupyter notebook support_ticket_classifier.ipynb
```

Run the cells in order from top to bottom. Each step is clearly labelled.

**5. Try the prediction function**

At the end of the notebook there's a `predict_ticket()` function you can test with any text:

```python
result = predict_ticket("The file server is completely down, nobody can access shared drives")
print(result)
# {'category': 'Fileservice', 'priority': 'Medium', 'route_to': 'Storage & File-Share Team'}
```

---

## Project Structure

```
support-ticket-classifier/
│
├── support_ticket_classifier.ipynb   ← main notebook (run this)
│
├── X_train.csv                       ← training ticket text
├── X_test.csv                        ← test ticket text
├── y_train.csv                       ← training labels
├── y_test.csv                        ← test labels
│
├── models/                           ← saved model files (auto-generated)
│   ├── tfidf_vectorizer.pkl
│   ├── best_category_model.pkl
│   └── best_priority_model.pkl
│
├── charts/                           ← saved charts (auto-generated)
│   ├── 04_confusion_matrix.png
│   └── 06_priority_confusion_matrix.png
│
├── requirements.txt
└── README.md
```

---

## Requirements

```
scikit-learn
nltk
pandas
numpy
matplotlib
seaborn
joblib
```

Install with:
```bash
pip install -r requirements.txt
```

After first run, NLTK will also download small language files (stopwords, tokenizer, lemmatizer). This happens automatically when you run the notebook.

---

*This project was completed as part of the Future Interns ML internship programme (2026).*
