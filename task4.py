import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
sns.set(style="whitegrid")
HAVE_TEXTBLOB = False
HAVE_WORDCLOUD = False
HAVE_SKLEARN = False
try:
    from textblob import TextBlob
    HAVE_TEXTBLOB = True
except Exception:
    pass
try:
    from wordcloud import WordCloud
    HAVE_WORDCLOUD = True
except Exception:
    pass
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, accuracy_score
    import joblib
    HAVE_SKLEARN = True
except Exception:
    pass

INPUT_FILE = "twitter_training.csv"
OUTPUT_FILE = "twitter_sentiment_output.csv"
CHUNKSIZE = 20000
POS_TEXT_CAP = 200000
NEG_TEXT_CAP = 200000

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(INPUT_FILE + " not found")

sample = pd.read_csv(INPUT_FILE, nrows=200)
text_col = None
cands = [c for c in sample.columns if c.lower() in ('text','tweet','content','message','tweet_text')]
if cands:
    text_col = cands[0]
else:
    for c in sample.columns:
        if sample[c].dtype == object:
            ss = sample[c].dropna().astype(str).head(20).tolist()
            if any(len(s.split())>3 for s in ss):
                text_col = c
                break
if text_col is None:
    text_col = sample.columns[0]

analyzer = SentimentIntensityAnalyzer()
reader = pd.read_csv(INPUT_FILE, chunksize=CHUNKSIZE)
first = True
from collections import Counter
sent_counts = Counter()
pol_vals = []
compound_vals = []
pos_acc = []
neg_acc = []
all_acc = []
rows = 0

def vec_clean(s):
    return (s.astype(str)
            .str.lower()
            .str.replace(r'http\S+|www\S+|https\S+', '', regex=True)
            .str.replace(r'@\w+', '', regex=True)
            .str.replace(r'#', ' ', regex=True)
            .str.replace(r'[^a-z\s]', ' ', regex=True)
            .str.replace(r'\s+', ' ', regex=True)
            .str.strip())

for i,chunk in enumerate(reader):
    chunk = chunk.copy()
    chunk['text_raw'] = chunk[text_col].astype(str)
    chunk['text'] = vec_clean(chunk['text_raw'])
    chunk['vader_compound'] = chunk['text'].apply(lambda x: analyzer.polarity_scores(x)['compound'])
    if HAVE_TEXTBLOB:
        chunk['textblob_polarity'] = chunk['text'].apply(lambda x: TextBlob(x).sentiment.polarity)
        chunk['textblob_subjectivity'] = chunk['text'].apply(lambda x: TextBlob(x).sentiment.subjectivity)
    else:
        chunk['textblob_polarity'] = np.nan
        chunk['textblob_subjectivity'] = np.nan
    chunk['sentiment'] = chunk['vader_compound'].apply(lambda v: 'Positive' if v>0.05 else ('Negative' if v<-0.05 else 'Neutral'))
    sent_counts.update(chunk['sentiment'].value_counts().to_dict())
    if HAVE_TEXTBLOB:
        pol_vals.extend(chunk['textblob_polarity'].dropna().tolist())
    else:
        compound_vals.extend(chunk['vader_compound'].tolist())
    pos_list = chunk.loc[chunk['sentiment']=='Positive','text'].tolist()
    neg_list = chunk.loc[chunk['sentiment']=='Negative','text'].tolist()
    if pos_list and HAVE_WORDCLOUD:
        pos_acc.append(" ".join(pos_list))
    if neg_list and HAVE_WORDCLOUD:
        neg_acc.append(" ".join(neg_list))
    all_acc.append(" ".join(chunk['text'].tolist()))
    if first:
        chunk.to_csv(OUTPUT_FILE, index=False, mode='w', encoding='utf-8')
        first = False
    else:
        chunk.to_csv(OUTPUT_FILE, index=False, mode='a', header=False, encoding='utf-8')
    rows += len(chunk)
    print(f"Processed chunk {i+1}, total rows {rows}")

print("Done chunked processing")
print("Sentiment counts:", dict(sent_counts))

vals = np.array(pol_vals) if len(pol_vals)>0 else np.array(compound_vals)

plt.figure(figsize=(6,4))
labels = ['Positive','Neutral','Negative']
counts = [sent_counts.get('Positive',0), sent_counts.get('Neutral',0), sent_counts.get('Negative',0)]
import seaborn as sns
sns.barplot(x=labels, y=counts, order=labels)
plt.title("Sentiment Distribution (VADER)")
plt.ylabel("Count")
plt.savefig("sentiment_distribution.png", dpi=200, bbox_inches='tight')
plt.show()

plt.figure(figsize=(7,4))
import matplotlib.pyplot as plt
plt.hist(vals, bins=40, density=False)
plt.title("Polarity / Compound Distribution")
plt.xlabel("Value")
plt.savefig("polarity_distribution.png", dpi=200, bbox_inches='tight')
plt.show()

pos_text = (" ".join(pos_acc))[:POS_TEXT_CAP] if pos_acc else ""
neg_text = (" ".join(neg_acc))[:NEG_TEXT_CAP] if neg_acc else ""
all_text = (" ".join(all_acc))[:400000] if all_acc else ""

if HAVE_WORDCLOUD and pos_text and len(pos_text.split())>10:
    wc = WordCloud(width=800, height=400, background_color='white').generate(pos_text)
    plt.figure(figsize=(10,5)); plt.imshow(wc); plt.axis("off"); plt.title("Positive WordCloud")
    plt.savefig("wordcloud_positive.png", dpi=200, bbox_inches='tight'); plt.show()
else:
    print("Positive wordcloud skipped")

if HAVE_WORDCLOUD and neg_text and len(neg_text.split())>10:
    wc = WordCloud(width=800, height=400, background_color='black', colormap='Reds').generate(neg_text)
    plt.figure(figsize=(10,5)); plt.imshow(wc); plt.axis("off"); plt.title("Negative WordCloud")
    plt.savefig("wordcloud_negative.png", dpi=200, bbox_inches='tight'); plt.show()
else:
    print("Negative wordcloud skipped")

if HAVE_WORDCLOUD and all_text and len(all_text.split())>20:
    wc = WordCloud(width=800, height=400, background_color='white').generate(all_text)
    plt.figure(figsize=(10,5)); plt.imshow(wc); plt.axis("off"); plt.title("All Tweets WordCloud")
    plt.savefig("wordcloud_all.png", dpi=200, bbox_inches='tight'); plt.show()
else:
    print("All tweets wordcloud skipped")

if HAVE_SKLEARN:
    try:
        sample_n = 50000
        df_sample2 = pd.read_csv(INPUT_FILE, nrows=sample_n)
        df_sample2['text'] = vec_clean(df_sample2[text_col].astype(str))
        if 'label' in df_sample2.columns:
            ycol = 'label'
        elif 'sentiment_label' in df_sample2.columns:
            ycol = 'sentiment_label'
        elif 'target' in df_sample2.columns:
            ycol = 'target'
        else:
            analyzer2 = SentimentIntensityAnalyzer()
            df_sample2['vader_compound'] = df_sample2['text'].apply(lambda x: analyzer2.polarity_scores(x)['compound'])
            df_sample2['label'] = df_sample2['vader_compound'].apply(lambda v: 'Positive' if v>0.05 else ('Negative' if v<-0.05 else 'Neutral'))
            ycol = 'label'
        Xtexts = df_sample2['text'].fillna('')
        tf = TfidfVectorizer(stop_words='english', max_features=20000, ngram_range=(1,2))
        Xmat = tf.fit_transform(Xtexts)
        yvec = df_sample2[ycol].astype(str)
        Xtr, Xte, ytr, yte = train_test_split(Xmat, yvec, test_size=0.2, random_state=42, stratify=yvec)
        clf = LogisticRegression(max_iter=1000, class_weight='balanced', solver='liblinear', random_state=42)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(clf, Xtr, ytr, cv=cv, scoring='accuracy', n_jobs=-1)
        clf.fit(Xtr, ytr)
        ypred = clf.predict(Xte)
        yproba = clf.predict_proba(Xte) if hasattr(clf,'predict_proba') else None
        acc = accuracy_score(yte, ypred)
        print("CV acc mean {:.4f} std {:.4f}".format(cv_scores.mean(), cv_scores.std()))
        print("Test acc", acc)
        print(classification_report(yte, ypred))
        cm = confusion_matrix(yte, ypred, labels=clf.classes_)
        plt.figure(figsize=(6,5)); sns.heatmap(cm, annot=True, fmt='d', xticklabels=clf.classes_, yticklabels=clf.classes_, cmap='Blues'); plt.title('Confusion Matrix'); plt.savefig("confusion_matrix_model.png", dpi=200, bbox_inches='tight'); plt.show()
        if yproba is not None and len(clf.classes_)==2:
            pos_index = list(clf.classes_).index('Positive') if 'Positive' in clf.classes_ else 1
            yprobpos = yproba[:, pos_index]
            fpr,tpr,_ = roc_curve([1 if v=='Positive' else 0 for v in yte], yprobpos)
            roc_auc = roc_auc_score([1 if v=='Positive' else 0 for v in yte], yprobpos)
            plt.figure(figsize=(6,5)); plt.plot(fpr,tpr,label=f'ROC (AUC={roc_auc:.4f})'); plt.plot([0,1],[0,1],'--'); plt.legend(); plt.title('ROC'); plt.savefig("roc_model.png", dpi=200, bbox_inches='tight'); plt.show()
        joblib.dump(tf, "tfidf_vectorizer.joblib")
        joblib.dump(clf, "logreg_sentiment_model.joblib")
        print("Modeling outputs saved")
    except Exception as e:
        print("Modeling skipped due to error:", e)
else:
    print("Modeling skipped (scikit-learn not installed)")

print("Outputs:")
print("-", OUTPUT_FILE)
print("- sentiment_distribution.png")
print("- polarity_distribution.png")
if HAVE_WORDCLOUD:
    print("- wordcloud_positive.png, wordcloud_negative.png, wordcloud_all.png (if generated)")
if HAVE_SKLEARN:
    print("- tfidf_vectorizer.joblib, logreg_sentiment_model.joblib, confusion_matrix_model.png, roc_model.png (if generated)")
