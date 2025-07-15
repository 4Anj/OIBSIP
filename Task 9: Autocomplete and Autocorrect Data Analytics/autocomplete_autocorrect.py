import pandas as pd
import re
import nltk
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
nltk.download('punkt')
nltk.download('words')

autocorrect_df = pd.read_csv(r"C:\Users\anjan\Downloads\test.csv").head(100)
autocomplete_df = pd.read_csv(r"C:\Users\anjan\Downloads\search_data_how to get a job at.csv")

def clean_text(text):
    return re.sub(r'[^a-zA-Z\s]', '', str(text)).lower().strip()

WORDS = Counter(nltk.corpus.words.words())

def P(word, N=sum(WORDS.values())): 
    return WORDS[word] / N

def correction(word): 
    return max(candidates(word), key=P)

def candidates(word): 
    return (known([word]) or known(edits1(word)) or known(edits2(word)) or [word])

def known(words): 
    return set(w for w in words if w in WORDS)

def edits1(word):
    letters    = 'abcdefghijklmnopqrstuvwxyz'
    splits     = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes    = [L + R[1:]           for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    replaces   = [L + c + R[1:]       for L, R in splits if R for c in letters]
    inserts    = [L + c + R           for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)

def edits2(word): 
    return (e2 for e1 in edits1(word) for e2 in edits1(e1))

def autocorrect_sentence(sentence):
    tokens = sentence.lower().split()
    corrected = [correction(word) for word in tokens]
    return ' '.join(corrected)

autocorrect_df['corrected_text'] = autocorrect_df['augmented_text'].apply(autocorrect_sentence)

def word_accuracy(row):
    true_words = row['text'].lower().split()
    predicted_words = row['corrected_text'].lower().split()
    correct = sum(1 for a, b in zip(true_words, predicted_words) if a == b)
    return correct / max(len(true_words), 1)

autocorrect_df['word_accuracy'] = autocorrect_df.apply(word_accuracy, axis=1)
autocorrect_score = autocorrect_df['word_accuracy'].mean()

autocomplete_df.dropna(subset=['search_term', 'auto_complete_suggestion'], inplace=True)

corpus = autocomplete_df['search_term'].astype(str).tolist()
tokens = []
for text in corpus:
    tokens.extend(nltk.word_tokenize(clean_text(text)))

def build_ngram_model(tokens, n=2):
    model = defaultdict(Counter)
    for i in range(len(tokens) - n):
        prefix = tuple(tokens[i:i+n-1])
        next_word = tokens[i+n-1]
        model[prefix][next_word] += 1
    return model

bigram_model = build_ngram_model(tokens, n=2)

def predict_next_word(prefix, model):
    words = nltk.word_tokenize(clean_text(prefix))
    if not words:
        return ""
    last_word = tuple(words[-1:])
    if last_word in model:
        return model[last_word].most_common(1)[0][0]
    return ""

autocomplete_df['predicted_word'] = autocomplete_df['search_term'].apply(lambda x: predict_next_word(x, bigram_model))


def match_found(row):
    return row['predicted_word'] in str(row['auto_complete_suggestion']).lower().split()

autocomplete_df['match'] = autocomplete_df.apply(match_found, axis=1)
autocomplete_score = autocomplete_df['match'].mean()

word_freq = Counter(tokens)
wc = WordCloud(width=800, height=400, background_color="white").generate_from_frequencies(word_freq)

plt.figure(figsize=(10, 5))
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
plt.title("Word Frequency in Autocomplete Data")
plt.show()

print("\n📊 Accuracy Scores:")
print(f"Autocorrect Word-Level Accuracy (Top 100 samples): {autocorrect_score:.2%}")
print(f"Autocomplete Word Match Accuracy: {autocomplete_score:.2%}")

print("\n🔍 Sample Autocorrect Results:")
print(autocorrect_df[['augmented_text', 'corrected_text', 'text', 'word_accuracy']].head(5))

print("\n🔍 Sample Autocomplete Results:")
print(autocomplete_df[['search_term', 'predicted_word', 'auto_complete_suggestion', 'match']].head(5))

from textblob import TextBlob
import time

def textblob_autocorrect(sentence):
    return str(TextBlob(sentence).correct())

start_time = time.time()
autocorrect_df['textblob_corrected'] = autocorrect_df['augmented_text'].apply(textblob_autocorrect)
textblob_time = time.time() - start_time

def word_accuracy_textblob(row):
    target = row['text'].lower().split()
    corrected = row['textblob_corrected'].lower().split()
    correct = sum(1 for a, b in zip(target, corrected) if a == b)
    return correct / max(len(target), 1)

autocorrect_df['textblob_accuracy'] = autocorrect_df.apply(word_accuracy_textblob, axis=1)
textblob_score = autocorrect_df['textblob_accuracy'].mean()

import matplotlib.pyplot as plt

methods = ['Norvig', 'TextBlob']
scores = [autocorrect_score, textblob_score]

plt.figure(figsize=(6, 4))
plt.bar(methods, scores, color=['skyblue', 'lightgreen'])
plt.title('Autocorrect Accuracy Comparison')
plt.ylabel('Word-Level Accuracy')
plt.ylim(0, 1)
plt.show()
