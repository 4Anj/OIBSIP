# Autocorrect & Autocomplete System using NLP (Norvig, Bigrams, TextBlob)
This project implements and evaluates:
A custom autocorrect system using Peter Norvig’s algorithm and compares it with TextBlob's correction

An autocomplete system using a Bigram Language Model

Full evaluation and visualization of accuracy, word prediction, and text suggestions based on real datasets

### Database Used
| File Name                             | Description                                                       |
| ------------------------------------- | ----------------------------------------------------------------- |
| `test.csv`                            | Contains `augmented_text` (misspelled) and `text` (correct)       |
| `search_data_how to get a job at.csv` | Contains user `search_term` and actual `auto_complete_suggestion` |


### Project Highlights
Autocorrect (Norvig Algorithm)

Uses edit distance and a known vocabulary (nltk.corpus.words)

Applies word-by-word correction on noisy/misspelled sentences

Calculates word-level accuracy

Output:
```vbnet
Autocorrect Word-Level Accuracy (Top 100 samples): 57.38%
Sample:
augmented_text:  project looks to muelsnig ngeetic alternative
corrected_text:  project looks to muelsnig genetic alternative
true_text:       project looks to muelsnig genetic alternative
word_accuracy:   66.7%
```

### Autocomplete (Bigram Model)
Tokenizes and builds bigrams from search query history

Predicts the next word given a prefix

Checks if predicted word is part of actual autocomplete suggestion

# Output:

```vbnet
Autocomplete Word Match Accuracy: 89.16%
Sample:
search_term:              How to get a job at
predicted_word:           how
auto_complete_suggestion: how to get a job at google
match:                    True
```

### Evaluation & Visualizations
Accuracy comparison (Norvig vs TextBlob)

WordCloud of autocomplete tokens

Bar chart of model accuracies

Sample predictions from both tasks

### Techniques Used
Text Cleaning & Tokenization

Peter Norvig's Autocorrect Algorithm

TextBlob Correction

Bigram Language Modeling for Autocomplete

Accuracy Metrics

Data Visualization with Matplotlib, WordCloud, Seaborn

### Installation & Setup
Requirements
```bash
pip install pandas nltk matplotlib wordcloud plotly textblob
```

NLTK Corpora
```python
import nltk
nltk.download('punkt')
nltk.download('words')
```

### How to Run
Place the dataset files: 
test.csv
Search_data_how to get a job at.csv

Run the script:
```bash
python autocorrect_autocomplete.py
```

### Results Summary
| Task         | Accuracy                |
| ------------ | ----------------------- |
| Autocorrect  | 57.38% (word-level)     |
| Autocomplete | 89.16% (match accuracy) |

### Future Ideas
Improve autocorrect by training with a domain-specific corpus

Use trigrams or LSTM/transformers for context-aware autocomplete

Build a real-time REST API or chatbot plugin for auto suggestions
