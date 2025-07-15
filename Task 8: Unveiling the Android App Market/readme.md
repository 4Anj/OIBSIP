# Google Play Store App Analysis & User Review Sentiment Insights
This project performs comprehensive data cleaning, visualization, and sentiment analysis on datasets from the Google Play Store, including apps metadata and user reviews. It explores patterns in app categories, pricing, installs, ratings, and user sentiment using both static and interactive visualizations.

### Dataset Description
1. apps.csv
App-level metadata from Google Play Store
Columns include: App, Category, Rating, Reviews, Size, Installs, Price, Type, etc.

2. user_reviews.csv
User-submitted translated reviews
Columns include: App, Translated_Review, Sentiment

### Tools & Libraries Used
pandas, numpy – Data processing
matplotlib, seaborn – Data visualization
plotly.express – Interactive plotting
TextBlob – Sentiment polarity analysis
warnings – To suppress unnecessary logs

## EDA Highlights
### Data Cleaning Steps
Handled missing and duplicate values
Cleaned and converted:
Size from strings like "20M"/"300k" to float (MB)
Installs by removing "+" and ","
Price by removing "$"
Converted data types and dropped irrelevant columns

## Key Visualizations
### App-Level Insights
Category distribution: Most popular app categories

Rating distribution: Skewed towards high ratings

Average rating by category: Some niche categories receive better average feedback

App size vs rating: No strong correlation found visually

Free vs Paid Apps: Free apps dominate (majority share)

Top Installed Apps: Lists the 10 most downloaded apps

Top Revenue Paid Apps: Estimated using Price × Installs

### Review-Level Insights
Performed sentiment analysis using TextBlob's polarity

Classified sentiments as:
Positive (> 0)
Negative (< 0)
Neutral (= 0)
Visualized overall sentiment distribution

### Interactive Visualization
Created an interactive Plotly scatter plot:
X-axis: App size (MB)
Y-axis: Rating
Color: App category
Hover: App names

```python
fig = px.scatter(df, x='Size', y='Rating', color='Category', hover_data=['App'])
fig.show()
```

### Insights & Recommendations
Free Apps Dominate the Play Store (~90% of listings). Monetization strategies should consider in-app ads or purchases.

Top Revenue Apps tend to be niche high-value apps with fewer installs but higher prices.

User Sentiment is mostly positive, which supports the high average ratings.

Category and Size have varying effects on ratings; no strong linear relation.

Most Installed Apps are often free, simple, and serve utility or entertainment purposes.

### Requirements
```nginx

pandas
numpy
matplotlib
seaborn
plotly
textblob
```

Install with:
```bash
pip install pandas numpy matplotlib seaborn plotly textblob
```
