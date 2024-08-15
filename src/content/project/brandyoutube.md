---
title: "Enhance your brand using YouTube"
description: "The in-depth analysis of YouTube trends"
pubDate: "Jun 3 2024"
heroImage: "https://i.imgur.com/zSsGSRG.png"
badge: "Data Analysis"
tags: ["Sentiment Analysis", "Trend Analysis"]
---

# Enhance your brand using YouTube

## 📖 Background

You're a data scientist at a global marketing agency that helps some of the world's largest companies enhance their online presence.

Your new project is exciting: identify the most effective YouTube videos to promote your clients’ brands.

Forget simple metrics like views or likes; your job is to dive deep and discover who really connects with audiences through innovative content analysis.

## 💾 The Data

The data for this competition is stored in two tables, `videos_stats` and `comments`.

### `videos_stats.csv`

This table contains aggregated data for each YouTube video:

- **Video ID**: A unique identifier for each video.
- **Title**: The title of the video.
- **Published At**: The publication date of the video.
- **Keyword**: The main keyword or topic of the video.
- **Likes**: The number of likes the video has received.
- **Comments**: The number of comments on the video.
- **Views**: The total number of times the video has been viewed.

### `comments.csv`

This table captures details about comments made on YouTube videos:

- **Video ID**: The identifier for the video the comment was made on (matches the `Videos Stats` table).
- **Comment**: The text of the comment.
- **Likes**: How many likes this comment has received.
- **Sentiment**: The sentiment score ranges from 0 (negative) to 2 (positive), indicating the tone of a comment.

# Executive Summary

This report explores how data science techniques can be used to identify the most effective YouTube videos for promoting a client's brand. It analyzes two datasets, "videos_stats.csv" containing video metadata and "comments.csv" containing comments on those videos.

Data Cleaning and Exploration

- Both datasets were cleaned to handle missing values, outliers, and data types.
- Exploratory data analysis revealed insights into video statistics, engagement metrics, and keyword classifications.

Engagement Metrics

- A new metric, "Engagement Rate," was calculated by combining likes, comments, and views.
- The average engagement rate was compared by keyword and industry to identify high-performing areas.

Keyword-Based Analysis

- Keywords were categorized into industries to understand industry-specific trends.
- Word clouds were used to visualize the most frequent keywords within each industry.

Sentiment Analysis

- Sentiment analysis was performed on comments to categorize them as positive, neutral, or negative.
- Overall sentiment was positive, with further exploration revealing industry-specific sentiment variations.
- Emotion analysis was conducted to identify the prevalence of various emotions in the comments.
- Average emotion scores were calculated by keyword and industry, revealing interesting trends.
- Time-series analysis showed the evolution of average emotion scores over time, uncovering potential peaks and trends.

Key Findings

- Engagement rate varies by keyword and industry.
- Positive comments are most common overall, but sentiment varies by industry.
- Trust is the dominant emotion, followed by anticipation and joy.
- Specific emotions show peaks and trends over time, offering insights into audience reactions.

# Preparing Data

## Install Libraries

```python
!pip3 install nrclex seaborn wordcloud langdetect numpy pandas matplotlib tqdm scikit-learn nltk spacy plotly
!python3 -m spacy download en_core_web_sm
```

## Importing Libraries

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from nrclex import NRCLex
import nltk
import numpy as np
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import plotly.express as px
import plotly.figure_factory as ff
import warnings

nltk.download('punkt')
warnings.filterwarnings("ignore")
```

## Importing Data

```python
videos_stats = pd.read_csv('videos_stats.csv')
comments = pd.read_csv('comments.csv')
```

## Cleaning Data

### Cleaning Video Data

```python
vs_clean = videos_stats.copy()
for column in videos_stats.columns:
    if vs_clean[column].dtype == 'object':
        vs_clean[column].fillna(vs_clean[column].mode()[0], inplace=True)
    else:
        vs_clean[column].fillna(vs_clean[column].median(), inplace=True)
vs_clean['Published At'] = pd.to_datetime(vs_clean['Published At'], errors='coerce')
vs_clean.drop_duplicates(inplace=True)
q_low = vs_clean['Likes'].quantile(0.01)
q_hi  = vs_clean['Likes'].quantile(0.99)
vs_clean = vs_clean[(vs_clean['Likes'] > q_low) & (vs_clean['Likes'] < q_hi)]
vs_clean.to_csv('videos_stats_cleaned.csv', index=False)
vs_clean.head()
```

### Cleaning Comments Data

```python
cm_clean = comments.copy()
cm_clean.dropna(subset=['Comment'], inplace=True)
cm_clean['Likes'] = cm_clean['Likes'].fillna(0)
cm_clean['Sentiment'] = cm_clean['Sentiment'].fillna(cm_clean['Sentiment'].median())
cm_clean['Likes'] = cm_clean['Likes'].astype(int)
cm_clean['Sentiment'] = cm_clean['Sentiment'].astype(int)
cm_clean['Comment'] = cm_clean['Comment'].str.strip()
cm_clean = cm_clean[(cm_clean['Sentiment'] >= 0) & (cm_clean['Sentiment'] <= 2)]
cm_clean.to_csv('comments_cleaned.csv', index=False)
cm_clean.head()

```

# Exploratory Data Analysis of YouTube Trends

## Data Validation

### Validating data types

```python
videos_stats.info()
```

### Validating numerical data

```python
videos_stats.select_dtypes("number")
```

### Separate Year, Month, Day

```python
vs_df = pd.DataFrame(videos_stats)
vs_df['Year'] = vs_df['Published At'].apply(lambda x: x.split('/')[-1]).astype(int)
vs_df['Month'] = vs_df['Published At'].apply(lambda x: x.split('/')[1]).astype(int)
vs_df['Day'] = vs_df['Published At'].apply(lambda x: x.split('/')[0]).astype(int)
vs_df
```

```python
vs_df.select_dtypes("number")
```

## Merging Data

```python
merged_df = pd.merge(cm_clean, vs_clean, on='Video ID', how='right', indicator=True)
merged_df.drop_duplicates(subset=['Comment'], keep='first', inplace=True)
merged_df = merged_df.reset_index(drop=True)
merged_df
```

```python
from langdetect import detect

def is_english(text):
    try:
        return detect(text) == 'en'
    except:
        # Handle potential errors (e.g., empty text)
        return False
```

```python
merged_df = merged_df[merged_df['Comment'].apply(is_english)]
```

## Engagement Metrics

To calculate a YouTube channel's engagement rate, divide the total number of likes, comments, shares, and other engagements by the total number of views, then multiply by 100. This formula applies to both individual videos and entire channels.

"Total Number of Likes + Total Number of Comments" / Views \* 100

basic engagement metrics such as views, likes, and comments and identify which types of content are most popular in each industry.

```python
comments_count = merged_df.groupby('Video ID').size().reset_index(name='comments_count')
merged_df = pd.merge(merged_df, comments_count, on='Video ID', how='left')
```

```python
merged_df['Engagement Rate'] = ((merged_df['Likes_y'] + merged_df['comments_count']) / merged_df['Views']) * 100

keyword_engagement = merged_df.groupby('Keyword')['Engagement Rate'].mean()

keyword_engagement_sorted = keyword_engagement.sort_values()
```

```python
n_bars = len(keyword_engagement_sorted)
colors = ['blue' if i < 6 else 'red' if i >= n_bars - 6 else 'green' for i in range(n_bars)]

plt.figure(figsize=(10, 12))
plt.barh(keyword_engagement_sorted.index, keyword_engagement_sorted.values, color=colors, height=0.4)
plt.ylabel('Keyword')
plt.xlabel('Average Engagement Rate (%)')
plt.title('Average Engagement Rate by Keyword, Sorted')
plt.show()

```

## Word Cloud of Keywords

```python
keyword_frequencies = {keyword: value for keyword, value in zip(keyword_engagement.index, keyword_engagement.values)}
wordcloud = WordCloud(width=800, height=400, max_font_size=100, background_color='white').generate_from_frequencies(keyword_frequencies)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()
```

## Keyword-based Classification

```python
unique_keywords = merged_df['Keyword'].unique()
unique_keywords
```

- Technology
- Education
- Media & Entertainment
- Food & Beverage
- Gaming
- Sports
- Arts & Entertainment
- Pets & Animals

```python
industry_map = {
    "Technology": [
        "tech", "apple", "google", "computer science", "data science", "machine learning"
    ],
    "Business & Finance": [
        "business", "finance", "crypto", "interview", "news"
    ],
    "Gaming": [
        "gaming", "tutorial", "nintendo", "xbox", "minecraft", "game development"
    ],
    "Media & Entertainment": [
        "movies", "marvel", "mrbeast", "cnn", "mukbang", "reaction"
    ],
    "Sports": [
        "sports", "chess", "cubes"
    ],
    "Education": [
        "how-to", "history", "literature", "education", "math", "chemistry", "biology", "physics", "sat"
    ],
    "Lifestyle & Leisure": [
        "food", "bed", "animals", "trolling", "asmr", "music", "lofi"
    ]
}
```

```python
industry_map_data = [(keyword, industry) for industry, keywords in industry_map.items() for keyword in keywords]
industry_map_df = pd.DataFrame(industry_map_data, columns=['Keyword', 'Industry'])
industry_map_df.head()
```

```python
merged_df = pd.merge(merged_df, industry_map_df, on='Keyword', how='left')
```

## Engagement Rate by Industry

```python

industry_engagement_sorted = merged_df.groupby('Industry')['Engagement Rate'].mean().sort_values(ascending=True)


nn_bars = len(industry_engagement_sorted)

plt.figure(figsize=(10, 12))
plt.barh(industry_engagement_sorted.index, industry_engagement_sorted.values, color=colors, height=0.4)
plt.ylabel('Industry')
plt.xlabel('Average Engagement Rate (%)')
plt.title('Average Engagement Rate by Industry, Sorted')
plt.show()

```

```python

keyword_popularity = merged_df.groupby(['Industry', 'Keyword']).agg({
    'Likes_x': 'sum',           # Total likes
    'Comments': 'sum',          # Total comments
    'Views': 'sum',             # Total views
    'Engagement Rate': 'mean'   # Mean engagement rate
})


keyword_popularity = keyword_popularity.sort_values(by='Likes_x', ascending=False)
keyword_popularity.head()

```

```python
keyword_popularity_df = pd.DataFrame(keyword_popularity, columns=[])
keyword_popularity_df.head()
```

```python
industry_engagement= keyword_popularity.groupby('Industry')['Engagement Rate'].mean().sort_values(ascending=False)

plt.figure(figsize=(12, 10))
plt.title('Keyword Popularity by Industry')
plt.xlabel('Engagement Rate')
plt.ylabel('Keyword')


for industry in industry_engagement.index:
    data = keyword_popularity.loc[industry].sort_values(by='Engagement Rate', ascending=False)
    plt.barh(data.index.get_level_values('Keyword'), data['Engagement Rate'], label=industry)

plt.legend()
plt.show()

```

```python
industry_view = keyword_popularity.groupby('Industry')['Views'].mean().sort_values(ascending=False)


plt.figure(figsize=(12, 10))
plt.title('Keyword Views by Industry')
plt.xlabel('Views')
plt.ylabel('Keyword')

for industry in industry_view.index:
    data = keyword_popularity.loc[industry].sort_values(by='Views', ascending=False)
    plt.barh(data.index.get_level_values('Keyword'), data['Views'], label=industry)

plt.legend()

plt.show()

```

# Sentiment Analysis of Video Comments

Dataset provides the sentiment scale from 0 to 2. <br>

Since Sentiment is only identifying the negativity and positive, it is not as detail as to identify the emotion & opinions. <br>

So we will further investigate the sentiment of the comments after identifying the propotion of positive and negative comments. <br>

## Sentiment Distribution

```python
temp = merged_df.groupby('Sentiment').count()['Comments'].reset_index().sort_values(by='Comments',ascending=False)
sentiment_mapping = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
temp['Sentiment_Words'] = temp['Sentiment'].map(sentiment_mapping)
temp
```

```python
plt.figure(figsize=(12,6))
sns.countplot(x='Sentiment',data=merged_df)
```

This is one of the first element that goes into a sentiment analysis system which is **Opinion**

Opinion has three divisions: positive, neutral, and negative.

In this case

- Positive = 2
- neutral = 1
- nagative = 0

```python
sentiment_percentage = merged_df.Sentiment.value_counts()/len(merged_df)
sentiment_percentage
```

Overall Positive comments are most common among videos.

Positive comments are 62% <br />
Neutral comments are 25.1% <br />
Negative comments are 12.9% <br />

Let's find out which types

## Keyword by Average Sentiment

```python
keyword_sentiment = merged_df.groupby(['Industry', 'Keyword']).agg({
    'Sentiment': 'mean'
})

keyword_sentiment = keyword_sentiment.sort_values(by=['Industry', 'Sentiment'], ascending=[False, False])

keyword_sentiment.head()
```

```python
keyword_sentiment = merged_df.groupby(['Industry', 'Keyword']).agg({'Sentiment': 'mean'}).reset_index()
def categorize_sentiment(sentiment):
    if 0 <= sentiment < 0.666:
        return 'Negative'
    elif 0.666 <= sentiment < 1.332:
        return 'Neutral'
    elif 1.332 <= sentiment <= 2:
        return 'Positive'
keyword_sentiment['Sentiment Category'] = keyword_sentiment['Sentiment'].apply(categorize_sentiment)
keyword_sentiment = keyword_sentiment.sort_values(by='Sentiment', ascending=False)
sns.set(style="whitegrid")
plt.figure(figsize=(14, 8))
bar_plot = sns.barplot(data=keyword_sentiment, x='Sentiment', y='Keyword', hue='Sentiment Category', palette='coolwarm')
plt.title('Keyword by Average Sentiment', fontsize=18)
plt.xlabel('Average Sentiment', fontsize=14)
plt.ylabel('Keyword', fontsize=14)
plt.legend(title='Sentiment Category', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
```

## Keyword Sentiment by Industry

```python
industry_sentiment_sum = keyword_sentiment.groupby('Industry')['Sentiment'].sum().sort_values(ascending=False)

plt.figure(figsize=(12, 10))
plt.title('Keyword Sentiment by Industry')
plt.xlabel('Sentiment')
plt.ylabel('Keyword')

for industry in industry_sentiment_sum.index:

    data_sentiment = keyword_sentiment[keyword_sentiment['Industry'] == industry].sort_values(by='Sentiment', ascending=False)


    plt.barh(data_sentiment['Keyword'], data_sentiment['Sentiment'], label=industry)

plt.legend()
plt.show()
```

## Emotion Analysis

The categorical model of emotion analysis places a person's emotions into six basic categories, like anger, fear, disgust, joy, sadness, and surprise. Specific words are linked to relevant emotion tags and used to detect both related and unrelated emotions([Reference](https://www.delve.ai/blog/emotion-analysis#:~:text=The%20categorical%20model%20of%20emotion,both%20related%20and%20unrelated%20emotions.)).

https://pypi.org/project/NRCLex/

```python
def get_emotion_scores(comment):
    emotions = NRCLex(comment).affect_frequencies
    return emotions
comment_df = merged_df.copy()
comment_df['Emotion Scores'] = comment_df['Comment'].apply(get_emotion_scores)
emotion_columns = ['fear', 'anger', 'anticipation', 'trust', 'surprise', 'sadness', 'disgust', 'joy']
for emotion in emotion_columns:
    comment_df[emotion] = comment_df['Emotion Scores'].apply(lambda x: x.get(emotion.lower(), 0))
comment_df.drop(columns=['Emotion Scores'], inplace=True)
comment_df.to_csv('comment_df.csv', index=False)
comment_df.head()

```

## Emotion Distribution

```python
average_emotions = comment_df[emotion_columns].mean().sort_values()

plt.figure(figsize=(10, 6))
average_emotions.plot(kind='bar')
plt.title('Overall Average of Each Emotion (Sorted)')
plt.xlabel('Emotions')
plt.ylabel('Average Score')
plt.xticks(rotation=45)
plt.show()
```

## Average Emotion Scores by Keyword and Industry

```python
keyword_avg_df = comment_df.groupby('Keyword')[emotion_columns].mean()
keyword_avg_df['Total'] = keyword_avg_df.sum(axis=1)
keyword_avg_df = keyword_avg_df.sort_values(by='Total', ascending=False).drop(columns='Total').reset_index()
plt.figure(figsize=(12, 10))
sns.heatmap(keyword_avg_df.set_index('Keyword'), annot=True, cmap="coolwarm", cbar=True, linewidths=0.5, linecolor='gray')
plt.title("Average Emotion Scores by Keyword", fontsize=16)
plt.xlabel("Emotions", fontsize=14)
plt.ylabel("Keywords", fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()
```

```python
industry_avg_df = comment_df.groupby('Industry')[emotion_columns].mean()

industry_avg_df['Total'] = industry_avg_df.sum(axis=1)

industry_avg_df = industry_avg_df.sort_values(by='Total', ascending=False).drop(columns='Total').reset_index()

plt.figure(figsize=(12, 10))
sns.heatmap(industry_avg_df.set_index('Industry'), annot=True, cmap="coolwarm", cbar=True, linewidths=0.5, linecolor='gray')
plt.title("Average Emotion Scores by Industry", fontsize=16)
plt.xlabel("Emotions", fontsize=14)
plt.ylabel("Industry", fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()
```

## Average Emotion Scores Over Time

```python
resampled_data = comment_df.resample('Y', on='Published At')[emotion_columns].mean()

plt.figure(figsize=(14, 15))

for emotion in emotion_columns:
    plt.plot(resampled_data.index.year, resampled_data[emotion], marker='o', markersize=8, linestyle='-', linewidth=2, label=emotion.capitalize())

plt.xlabel('Year')
plt.ylabel('Average Emotion Score')
plt.title('Average Yearly Emotion Scores Over Time')
plt.legend()
plt.yticks(np.arange(resampled_data.min().min(), resampled_data.max().max() + 0.005, 0.005))
plt.grid(True)
plt.show()
```

Specific Trends

```python
trends = {}
for emotion in emotion_columns:
    emotion_data = resampled_data[emotion]
    peaks = emotion_data[(emotion_data.shift(1) < emotion_data) & (emotion_data.shift(-1) < emotion_data)]
    trends[emotion] = {
        'peaks': peaks,
        'max': emotion_data.max(),
        'min': emotion_data.min(),
        'trend': 'decreasing' if emotion_data.iloc[-1] < emotion_data.iloc[0] else 'increasing'
    }
for emotion, data in trends.items():
    print(f"{emotion.capitalize()}:")
    print(f"  Peaks: {data['peaks'].index.year.tolist()} at values {data['peaks'].values}")
    print(f"  Max: {data['max']}")
    print(f"  Min: {data['min']}")
    print(f"  Overall trend: {data['trend']}")

```

#### General observation

- Trust is the most dominant emotion throughout the years, consistently having the highest average scores compared to other emotions.
- Anticipation and Joy also show relatively high scores, with Anticipation peaking several times.
- Surprise, Sadness, Fear, Anger, and Disgust have lower average scores compared to Trust, Anticipation, and Joy.

#### Specific Trends:

- Trust:
  - Peaks around 2007 and 2013.
  - Shows a general decline from 2014 to 2022.
- Anticipation:
  - Highly variable with notable peaks in 2007, 2011, and 2014.
  - Slightly more stable but still fluctuating in the later years.
- Joy:
  - Peaks around 2007 and 2011, followed by smaller peaks in subsequent years.
  - Somewhat stable with moderate fluctuations.
- Sadness, Fear, Anger, and Disgust:
  - All four emotions show lower and more consistent scores over time.
  - Notable peaks for Fear and Sadness around 2010 and 2011.
  - Disgust and Anger have occasional minor peaks but remain relatively low.

#### Yearly Patterns

- Early Years (2006-2010):
  - High variability in emotions, with multiple peaks, especially for Trust, Anticipation, and Fear.
- Middle Years (2011-2015):
  - Trust and Anticipation show notable peaks.
  - Joy remains relatively high but stable.
  - Lower emotions maintain consistency.
- Recent Years (2016-2022):
  - A slight downward trend in Trust.
  - Anticipation and Joy show less variability.
  - Fear, Sadness, Anger, and Disgust remain consistently low.

# 3. Development of a Video Ranking Model

Recommender systems for YouTube from the paper: Deep Neural Networks for YouTube Recommendations

## Model

![img](https://i.imgur.com/OEEbMWz.png)

## Candidate Generation

![img](https://i.imgur.com/MKyASAG.png)

## Ranking

![img](https://i.imgur.com/H2VNR8X.png)

---

```python
from sklearn.model_selection import train_test_split
```

```python
rankingdata =pd.DataFrame(comment_df)
```

```python
features = ['Views', 'Likes_y', 'Comments', 'Sentiment']
target = 'Engagement Rate'

train_df, test_df = train_test_split(rankingdata, test_size=0.3, random_state=42)

X_train = train_df[features]
y_train = train_df[target]
X_test = test_df[features]
y_test = test_df[target]

print("Training Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)
```

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42)
}

evaluation_metrics = {
    "Mean Absolute Error": mean_absolute_error,
    "Mean Squared Error": mean_squared_error
}


for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print(f"Evaluating {name}:")
    for metric_name, metric_func in evaluation_metrics.items():
        metric_value = metric_func(y_test, predictions)
        print(f"{metric_name}: {metric_value:.4f}")
    print("="*30)
```

```python
predictions = model.predict(X_test)

mse = mean_squared_error(y_test, predictions)
print("Mean Squared Error:", mse)
```

---

```python
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
```

```python
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.3)
plt.xlabel('Actual Engagement Rate')
plt.ylabel('Predicted Engagement Rate')
plt.title('Actual vs Predicted Engagement Rate')
plt.show()
```

```python
importances = model.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance')
plt.gca().invert_yaxis()
plt.show()
```

```python
from sklearn.model_selection import GridSearchCV

# Define the parameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 4, 5]
}

# Initialize the model
grid_search = GridSearchCV(estimator=GradientBoostingRegressor(random_state=42), param_grid=param_grid, cv=3, n_jobs=-1, scoring='neg_mean_squared_error')

# Train the model
grid_search.fit(X_train, y_train)

# Best parameters and best score
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print("Best Parameters:", best_params)
print("Best Score:", best_score)

# Train the final model with best parameters
final_model = GradientBoostingRegressor(**best_params, random_state=42)
final_model.fit(X_train, y_train)

# Predict on the test set
final_pred = final_model.predict(X_test)

# Evaluate the final model
final_mse = mean_squared_error(y_test, final_pred)
print("Final Mean Squared Error:", final_mse)
```

---

# 4. Strategic Recommendation for E-Learning Collaboration

```python
data_ai_keywords = ["machine learning", "data science", "computer science"]
filtered_df = comment_df[comment_df['Keyword'].isin(data_ai_keywords)]
def sentiment_score(sentiment):
  if sentiment == "positive":
    return 1
  elif sentiment == "negative":
    return -1
  else:
    return 0
filtered_df["sentiment_score"] = filtered_df["Sentiment"].apply(sentiment_score)
filtered_df["engagement_score"] = (filtered_df["sentiment_score"] + filtered_df["comments_count"]) / 2
beginner_keywords = ["education", "tutorial", "basics"]
advanced_keywords = ["advanced", "deep learning", "research"]
weight = 0.8
filtered_df["final_score"] = (weight * filtered_df["engagement_score"]) + (1 - weight)
top_videos = filtered_df.sort_values(by="final_score", ascending=False).head(3)
top_videos_list = top_videos[["Video ID", "Title"]].to_dict('records')
for video in top_videos_list:
  print(f"\t- Video ID: {video['Video ID']}, Title: {video['Title']}")
top_video_ids = top_videos["Video ID"].to_list()
print("Top 3 video IDs for promotion:", top_video_ids)
```
