import os
import pandas as pd
import jieba
import requests
from collections import Counter
import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from gensim import models
import gensim.corpora as corpora
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis

# Load the data from GitHub
file_url = 'https://github.com/WENDI6666/BEMM466-Business-Project/raw/main/Taobao-2021-2024.xlsx'
df = pd.read_excel(file_url)

# Remove duplicate and short comments
df = df.drop_duplicates(subset=['comment'])
df = df[df['comment'].str.len() > 2]

# Load stopwords from GitHub
stopwords_url = "https://raw.githubusercontent.com/WENDI6666/BEMM466-Business-Project/1cde83fe1453aa539850472e6fb6ccd9e2e0ae91/stopword(1).txt"
stopwords = set(requests.get(stopwords_url).text.splitlines())

# Load custom dictionary for jieba
user_dict_url = 'https://github.com/WENDI6666/BEMM466-Business-Project/raw/9e2dc9ed8541f3a1f2e5cc5c44a811a1cb98d40b/userdict.txt'
user_dict = requests.get(user_dict_url).text
with open('temp_userdict.txt', 'w', encoding='utf-8') as f:
    f.write(user_dict)
jieba.load_userdict('temp_userdict.txt')

# Synonym dictionary
synonym_dict = {
    '快递': ['物流','配送'],
    '客服': ['小蜜','客户服务','人工客服','平台客服','商家客服','客服客服'],
    '价格': ['价钱', '费用'],
    '垃圾': ['辣鸡','垃圾淘宝','辣鸡淘宝','垃圾垃圾','垃圾垃圾垃圾'],
    '拼多多': ['pdd','拼夕夕'],
    '评论': ['评价'],
    '封号': ['封禁账号'],
    '平板': ['iPad','ipad','pad'],
    '微信支付': ['微信付款'],
    '商品': ['产品'],
    '页面': ['界面'],
    '深色模式': ['夜间模式'],
    '晚上': ['夜间'],
    '道歉': ['抱歉'],
    '客户': ['买家','用户'],
    '购物': ['买东西','淘','买','逛']
}

# Function for segmentation, synonym replacement, and stopword removal
def segment_and_replace_synonyms(text):
    text = re.sub(r"[^\w\u4e00-\u9fa5]", "", text.lower())  # Remove non-text characters, no spaces added
    words = jieba.lcut(text)
    processed_words = []
    for word in words:
        if len(word) >= 2 and word not in stopwords:
            for key, synonyms in synonym_dict.items():
                if word in synonyms:
                    word = key  # Replace synonym with key
                    break
            processed_words.append(word)
    return processed_words

# Apply text processing
df['segmented_comment'] = df['comment'].apply(segment_and_replace_synonyms)

# Word frequency count
word_counts = Counter(word for sentence in df['segmented_comment'] for word in sentence)

# Display top 20 word frequencies
top_20_word_counts = word_counts.most_common(20)
print("Top 20 word frequencies:")
for word, count in top_20_word_counts:
    print(f"{word}: {count}")

# Convert list of words to a string for TF-IDF and LDA
df['segmented_comment'] = df['segmented_comment'].apply(lambda x: ' '.join(x))

# Compute TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
tfidf_matrix = vectorizer.fit_transform(df['segmented_comment'])
words = vectorizer.get_feature_names_out()
tfidf_sum = tfidf_matrix.sum(axis=0)
top_tfidf_words = pd.Series(tfidf_sum.A1, index=words).sort_values(ascending=False).head(20)

print("Top 20 TF-IDF words:")
print(top_tfidf_words)

# Generate a word cloud
top_200_words = dict(word_counts.most_common(200))
wordcloud = WordCloud(
    font_path='simhei.ttf',
    width=800,
    height=600,
    background_color='white'
).generate_from_frequencies(top_200_words)

plt.figure(figsize=(10, 8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

# LDA Model
tf_vectorizer = CountVectorizer(max_features=1000, max_df=0.5, min_df=10)
tf = tf_vectorizer.fit_transform(df['segmented_comment'])
dictionary = corpora.Dictionary(df['segmented_comment'].apply(lambda x: x.split()))
corpus_bow = [dictionary.doc2bow(text.split()) for text in df['segmented_comment']]

lda_model = models.LdaModel(corpus_bow, num_topics=6, id2word=dictionary, passes=20, random_state=42)
for idx, topic in lda_model.print_topics(num_words=10):
    print(f"Topic {idx + 1}: {topic}")

# Visualize LDA model
lda_display = gensimvis.prepare(lda_model, corpus_bow, dictionary)
pyLDAvis.display(lda_display)

# Function to load sentiment dictionary from a URL
def load_sentiment_dictionary_from_url(url):
    response = requests.get(url)
    response.raise_for_status()  # Raise an error if the request failed
    return [line.strip() for line in response.text.splitlines()]

# Load stopwords from the new GitHub URL
new_stopwords_url = "https://raw.githubusercontent.com/WENDI6666/BEMM466-Business-Project/1cde83fe1453aa539850472e6fb6ccd9e2e0ae91/stopword(2)%20.txt"
new_stopwords = set(requests.get(new_stopwords_url).text.splitlines())

# Sentiment analysis using dictionaries from GitHub
sentiment_urls = {
    "negative_evaluation_words": "https://raw.githubusercontent.com/WENDI6666/BEMM466-Business-Project/1cde83fe1453aa539850472e6fb6ccd9e2e0ae91/%E8%B4%9F%E9%9D%A2%E8%AF%84%E4%BB%B7%E8%AF%8D%E8%AF%AD%EF%BC%88%E4%B8%AD%E6%96%87%EF%BC%89.txt",
    "negative_sentiment_words": "https://raw.githubusercontent.com/WENDI6666/BEMM466-Business-Project/1cde83fe1453aa539850472e6fb6ccd9e2e0ae91/%E8%B4%9F%E9%9D%A2%E6%83%85%E6%84%9F%E8%AF%8D%E8%AF%AD%EF%BC%88%E4%B8%AD%E6%96%87%EF%BC%89.txt",
    "positive_evaluation_words": "https://raw.githubusercontent.com/WENDI6666/BEMM466-Business-Project/1cde83fe1453aa539850472e6fb6ccd9e2e0ae91/%E6%AD%A3%E9%9D%A2%E8%AF%84%E4%BB%B7%E8%AF%8D%E8%AF%AD%EF%BC%88%E4%B8%AD%E6%96%87%EF%BC%89.txt",
    "positive_sentiment_words": "https://raw.githubusercontent.com/WENDI6666/BEMM466-Business-Project/1cde83fe1453aa539850472e6fb6ccd9e2e0ae91/%E6%AD%A3%E9%9D%A2%E6%83%85%E6%84%9F%E8%AF%8D%E8%AF%AD%EF%BC%88%E4%B8%AD%E6%96%87%EF%BC%89.txt",
    "custom_negative_words": "https://raw.githubusercontent.com/WENDI6666/BEMM466-Business-Project/1cde83fe1453aa539850472e6fb6ccd9e2e0ae91/custom_negative_words.txt",
    "custom_positive_words": "https://raw.githubusercontent.com/WENDI6666/BEMM466-Business-Project/1cde83fe1453aa539850472e6fb6ccd9e2e0ae91/custom_positive_words.txt",
    "negation_words": "https://raw.githubusercontent.com/WENDI6666/BEMM466-Business-Project/1cde83fe1453aa539850472e6fb6ccd9e2e0ae91/Negation%20Words.txt"
}

negative_words = set(load_sentiment_dictionary_from_url(sentiment_urls["negative_evaluation_words"]) +
                     load_sentiment_dictionary_from_url(sentiment_urls["negative_sentiment_words"]) +
                     load_sentiment_dictionary_from_url(sentiment_urls["custom_negative_words"]))

positive_words = set(load_sentiment_dictionary_from_url(sentiment_urls["positive_evaluation_words"]) +
                     load_sentiment_dictionary_from_url(sentiment_urls["positive_sentiment_words"]) +
                     load_sentiment_dictionary_from_url(sentiment_urls["custom_positive_words"]))

negation_words = load_sentiment_dictionary_from_url(sentiment_urls["negation_words"])

# Sentiment scoring function
def sentiment_score(segmented_comment):
    score = 0
    negation = 1
    for word in segmented_comment.split():
        if word in negation_words:
            negation = -1
        elif word in positive_words:
            score += 2 * negation
            negation = 1
        elif word in negative_words:
            score -= 2 * negation
            negation = 1
    return score

# Apply sentiment analysis
df['sentiment_score'] = df['segmented_comment'].apply(sentiment_score)

# Export results to CSV
output_file = os.path.join(os.path.expanduser("~"), 'Desktop', 'sentiment_scores_with_custom_dict.csv')
df.to_csv(output_file, index=False, encoding='utf-8-sig')

print(f"Sentiment scores have been exported to the desktop as {output_file}")