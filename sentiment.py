from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

# Sample tweet
tweet = "@MehranShakarami today's cold @ home 😒 https://mehranshakarami.com"

# Preprocess the tweet
tweet_words = []
for word in tweet.split(' '):
    if word.startswith('@') and len(word) > 1:
        word = '@user'
    elif word.startswith('https'):
        word = "https"
    tweet_words.append(word)

cleaned_tweet = " ".join(tweet_words)

# Load model and tokenizer
roberta = "cardiffnlp/twitter-roberta-base-sentiment"
model = AutoModelForSequenceClassification.from_pretrained(roberta)
tokenizer = AutoTokenizer.from_pretrained(roberta)

# Sentiment labels
labels = ['Negative', 'Neutral', 'Positive']

# Encode the tweet
encoded_tweet = tokenizer(cleaned_tweet, return_tensors='pt')

# Get model output
output = model(**encoded_tweet)
scores = output.logits[0].detach().numpy()
scores = softmax(scores)

# Print the scores and labels
for i in range(len(scores)):
    print(f"{labels[i]}: {scores[i]:.4f}")

# Print all scores for verification
print("Scores:", scores)
