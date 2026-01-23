from textblob import TextBlob

def analyze_sentiment(text):
    try:
        blob = TextBlob(str(text))
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        if polarity > 0.1:
            sentiment = 'POSITIVE'
        elif polarity < -0.1:
            sentiment = 'NEGATIVE'
        else:
            sentiment = 'NEUTRAL'
        
        return {
            'sentiment': sentiment,
            'compound': polarity,
            'positive': max(0, polarity),
            'negative': abs(min(0, polarity)),
            'neutral': 1 - abs(polarity)
        }
    except:
        return {
            'sentiment': 'NEUTRAL',
            'compound': 0,
            'positive': 0,
            'negative': 0,
            'neutral': 1
        }