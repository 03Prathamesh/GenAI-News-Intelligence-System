import re

def clean_text(text):
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    text = re.sub(r'<.*?>', '', text)
    
    text = re.sub(r'[^a-z\s]', '', text)
    
    text = re.sub(r'\s+', ' ', text).strip()
    
    words = text.split()
    
    stop_words = {
        'a', 'an', 'the', 'and', 'or', 'but', 'if', 'while', 'when', 'where',
        'how', 'what', 'why', 'which', 'who', 'whom', 'this', 'that', 'these',
        'those', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have',
        'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'will', 'would',
        'shall', 'should', 'can', 'could', 'may', 'might', 'must', 'for', 'of',
        'to', 'in', 'on', 'at', 'by', 'with', 'about', 'against', 'between',
        'into', 'through', 'during', 'before', 'after', 'above', 'below', 'from',
        'up', 'down', 'out', 'off', 'over', 'under', 'again', 'further', 'then',
        'once', 'here', 'there', 'all', 'any', 'both', 'each', 'few', 'more',
        'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
        'same', 'so', 'than', 'too', 'very', 'just', 'now'
    }
    
    cleaned_words = [word for word in words if word not in stop_words and len(word) > 2]
    
    cleaned_text = ' '.join(cleaned_words)
    
    return cleaned_text