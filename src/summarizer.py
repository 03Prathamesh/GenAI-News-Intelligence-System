import re
from collections import defaultdict

class TextSummarizer:
    def __init__(self):
        self.stopwords = {
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
        
    def summarize(self, text, max_sentences=3):
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) <= max_sentences:
            return text
        
        word_freq = defaultdict(int)
        words = re.findall(r'\b\w+\b', text.lower())
        
        for word in words:
            if word not in self.stopwords and len(word) > 3:
                word_freq[word] += 1
        
        sentence_scores = defaultdict(int)
        
        for i, sentence in enumerate(sentences):
            sentence_words = re.findall(r'\b\w+\b', sentence.lower())
            for word in sentence_words:
                if word in word_freq:
                    sentence_scores[i] += word_freq[word]
        
        if not sentence_scores:
            return sentences[0] if sentences else text
        
        top_sentences = sorted(sentence_scores.items(), 
                              key=lambda x: x[1], 
                              reverse=True)[:max_sentences]
        
        top_sentences = sorted([idx for idx, _ in top_sentences])
        
        summary = '. '.join([sentences[i] for i in top_sentences])
        if summary and not summary.endswith('.'):
            summary += '.'
            
        return summary
    
    def get_keywords(self, text, n=10):
        words = re.findall(r'\b\w+\b', text.lower())
        freq = defaultdict(int)
        
        for word in words:
            if word not in self.stopwords and len(word) > 3:
                freq[word] += 1
        
        if not freq:
            return []
        
        keywords = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:n]
        return [word for word, _ in keywords]