import spacy
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from typing import Dict, List, Any
import random
import textblob
import transformers

class AdvancedTextAnalyzer:
    def __init__(self, language='en'):
        # Download necessary resources
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)

        # Load spaCy model
        try:
            self.nlp = spacy.load('en_core_web_lg')
        except OSError:
            print("Downloading spaCy English large model...")
            spacy.cli.download('en_core_web_lg')
            self.nlp = spacy.load('en_core_web_lg')
        
        # Load Hugging Face transformer for advanced NLP
        self.summarizer = transformers.pipeline("summarization")
        
        # Set language-specific resources
        self.stop_words = set(stopwords.words(language))
        self.language = language

    def extract_key_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract and categorize named entities from the text
        """
        doc = self.nlp(text)
        entities = {}
        
        for ent in doc.ents:
            if ent.label_ not in entities:
                entities[ent.label_] = []
            entities[ent.label_].append(ent.text)
        
        return entities

    def sentiment_analysis(self, text: str) -> Dict[str, float]:
        """
        Perform sentiment analysis on the text
        """
        blob = textblob.TextBlob(text)
        return {
            'polarity': blob.sentiment.polarity,
            'subjectivity': blob.sentiment.subjectivity
        }

    def generate_comprehensive_explanation(self, text: str) -> Dict[str, Any]:
        """
        Generate a multi-dimensional analysis of the text
        """
        # Basic text processing
        doc = self.nlp(text)
        sentences = sent_tokenize(text)
        
        # Entity extraction
        entities = self.extract_key_entities(text)
        
        # Sentiment analysis
        sentiment = self.sentiment_analysis(text)
        
        # Key phrase extraction
        noun_chunks = [chunk.text for chunk in doc.noun_chunks]
        key_phrases = list(set(noun_chunks))[:10]
        
        # Text summarization
        try:
            summary = self.summarizer(text, max_length=130, min_length=30, do_sample=False)[0]['summary_text']
        except Exception:
            summary = sentences[0] if sentences else text[:200]
        
        # Advanced linguistic breakdown
        linguistic_analysis = {
            'total_words': len(word_tokenize(text)),
            'unique_words': len(set(word.lower() for word in word_tokenize(text) if word.isalnum())),
            'avg_sentence_length': sum(len(word_tokenize(sentence)) for sentence in sentences) / len(sentences)
        }
        
        # Comprehensive explanation generation
        explanation_components = [
            f"Linguistic Overview: The text contains {linguistic_analysis['total_words']} words "
            f"with an average sentence length of {linguistic_analysis['avg_sentence_length']:.2f}.",
            
            f"Key Entities: Discovered {sum(len(entities.get(k, [])) for k in entities)} significant entities "
            f"across categories like {', '.join(entities.keys())}.",
            
            f"Sentiment Analysis: The text has a polarity of {sentiment['polarity']:.2f} "
            f"({'positive' if sentiment['polarity'] > 0 else 'negative' if sentiment['polarity'] < 0 else 'neutral'}) "
            f"with a subjectivity of {sentiment['subjectivity']:.2f}.",
            
            f"Key Phrases: {', '.join(key_phrases)}",
            
            f"Summary: {summary}"
        ]
        
        return {
            'full_explanation': '\n\n'.join(explanation_components),
            'entities': entities,
            'sentiment': sentiment,
            'key_phrases': key_phrases,
            'linguistic_analysis': linguistic_analysis,
            'summary': summary
        }

    def generate_advanced_insights(self, text: str) -> str:
        """
        Generate advanced insights with narrative explanation
        """
        analysis = self.generate_comprehensive_explanation(text)
        
        insights_templates = [
            "The text reveals a complex narrative characterized by {complexity_description}. "
            "Key entities such as {entities} play a pivotal role in understanding its deeper meaning.",
            
            "Diving into the linguistic landscape, we uncover a {sentiment_tone} exploration "
            "that touches upon critical themes like {key_phrases}.",
            
            "This text is a nuanced composition that balances {linguistic_characteristics}, "
            "offering insights into {thematic_elements}."
        ]
        
        # Generate dynamic insights
        complexity_desc = 'intricate linguistic patterns' if analysis['linguistic_analysis']['avg_sentence_length'] > 15 else 'concise communication'
        sentiment_tone = 'emotionally charged' if abs(analysis['sentiment']['polarity']) > 0.5 else 'balanced'
        
        insights = random.choice(insights_templates).format(
            complexity_description=complexity_desc,
            entities=', '.join(analysis['entities'].get('PERSON', [])[:3]),
            sentiment_tone=sentiment_tone,
            key_phrases=', '.join(analysis['key_phrases'][:3]),
            linguistic_characteristics='brevity and depth',
            thematic_elements='contemporary discourse'
        )
        
        return insights

# Example usage
if __name__ == "__main__":
    analyzer = AdvancedTextAnalyzer()
    sample_text = "Climate change is one of the most significant challenges facing our global community today."
    result = analyzer.generate_comprehensive_explanation(sample_text)
    print(result['full_explanation'])
