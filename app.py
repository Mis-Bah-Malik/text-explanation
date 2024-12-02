import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
import random
import spacy

class AdvancedTextProcessor:
    def __init__(self, language='english'):
        # Download necessary NLTK resources
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        
        # Initialize NLTK and spaCy resources
        self.language = language
        self.stop_words = set(stopwords.words(language))
        self.lemmatizer = WordNetLemmatizer()
        
        # Load spaCy model for advanced NLP processing
        try:
            self.nlp = spacy.load('en_core_web_sm')
        except OSError:
            print("Downloading spaCy English model...")
            from spacy.cli import download
            download('en_core_web_sm')
            self.nlp = spacy.load('en_core_web_sm')
    
    def extract_key_insights(self, text):
        """
        Extract key insights, entities, and relationships from the text
        """
        # Process text with spaCy
        doc = self.nlp(text)
        
        # Extract named entities
        entities = {}
        for ent in doc.ents:
            if ent.label_ not in entities:
                entities[ent.label_] = []
            entities[ent.label_].append(ent.text)
        
        # Extract important noun phrases
        noun_phrases = [chunk.text for chunk in doc.noun_chunks]
        
        # Extract key verbs and actions
        verbs = [token.lemma_ for token in doc if token.pos_ == "VERB"]
        
        return {
            'entities': entities,
            'noun_phrases': list(set(noun_phrases)),
            'verbs': list(set(verbs))
        }
    
    def generate_comprehensive_explanation(self, text):
        """
        Generate a comprehensive, multi-layered explanation of the text
        """
        # Basic sentence tokenization
        sentences = sent_tokenize(text)
        
        # Extract key insights
        insights = self.extract_key_insights(text)
        
        # Explanation sections
        explanations = []
        
        # 1. Overview Section
        overview = f"Overview: The text discusses a topic that involves key elements such as: {', '.join(insights['noun_phrases'][:5])}."
        explanations.append(overview)
        
        # 2. Detailed Explanation of Entities
        if insights['entities']:
            entity_explanation = "Key Entities Breakdown:\n"
            for entity_type, entities in insights['entities'].items():
                entity_explanation += f"- {entity_type} Entities: {', '.join(entities[:3])}\n"
            explanations.append(entity_explanation)
        
        # 3. Contextual Analysis
        context_analysis = "Contextual Analysis:\n"
        context_templates = [
            "The text explores the interconnection between {topics} through multiple perspectives.",
            "Key actions and themes include: {verbs}, which provide insight into the core message.",
            "The narrative weaves together concepts of {topics} to communicate its central idea."
        ]
        
        # Randomly select and format a context template
        context_template = random.choice(context_templates).format(
            topics=' and '.join(insights['noun_phrases'][:3]),
            verbs=' and '.join(insights['verbs'][:3])
        )
        context_analysis += context_template
        explanations.append(context_analysis)
        
        # 4. Sentence-Level Deep Dive
        if len(sentences) > 1:
            deep_dive = "Sentence-Level Insights:\n"
            for i, sentence in enumerate(sentences, 1):
                # Process each sentence
                sent_doc = self.nlp(sentence)
                
                # Extract key tokens
                key_tokens = [
                    token.text for token in sent_doc 
                    if not token.is_stop and token.pos_ in ["NOUN", "VERB", "ADJ"]
                ]
                
                deep_dive += f"Sentence {i}: {sentence}\n"
                deep_dive += f"Key Elements: {', '.join(key_tokens)}\n\n"
            
            explanations.append(deep_dive)
        
        # 5. Conclusion
        conclusion = f"Conclusion: The text fundamentally revolves around {' and '.join(insights['noun_phrases'][:2])}, " \
                     f"highlighting the importance of {' and '.join(insights['verbs'][:2])} in understanding the core message."
        explanations.append(conclusion)
        
        # Combine all explanations
        full_explanation = "\n\n".join(explanations)
        
        return full_explanation
    
    def translate_to_urdu(self, text):
        """
        Basic translation simulation (replace with proper translation service)
        """
        # Rudimentary translation mapping
        translation_map = {
            'the': 'اس',
            'is': 'ہے',
            'are': 'ہیں',
            'and': 'اور',
            'in': 'میں',
            'of': 'کا',
            'to': 'کو'
        }
        
        words = text.split()
        translated_words = [translation_map.get(word.lower(), word) for word in words]
        return ' '.join(translated_words)
