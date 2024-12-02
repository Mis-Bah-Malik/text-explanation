import streamlit as st
import traceback

# Conditional Imports
try:
    import nltk
    import spacy
    import textstat
    from gtts import gTTS
except ImportError:
    st.error("Required libraries are not installed. Please install via requirements.txt")
    st.stop()

# Optional Imports with Fallbacks
try:
    from transformers import pipeline
    has_transformers = True
except ImportError:
    has_transformers = False
    st.warning("Text generation capabilities will be limited")

try:
    import google.generativeai as genai
    has_genai = True
except ImportError:
    has_genai = False
    st.warning("Google Generative AI features will be unavailable")

# Ensure NLTK resources are downloaded
try:
    nltk.download('punkt', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
except Exception as e:
    st.warning(f"NLTK resource download failed: {e}")

class SimpleTextProcessor:
    def __init__(self):
        try:
            self.nlp = spacy.load('en_core_web_sm')
        except Exception as e:
            st.error(f"SpaCy model loading failed: {e}")
            self.nlp = None

    def analyze_paragraph(self, text):
        if not self.nlp:
            return {"error": "SpaCy model not loaded"}
        
        doc = self.nlp(text)
        
        analysis = {
            'basic_info': {
                'total_words': len(doc),
                'total_sentences': len(list(doc.sents)),
                'reading_ease': textstat.flesch_reading_ease(text),
                'reading_grade': textstat.flesch_kincaid_grade(text)
            },
            'linguistic_features': {
                'named_entities': [(ent.text, ent.label_) for ent in doc.ents],
                'key_phrases': [chunk.text for chunk in doc.noun_chunks][:5]
            }
        }
        
        return analysis

def main():
    st.set_page_config(page_title="Text Analysis", page_icon="📚")
    
    st.title("📚 Simple Text Analysis Tool")
    
    input_text = st.text_area("Enter your text here", height=200)
    
    if st.button("Analyze Text"):
        if input_text:
            try:
                processor = SimpleTextProcessor()
                result = processor.analyze_paragraph(input_text)
                
                st.subheader("Analysis Results")
                st.json(result)
                
            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.error(traceback.format_exc())
        else:
            st.warning("Please enter some text")

if __name__ == "__main__":
    main()
