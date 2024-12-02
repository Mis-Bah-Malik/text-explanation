import streamlit as st
import nltk
import spacy
import textstat

# Ensure NLTK resources are available
try:
    nltk.download('punkt', quiet=True)
except Exception as e:
    st.warning(f"NLTK download error: {e}")

class TextAnalyzer:
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')

    def analyze_text(self, text):
        doc = self.nlp(text)
        
        return {
            'word_count': len(doc),
            'sentence_count': len(list(doc.sents)),
            'reading_ease': textstat.flesch_reading_ease(text),
            'reading_grade': textstat.flesch_kincaid_grade(text),
            'named_entities': [(ent.text, ent.label_) for ent in doc.ents]
        }

def main():
    st.title("🔍 Simple Text Analysis")
    
    text_input = st.text_area("Enter your text here:", height=200)
    
    if st.button("Analyze"):
        try:
            analyzer = TextAnalyzer()
            result = analyzer.analyze_text(text_input)
            st.json(result)
        except Exception as e:
            st.error(f"Analysis error: {e}")

if __name__ == "__main__":
    main()
