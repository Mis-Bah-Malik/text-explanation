import streamlit as st
import nltk
import spacy
import textstat
import random
import os
from transformers import pipeline
from gtts import gTTS
import google.generativeai as genai
import traceback

# Ensure NLTK resources are downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

class AdvancedTextProcessor:
    def __init__(self):
        # Load spaCy model for advanced NLP
        self.nlp = spacy.load('en_core_web_sm')
        
        # Initialize text generation model
        try:
            # Use st.secrets or environment variable for API key
            api_key = st.secrets.get("GOOGLE_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            if api_key:
                genai.configure(api_key=api_key)
                self.generation_model = genai.GenerativeModel('gemini-pro')
            else:
                st.warning("Google API key not found. Generative AI features will be limited.")
                self.generation_model = None
        except Exception as e:
            st.warning(f"Generative AI model setup failed: {e}")
            self.generation_model = None
        
        # Hugging Face transformers pipeline for text generation
        try:
            self.text_generator = pipeline('text-generation', model='gpt2')
        except Exception as e:
            st.warning(f"Text generation model setup failed: {e}")
            self.text_generator = None
    
    def analyze_paragraph(self, text):
        """
        Comprehensive paragraph analysis
        """
        # Basic NLP processing
        doc = self.nlp(text)
        
        # Linguistic Analysis
        analysis = {
            'basic_info': {
                'total_words': len(doc),
                'total_sentences': len(list(doc.sents)),
                'reading_ease': textstat.flesch_reading_ease(text),
                'reading_grade': textstat.flesch_kincaid_grade(text)
            },
            'linguistic_features': {
                'named_entities': [(ent.text, ent.label_) for ent in doc.ents],
                'pos_distribution': self._get_pos_distribution(doc),
                'key_phrases': self._extract_key_phrases(doc)
            }
        }
        
        return analysis
    
    def _get_pos_distribution(self, doc):
        """
        Get Part of Speech distribution
        """
        pos_counts = {}
        for token in doc:
            pos = token.pos_
            pos_counts[pos] = pos_counts.get(pos, 0) + 1
        return pos_counts
    
    def _extract_key_phrases(self, doc):
        """
        Extract key phrases using linguistic rules
        """
        # Extract noun phrases and chunks
        key_phrases = [chunk.text for chunk in doc.noun_chunks]
        return list(set(key_phrases))[:5]  # Return top 5 unique phrases
    
    def generate_comprehensive_explanation(self, text):
        """
        Generate in-depth explanation of the paragraph
        """
        # Try advanced AI-powered explanation
        if self.generation_model:
            try:
                prompt = f"Provide a comprehensive, academic-style explanation of the following paragraph, breaking down its core ideas, linguistic nuances, and deeper meanings:\n\n{text}"
                response = self.generation_model.generate_content(prompt)
                return response.text
            except Exception as e:
                st.warning("AI-powered explanation failed. Using fallback method.")
        
        # Fallback explanation generation
        analysis = self.analyze_paragraph(text)
        
        explanation_template = f"""
        Paragraph Comprehensive Analysis:

        1. Structural Overview:
        - Total Words: {analysis['basic_info']['total_words']}
        - Total Sentences: {analysis['basic_info']['total_sentences']}

        2. Readability Metrics:
        - Flesch Reading Ease: {analysis['basic_info']['reading_ease']} 
          (Higher score indicates easier readability)
        - Flesch-Kincaid Grade Level: {analysis['basic_info']['reading_grade']}

        3. Linguistic Composition:
        Part of Speech Distribution:
        {', '.join([f"{k}: {v}" for k, v in analysis['linguistic_features']['pos_distribution'].items()])}

        4. Key Phrases and Entities:
        Named Entities: {analysis['linguistic_features']['named_entities']}
        Key Phrases: {analysis['linguistic_features']['key_phrases']}

        5. Interpretative Explanation:
        {self._generate_interpretative_text(text)}
        """
        
        return explanation_template
    
    def _generate_interpretative_text(self, text):
        """
        Generate an interpretative explanation using text generator
        """
        if not self.text_generator:
            return "Interpretative explanation could not be generated due to model limitations."
        
        try:
            generated_text = self.text_generator(
                f"Explain the deeper meaning of: {text}", 
                max_length=200, 
                num_return_sequences=1
            )[0]['generated_text']
            return generated_text
        except Exception:
            return "An interpretative explanation could not be generated."
    
    def translate_to_urdu(self, text):
        """
        Advanced translation simulation 
        (Recommend using professional translation service in production)
        """
        translation_patterns = [
            ("the", "اس"),
            ("is", "ہے"),
            ("are", "ہیں"),
            ("and", "اور"),
            ("in", "میں"),
            ("of", "کا"),
            ("to", "کو")
        ]
        
        for eng, urdu in translation_patterns:
            text = text.replace(eng, urdu)
        
        return text

class TextToSpeechAdvanced:
    def __init__(self, output_dir='audio_outputs'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def generate_speech(self, text, language='en'):
        """
        Generate high-quality speech with error handling
        """
        try:
            filename = f"speech_{language}_{hash(text)}.mp3"
            file_path = os.path.join(self.output_dir, filename)
            
            tts = gTTS(text=text, lang=language, slow=False)
            tts.save(file_path)
            return file_path
        except Exception as e:
            st.error(f"Speech generation error: {e}")
            return None

def main():
    st.set_page_config(
        page_title="Advanced Text Explanation",
        page_icon="📚",
        layout="wide"
    )
    
    # Title and Description
    st.title("📚 Advanced Paragraph Explanation & Analysis")
    st.markdown("""
    ### Comprehensive Text Understanding Tool
    - Deep linguistic analysis
    - Multi-language explanation
    - Text-to-Speech generation
    """)
    
    # Sidebar for configuration
    st.sidebar.header("🔧 Advanced Settings")
    language_option = st.sidebar.selectbox(
        "Select Output Language", 
        ["English", "Urdu"]
    )
    
    # Text input area
    input_text = st.text_area(
        "📝 Enter Paragraph for Comprehensive Analysis", 
        height=250,
        placeholder="Paste your paragraph here for in-depth explanation..."
    )
    
    # Process Button
    if st.button("🔍 Analyze Paragraph", type="primary"):
        if input_text.strip():
            try:
                # Initialize processors
                text_processor = AdvancedTextProcessor()
                tts_processor = TextToSpeechAdvanced()
                
                # Generate comprehensive explanation
                comprehensive_explanation = text_processor.generate_comprehensive_explanation(input_text)
                
                # Create tabs for different views
                tab1, tab2, tab3 = st.tabs([
                    "📊 Linguistic Analysis", 
                    "🔬 Comprehensive Explanation", 
                    "🎧 Audio Explanation"
                ])
                
                with tab1:
                    # Linguistic Analysis
                    analysis_result = text_processor.analyze_paragraph(input_text)
                    st.json(analysis_result)
                
                with tab2:
                    # Comprehensive Explanation
                    st.markdown("### 🧠 Paragraph Breakdown")
                    st.write(comprehensive_explanation)
                
                with tab3:
                    # Audio Generation
                    st.markdown("### 🔊 Audio Explanations")
                    
                    # English Audio
                    st.subheader("🇺🇸 English Explanation")
                    english_audio = tts_processor.generate_speech(comprehensive_explanation, 'en')
                    if english_audio:
                        with open(english_audio, 'rb') as audio_file:
                            st.audio(audio_file.read(), format='audio/mp3')
                    
                    # Urdu Audio (if selected)
                    if language_option == "Urdu":
                        st.subheader("🇵🇰 Urdu Explanation")
                        urdu_text = text_processor.translate_to_urdu(comprehensive_explanation)
                        urdu_audio = tts_processor.generate_speech(urdu_text, 'ur')
                        if urdu_audio:
                            with open(urdu_audio, 'rb') as audio_file:
                                st.audio(audio_file.read(), format='audio/mp3')
            
            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.error(traceback.format_exc())
        else:
            st.warning("Please enter a paragraph for analysis.")

if __name__ == "__main__":
    main()
