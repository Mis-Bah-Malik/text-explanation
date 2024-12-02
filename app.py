import os
import random
import streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from gtts import gTTS

# Download necessary NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

class TextProcessor:
    def __init__(self, language='english'):
        self.language = language
        self.stop_words = set(stopwords.words(language))
    
    def expand_paragraph(self, text):
        """
        Expand the given text by providing more context and details
        """
        # Tokenize sentences
        sentences = sent_tokenize(text)
        expanded_sentences = []
        
        for sentence in sentences:
            # Tokenize words
            words = word_tokenize(sentence)
            
            # Remove stop words for key terms
            key_terms = [word for word in words if word.lower() not in self.stop_words]
            
            # Generate expanded explanation
            expanded_sentence = self._generate_expansion(sentence, key_terms)
            expanded_sentences.append(expanded_sentence)
        
        return ' '.join(expanded_sentences)
    
    def _generate_expansion(self, original_sentence, key_terms):
        """
        Generate an expanded version of the sentence
        """
        if not key_terms:
            return original_sentence
        
        # Expansion templates
        expansion_templates = [
            f"To elaborate, {original_sentence} This means that the key concept revolves around the idea of {' and '.join(key_terms)}.",
            f"Let's dive deeper into the sentence: {original_sentence} The core elements include {' and '.join(key_terms)}.",
            f"Expanding on the previous statement: {original_sentence} We can understand this better by focusing on {' and '.join(key_terms)}."
        ]
        
        return random.choice(expansion_templates)
    
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

class TextToSpeech:
    def __init__(self, output_dir='audio_outputs'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def generate_speech(self, text, language='en', filename=None):
        """
        Generate speech from text using Google Text-to-Speech
        """
        if not filename:
            filename = f"speech_{language}_{hash(text)}.mp3"
        
        file_path = os.path.join(self.output_dir, filename)
        
        try:
            # Generate speech
            tts = gTTS(text=text, lang=language)
            tts.save(file_path)
            return file_path
        except Exception as e:
            st.error(f"Error generating speech: {e}")
            return None
    
    def play_audio(self, audio_file):
        """
        Play audio file in Streamlit
        """
        if audio_file and os.path.exists(audio_file):
            with open(audio_file, 'rb') as audio:
                st.audio(audio.read(), format='audio/mp3')

def main():
    st.title("📝 Text Explanation and Voice Generator")
    
    # Sidebar for configuration
    st.sidebar.header("🛠️ Application Settings")
    st.sidebar.info("Expand text and generate bilingual voice explanations")
    
    # Text input
    input_text = st.text_area("📄 Enter Text to Expand and Convert to Speech", 
                               height=200, 
                               placeholder="Paste your text here...")
    
    # Processor and TTS instances
    processor = TextProcessor()
    tts = TextToSpeech()
    
    if st.button("🔊 Expand and Generate Speech", type="primary"):
        if input_text:
            # Validate input length
            if len(input_text.split()) < 5:
                st.warning("Please enter a longer text for meaningful expansion.")
                return
            
            try:
                # Expand text
                expanded_text_en = processor.expand_paragraph(input_text)
                
                # Translate to Urdu (basic translation)
                expanded_text_ur = processor.translate_to_urdu(expanded_text_en)
                
                # Generate speech for English
                english_audio = tts.generate_speech(expanded_text_en, language='en')
                
                # Generate speech for Urdu
                urdu_audio = tts.generate_speech(expanded_text_ur, language='ur')
                
                # Display results
                st.subheader("🌍 Expanded Text (English)")
                st.write(expanded_text_en)
                
                st.subheader("🌏 Expanded Text (Urdu)")
                st.write(expanded_text_ur)
                
                # Audio playback
                st.subheader("🎧 Audio Explanations")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("🇺🇸 English Voice Explanation")
                    if english_audio:
                        tts.play_audio(english_audio)
                
                with col2:
                    st.write("🇵🇰 Urdu Voice Explanation")
                    if urdu_audio:
                        tts.play_audio(urdu_audio)
            
            except Exception as e:
                st.error(f"An error occurred: {e}")
        else:
            st.warning("Please enter some text to process.")

if __name__ == "__main__":
    main()
