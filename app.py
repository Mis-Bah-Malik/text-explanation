import streamlit as st
import base64
from advanced_text_analyzer import AdvancedTextAnalyzer
from gtts import gTTS
import os

class TextToSpeechGenerator:
    def __init__(self, output_dir='audio_outputs'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def generate_speech(self, text, language='en'):
        """
        Generate speech from text using Google Text-to-Speech
        """
        filename = f"speech_{language}_{hash(text)}.mp3"
        file_path = os.path.join(self.output_dir, filename)
        
        try:
            tts = gTTS(text=text, lang=language)
            tts.save(file_path)
            return file_path
        except Exception as e:
            st.error(f"Speech generation error: {e}")
            return None

def download_button(object_to_download, download_filename, button_text):
    """
    Generates a link to download the given object_to_download.
    """
    try:
        # Convert object to bytes
        b64 = base64.b64encode(str(object_to_download).encode()).decode()
    except Exception as e:
        b64 = base64.b64encode(object_to_download.encode()).decode()

    return f'''
        <a href="data:file/txt;base64,{b64}" download="{download_filename}">
            <button>{button_text}</button>
        </a>
    '''

def main():
    st.set_page_config(
        page_title="Advanced Text Intelligence",
        page_icon="🔍",
        layout="wide"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .reportview-container {
        background: linear-gradient(to right, #f0f2f6, #e6e9ef);
    }
    .sidebar .sidebar-content {
        background: linear-gradient(to bottom, #f0f2f6, #e6e9ef);
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Title and description
    st.title("🔬 Advanced Text Intelligence Platform")
    st.markdown("## Unlock Deeper Insights from Your Text")
    
    # Sidebar
    st.sidebar.header("🛠 Analysis Configuration")
    analysis_mode = st.sidebar.selectbox(
        "Analysis Mode",
        ["Comprehensive", "Quick Insights", "Detailed Breakdown"]
    )
    
    language_option = st.sidebar.selectbox(
        "Output Language",
        ["English", "Multi-Lingual Insights"]
    )
    
    # Text input area
    input_text = st.text_area(
        "📝 Enter Your Text for Deep Analysis", 
        height=300,
        placeholder="Paste your text here... (Minimum 50 words recommended)"
    )
    
    # Analysis button
    if st.button("🔍 Analyze Text", type="primary"):
        # Input validation
        if len(input_text.split()) < 10:
            st.warning("Please provide more substantial text for meaningful analysis.")
            return
        
        # Initialize analyzers
        text_analyzer = AdvancedTextAnalyzer()
        tts_generator = TextToSpeechGenerator()
        
        try:
            # Comprehensive analysis
            analysis_result = text_analyzer.generate_comprehensive_explanation(input_text)
            advanced_insights = text_analyzer.generate_advanced_insights(input_text)
            
            # Display results in tabs
            tab1, tab2, tab3 = st.tabs(["📊 Analysis Overview", "🧠 Advanced Insights", "🎙️ Audio"])
            
            with tab1:
                st.subheader("Text Intelligence Report")
                st.markdown(analysis_result['full_explanation'])
                
                # Download analysis report
                download_html = download_button(
                    analysis_result['full_explanation'], 
                    'text_analysis_report.txt', 
                    '📥 Download Analysis'
                )
                st.markdown(download_html, unsafe_allow_html=True)
            
            with tab2:
                st.subheader("Deep Contextual Insights")
                st.write(advanced_insights)
                
                # Entity and sentiment visualization
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Sentiment Polarity", f"{analysis_result['sentiment']['polarity']:.2f}")
                with col2:
                    st.metric("Subjectivity", f"{analysis_result['sentiment']['subjectivity']:.2f}")
            
            with tab3:
                st.subheader("Audio Explanation")
                
                # Generate audio
                audio_file = tts_generator.generate_speech(analysis_result['full_explanation'])
                if audio_file:
                    st.audio(audio_file, format='audio/mp3')
        
        except Exception as e:
            st.error(f"Analysis Error: {e}")

if __name__ == "__main__":
    main()
