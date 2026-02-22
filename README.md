# Sign Language Translator Dashboard

Hey everyone! This is my real-time, multi-lingual Sign Language Translator built with Python and Streamlit.

I originally started building this with a machine learning model (Random Forest trained on hand landmarks), but I quickly realized that for static, conversational signs, a **math/heuristic-based approach is actually WAY faster and perfectly accurate** without needing to collect gigabytes of training data. 

So I completely rewrote the engine! It now uses MediaPipe to track 21 hand landmarks and calculates exactly which of the 5 fingers are open or closed based on their geometric distance from the palm. Based on those combinations, it translates the gesture.

### Features
* **Real-time translation:** Uses your webcam to detect signs instantly.
* **Math-based heuristics:** No ML models required! Just pure coordinate geometry.
* **Trilingual Audio Output:** Translates the gestures into English, Hindi, and Marathi text.
* **Threaded TTS:** I noticed the Google Text-to-Speech (gTTS) API was freezing my OpenCV video feed while it downloaded the audio, so I threw the audio player into a separate background thread! Frame rates are buttery smooth now.

### Supported Signs
The system currently recognizes 20 distinct conversational combinations (e.g., "Hello", "I Love You", "Peace", "Awesome", "Excuse me"). You can see the full tuple mapping inside `app.py` under the `SIGNS_MAPPING` dictionary.

### How to Run
1. Clone the repo
2. Install the requirements: `pip install -r requirements.txt`
3. Run the Streamlit server: `streamlit run app.py`

Enjoy and feel free to contribute!
