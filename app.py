import streamlit as st
import cv2
import mediapipe as mp
import time
import os
import threading
from gtts import gTTS
import pygame

# Initialize pygame mixer once
if not pygame.mixer.get_init():
    pygame.mixer.init()

st.set_page_config(page_title="Heuristic Sign Language Dashboard", layout="wide")

# ==========================
# TRANSLATIONS & MAPPING
# ==========================

SIGNS_MAPPING = {
    (True, True, True, True, True): 'Hello',
    (False, False, False, False, False): 'I need help',
    (True, False, False, False, True): 'Need Water',
    (False, True, True, False, False): 'Peace',
    (False, True, False, False, False): 'Pay Attention',
    (True, True, False, False, False): 'Wait here',
    (False, False, False, False, True): 'Excuse me',
    (False, True, True, True, False): 'Options',
    (False, True, False, False, True): 'Awesome',
    (True, False, False, False, False): 'Good Job',
    (True, True, False, False, True): 'I Love You',
    (False, False, True, True, True): 'Perfect',
    (False, True, True, True, True): 'Many things',
    (True, False, True, True, True): 'Almost done',
    (True, True, True, True, False): 'Stop immediately',
    (False, False, False, True, True): 'Together',
    (True, False, False, True, True): 'Friends',
    (True, True, True, False, True): 'What is this?',
    (False, True, False, True, False): 'Confused',
    (False, False, True, True, False): 'Sorry'
}

TRANSLATIONS = {
    'Hello': {'English': 'Hello!', 'Hindi': 'नमस्ते, कैसे हो?', 'Marathi': 'नमस्कार, कसं काय?'},
    'I need help': {'English': 'I need help', 'Hindi': 'मुझे थोड़ी मदद चाहिए', 'Marathi': 'मला थोडी मदत हवी आहे'},
    'Need Water': {'English': 'Need Water', 'Hindi': 'मुझे थोड़ा पानी मिलेगा?', 'Marathi': 'थोडं पाणी मिळेल का?'},
    'Peace': {'English': 'Peace', 'Hindi': 'सब ठीक है यार', 'Marathi': 'सगळं ठीक आहे भावा'},
    'Pay Attention': {'English': 'Pay Attention', 'Hindi': 'ज़रा इधर ध्यान दो', 'Marathi': 'जरा इकडे लक्ष द्या'},
    'Wait here': {'English': 'Wait here', 'Hindi': 'थोड़ी देर यही रुको', 'Marathi': 'थोड्या वेळ इथेच थांबा'},
    'Excuse me': {'English': 'Excuse me', 'Hindi': 'सुनिए मेरी बात', 'Marathi': 'अहो, ऐका कि'},
    'Options': {'English': 'Options', 'Hindi': 'यहाँ और क्या क्या है?', 'Marathi': 'अजून काय पर्याय आहेत?'},
    'Awesome': {'English': 'Awesome!', 'Hindi': 'एकदम कमाल है यार!', 'Marathi': 'एकदम भारी झक्कास!'},
    'Good Job': {'English': 'Good Job!', 'Hindi': 'क्या बात है, बहुत बढ़िया!', 'Marathi': 'खूपच छान काम केलंय!'},
    'I Love You': {'English': 'I Love You', 'Hindi': 'मैं तुमसे प्यार करता हूँ', 'Marathi': 'माझं तुझ्यावर प्रेम आहे'},
    'Perfect': {'English': 'Perfect', 'Hindi': 'एकदम सही कहा', 'Marathi': 'एकदम बरोबर'},
    'Many things': {'English': 'Many things', 'Hindi': 'बहुत सारी बातें हैं', 'Marathi': 'खूप साऱ्या गोष्टी आहेत'},
    'Almost done': {'English': 'Almost done', 'Hindi': 'बस हो ही गया', 'Marathi': 'जवळपास संपलंच आहे'},
    'Stop immediately': {'English': 'Stop immediately', 'Hindi': 'अभी के अभी रुक जाओ!', 'Marathi': 'लगेच तिथेच थांबा!'},
    'Together': {'English': 'Together', 'Hindi': 'हम सब साथ हैं', 'Marathi': 'आपण सगळे एकत्र आहोत'},
    'Friends': {'English': 'Friends', 'Hindi': 'हम पक्के दोस्त हैं', 'Marathi': 'आपण घट्ट मित्र आहोत'},
    'What is this?': {'English': 'What is this?', 'Hindi': 'भाई ये सब क्या है?', 'Marathi': 'हे सगळं काय चाललंय?'},
    'Confused': {'English': 'Confused', 'Hindi': 'मुझे कुछ समझ नहीं आ रहा', 'Marathi': 'मला काहीच समजत नाहीये'},
    'Sorry': {'English': 'Sorry', 'Hindi': 'मुझे माफ़ कर दो', 'Marathi': 'मला माफ करून दे'}
}

# ==========================
# HEURISTIC LOGIC
# ==========================

def get_finger_states(hand_landmarks):
    """
    Returns a tuple of 5 booleans representing if (Thumb, Index, Middle, Ring, Pinky) are open.
    """
    fingers = []
    
    # Thumb: Check distance from tip vs ip to pinky mcp to abstract handedness
    thumb_tip_x = hand_landmarks.landmark[4].x
    thumb_ip_x = hand_landmarks.landmark[3].x
    pinky_mcp_x = hand_landmarks.landmark[17].x
    
    # If the tip is further from the pinky base than the inner joint, the thumb is "open"
    if abs(thumb_tip_x - pinky_mcp_x) > abs(thumb_ip_x - pinky_mcp_x):
        fingers.append(True)
    else:
        fingers.append(False)
        
    # Index, Middle, Ring, Pinky: tip y-coordinate vs pip y-coordinate
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]
    
    for tip, pip in zip(tips, pips):
        # Open if tip is physically above the pip (smaller y means higher in image coords)
        fingers.append(hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y)
        
    return tuple(fingers)

# ==========================
# AUDIO PLAYBACK (THREAD)
# ==========================

def play_audio_thread(text, lang_code):
    """
    Generates and plays TTS audio in a separate thread.
    """
    try:
        # Generate entirely unique filename to prevent access locking
        filename = f"audio_{int(time.time() * 1000)}.mp3"
        tts = gTTS(text=text, lang=lang_code)
        tts.save(filename)
        
        # Load and play
        pygame.mixer.music.load(filename)
        pygame.mixer.music.play()
        
        # Keep thread alive while playing
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)
            
        pygame.mixer.music.unload()
        
        # Delete file safely
        if os.path.exists(filename):
            os.remove(filename)
    except Exception as e:
        print(f"Audio Playback Error: {e}")

# ==========================
# STREAMLIT MAIN LOOP
# ==========================

def main():
    st.title("Live Multi-Lingual Sign Translation Dashboard (Rule-Based)")
    
    # Available physical cameras 
    available_cameras = []
    for i in range(4):
        cap_test = cv2.VideoCapture(i)
        if cap_test.read()[0]:
            available_cameras.append(i)
        cap_test.release()
    if not available_cameras:
        available_cameras = [0]
        
    # Sidebar
    st.sidebar.title("Settings")
    language = st.sidebar.selectbox("Select Language Output:", ["English", "Hindi", "Marathi"])
    camera_index = st.sidebar.selectbox("Select Camera Source:", available_cameras, index=0)
    run_webcam = st.sidebar.checkbox("Start Live Recognition", value=False)
    
    # Layout UI Elements
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Webcam Feed")
        frame_window = st.empty()
    
    with col2:
        st.markdown("### Translation")
        gesture_text_placeholder = st.empty()
        st.markdown("### Status")
        status_placeholder = st.empty()
        
    lang_codes = {"English": "en", "Hindi": "hi", "Marathi": "mr"}
    
    if run_webcam:
        # Initialize pure MediaPipe Tracking
        mp_hands = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        
        hands = mp_hands.Hands(
            static_image_mode=False, 
            max_num_hands=1,
            min_detection_confidence=0.7, 
            min_tracking_confidence=0.7
        )
        
        cap = cv2.VideoCapture(camera_index)
        
        # Hold state memory
        current_gesture = None
        gesture_start_time = 0.0
        last_spoken_gesture = None
        
        while run_webcam:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to access webcam.")
                break
                
            frame = cv2.flip(frame, 1) # Mirroring
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            results = hands.process(rgb_frame)
            detected_gesture_name = "None"
            
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                
                # Draw skeleton
                mp_drawing.draw_landmarks(
                    rgb_frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
                        
                # 1. Extract 5 Finger States
                finger_states = get_finger_states(hand_landmarks)
                
                # 2. Map combination to phrase
                detected_gesture_name = SIGNS_MAPPING.get(finger_states, "None")
            
            # 3. Time threshold and UI rendering
            if detected_gesture_name != "None":
                translated_text = TRANSLATIONS.get(detected_gesture_name, {}).get(language, detected_gesture_name)
                
                # Update UI immediately so user sees what is currently detected
                gesture_text_placeholder.markdown(f"<h1 style='text-align: center; color: #4CAF50;'>{translated_text}</h1>", unsafe_allow_html=True)
                
                if detected_gesture_name != current_gesture:
                    current_gesture = detected_gesture_name
                    gesture_start_time = time.time()
                    status_placeholder.info(f"Holding: {translated_text}...")
                else:
                    elapsed_time = time.time() - gesture_start_time
                    # 4. Critical 1.5 second hold requirement
                    if elapsed_time > 1.5:
                        if last_spoken_gesture != current_gesture:
                            status_placeholder.success(f"Spoken: {translated_text}")
                            
                            # Fire separate thread for TTS Audio
                            threading.Thread(
                                target=play_audio_thread, 
                                args=(translated_text, lang_codes[language]), 
                                daemon=True
                            ).start()
                            
                            last_spoken_gesture = current_gesture
            else:
                current_gesture = None
                gesture_start_time = None
                gesture_text_placeholder.markdown(f"<h1 style='text-align: center; color: gray;'>No Sign Detected</h1>", unsafe_allow_html=True)
                status_placeholder.empty()
            
            frame_window.image(rgb_frame, channels="RGB")
            
        cap.release()
        frame_window.empty()

if __name__ == "__main__":
    main()
