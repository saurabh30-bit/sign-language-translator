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
    Uses Dynamic Distance Normalization (scale and rotation invariant).
    """
    import math
    
    # Landmark 0: Wrist, Landmark 9: Middle Finger MCP (Base)
    wrist = hand_landmarks.landmark[0]
    middle_mcp = hand_landmarks.landmark[9]
    
    # Calculate baseline unit (1.0) for the user's specific hand size and camera distance
    base_dist = math.hypot(wrist.x - middle_mcp.x, wrist.y - middle_mcp.y)
    if base_dist == 0:
        base_dist = 0.001 # Prevent division by zero
        
    fingers = []
    
    # 1. Thumb (Tip: 4, Pinky MCP Base: 17)
    # The thumb swings outward horizontally when open, so the distance from its tip 
    # to the opposite side of the hand (pinky base) must exceed the baseline substantially.
    thumb_tip = hand_landmarks.landmark[4]
    pinky_mcp = hand_landmarks.landmark[17]
    thumb_dist = math.hypot(thumb_tip.x - pinky_mcp.x, thumb_tip.y - pinky_mcp.y)
    
    # Open thumb is generally > 1.2x the base palm distance
    fingers.append(thumb_dist / base_dist > 1.2)
        
    # 2. Four Fingers (Tips: 8, 12, 16, 20)
    # Fully extended fingers are structurally far from the wrist.
    # Closed knuckles curl into the palm, moving the tip significantly closer to the wrist.
    tips = [8, 12, 16, 20]
    for tip_idx in tips:
        tip = hand_landmarks.landmark[tip_idx]
        dist = math.hypot(tip.x - wrist.x, tip.y - wrist.y)
        
        # Extended finger tip > 1.5x the palm unit distance from the wrist
        fingers.append(dist / base_dist > 1.5)
        
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
        
    # Session state for custom signs
    if 'custom_signs' not in st.session_state:
        st.session_state['custom_signs'] = {}
    if 'last_taught_phrase' not in st.session_state:
        st.session_state['last_taught_phrase'] = ""

    # Sidebar
    st.sidebar.title("Settings")
    recognition_mode = st.sidebar.radio("Select Recognition Mode:", ["Conversational Phrases", "Teach the AI"])
    
    teaching_phrase = ""
    if recognition_mode == "Teach the AI":
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 🧠 Teach a New Sign")
        st.sidebar.write(f"**Custom memory:** {len(st.session_state['custom_signs'])} signs")
        teaching_phrase = st.sidebar.text_input("1. Type phrase & hit Enter\n2. Hold gesture 2 seconds to bind:", key="new_phrase").strip()
        if st.sidebar.button("Clear Memory"):
            st.session_state['custom_signs'] = {}
            st.rerun()

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
        mp_face_mesh = mp.solutions.face_mesh
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        
        hands = mp_hands.Hands(
            static_image_mode=False, 
            max_num_hands=2,
            min_detection_confidence=0.7, 
            min_tracking_confidence=0.7
        )
        face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        cap = cv2.VideoCapture(camera_index)
        
        # Hold state memory
        current_gesture = None
        gesture_start_time = 0.0
        last_spoken_gesture = None
        
        # UI state cache to prevent Streamlit WebSocket lag
        last_rendered_gesture_html = ""
        last_rendered_status_text = ""
        
        while run_webcam:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to access webcam.")
                break
                
            frame = cv2.flip(frame, 1) # Mirroring
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            results = hands.process(rgb_frame)
            face_results = face_mesh.process(rgb_frame)
            
            detected_gesture_name = "None"
            emotion_modifier = ""
            
            # --- Emotion Tagging ---
            if face_results.multi_face_landmarks:
                face_landmarks = face_results.multi_face_landmarks[0]
                
                # Face width for normalization: 234 (Left Edge) to 454 (Right Edge)
                face_left_x = face_landmarks.landmark[234].x
                face_right_x = face_landmarks.landmark[454].x
                face_width = abs(face_right_x - face_left_x)
                
                if face_width > 0:
                    # Lip corners: 61 (Left), 291 (Right)
                    left_x = face_landmarks.landmark[61].x
                    left_y = face_landmarks.landmark[61].y
                    right_x = face_landmarks.landmark[291].x
                    right_y = face_landmarks.landmark[291].y
                    
                    mouth_width = ((right_x - left_x)**2 + (right_y - left_y)**2)**0.5
                    mouth_ratio = mouth_width / face_width
                    
                    if mouth_ratio > 0.44:
                        emotion_modifier = " (Polite)"
                    elif mouth_ratio < 0.35:
                        emotion_modifier = " (Urgent)"
            # -----------------------
            
            if results.multi_hand_landmarks:
                # Setup modifier logic
                modifier_active = False
                primary_hand = results.multi_hand_landmarks[0]
                
                # Check for 2nd hand acting as a Shift/Modifier (Closed Fist)
                if len(results.multi_hand_landmarks) == 2:
                    hand1 = results.multi_hand_landmarks[0]
                    hand2 = results.multi_hand_landmarks[1]
                    
                    states1 = get_finger_states(hand1)
                    states2 = get_finger_states(hand2)
                    
                    closed_fist = (False, False, False, False, False)
                    if states1 == closed_fist and states2 != closed_fist:
                        modifier_active = True
                        primary_hand = hand2
                    elif states2 == closed_fist and states1 != closed_fist:
                        modifier_active = True
                        primary_hand = hand1

                # Draw skeleton for all detected hands
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        rgb_frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )
                        
                # 1. Extract 5 Finger States for the active primary hand
                finger_states = get_finger_states(primary_hand)
                
                # 2. Map combination to phrase
                detected_gesture_name = "None"
                is_actively_teaching = False
                
                if recognition_mode == "Conversational Phrases":
                    detected_gesture_name = SIGNS_MAPPING.get(finger_states, "None")
                    
                    # Apply "Shift Key" Modifier Logic
                    if modifier_active and detected_gesture_name != "None":
                        MODIFIER_DICTIONARY = {
                            'Hello': 'Goodbye',
                            'Yes': 'No',
                            'Need Help': 'I am Fine',
                            'Good Morning': 'Good Night',
                            'Pay Attention': 'Ignore This',
                            'Time': 'No Time',
                            'Together': 'Alone',
                            'Peace': 'War'
                        }
                        detected_gesture_name = MODIFIER_DICTIONARY.get(detected_gesture_name, f"Reverse of {detected_gesture_name}")
                else:
                    # Teach the AI Mode
                    if teaching_phrase and teaching_phrase != st.session_state['last_taught_phrase']:
                        detected_gesture_name = teaching_phrase
                        is_actively_teaching = True
                    else:
                        detected_gesture_name = st.session_state['custom_signs'].get(finger_states, "None")
            
            # 3. Time threshold and UI rendering
            if detected_gesture_name != "None":
                if recognition_mode == "Conversational Phrases":
                    base_text = str(TRANSLATIONS.get(detected_gesture_name, {}).get(language, detected_gesture_name))
                else:
                    base_text = str(detected_gesture_name)
                    
                # Apply Emotion Modifiers (Tone and Inflection)
                tone_color = "#4CAF50" # Default Green
                tts_text = base_text
                
                if emotion_modifier == " (Polite)":
                    tone_color = "#2196F3" # Friendly Blue
                    tts_text = base_text + " please!"
                elif emotion_modifier == " (Urgent)":
                    tone_color = "#F44336" # Urgent Red
                    tts_text = base_text + "!!!"
                
                # Update UI only if changed (prevents massive Streamlit WebSocket lag)
                if is_actively_teaching:
                    new_html = f"<h1 style='text-align: center; color: #FFA500;'>Learning: '{base_text}'...</h1>"
                else:
                    new_html = f"<h1 style='text-align: center; color: {tone_color};'>{base_text}</h1>"
                    
                if new_html != last_rendered_gesture_html:
                    gesture_text_placeholder.markdown(new_html, unsafe_allow_html=True)
                    last_rendered_gesture_html = new_html
                
                if detected_gesture_name != current_gesture:
                    current_gesture = detected_gesture_name
                    gesture_start_time = time.time()
                    
                    new_status = f"Hold shape steady to lock in '{base_text}'..." if is_actively_teaching else f"Holding: {base_text}..."
                    if new_status != last_rendered_status_text:
                        status_placeholder.info(new_status)
                        last_rendered_status_text = new_status
                else:
                    elapsed_time = time.time() - gesture_start_time
                    
                    if is_actively_teaching:
                        if elapsed_time > 2.0:
                            st.session_state['custom_signs'][finger_states] = base_text
                            st.session_state['last_taught_phrase'] = base_text
                            
                            new_status = f"🎉 SUCCESS! Bound hand shape to '{base_text}'"
                            if new_status != last_rendered_status_text:
                                status_placeholder.success(new_status)
                                last_rendered_status_text = new_status
                    else:
                        if elapsed_time > 1.5:
                            if last_spoken_gesture != current_gesture:
                                # Fire separate thread for TTS Audio with tone inflection
                                threading.Thread(
                                    target=play_audio_thread, 
                                    args=(tts_text, lang_codes[language]), 
                                    daemon=True
                                ).start()
                                
                                new_status = f"Spoken: {base_text}"
                                if new_status != last_rendered_status_text:
                                    status_placeholder.success(new_status)
                                    last_rendered_status_text = new_status
                                
                                last_spoken_gesture = current_gesture
            else:
                current_gesture = None
                gesture_start_time = None
                last_spoken_gesture = None  # Reset TTS memory so the same sign can be spoken again later!
                
                new_html = f"<h1 style='text-align: center; color: gray;'>No Sign Detected</h1>"
                if new_html != last_rendered_gesture_html:
                    gesture_text_placeholder.markdown(new_html, unsafe_allow_html=True)
                    last_rendered_gesture_html = new_html
                    
                if last_rendered_status_text != "":
                    status_placeholder.empty()
                    last_rendered_status_text = ""
            
            frame_window.image(rgb_frame, channels="RGB")
            
        cap.release()
        frame_window.empty()

if __name__ == "__main__":
    main()
