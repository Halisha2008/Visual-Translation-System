import cv2
import numpy as np
# import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from tensorflow.keras.models import load_model
from PIL import Image, ImageDraw, ImageFont
import arabic_reshaper
from bidi.algorithm import get_display

# 1. قائمة الكلمات ومعانيها العربية
actions_dict = {
    'Hello': 'مرحباً',
    'no': 'لا',
    'yes': 'نعم',
    'thank_you': 'شكراً',
    'help': 'أريد مساعدة',
    'how': 'كيف',
    'where': 'أين',
    'office': 'مكتب',
    'deputy_manager': 'معاون المدير'
}
actions = np.array(list(actions_dict.keys()))

# 2. تحميل النموذج المدرب
model = load_model('action_model.h5')

def put_arabic_text(img, text, position, font_size=32, color=(0, 255, 0)):
    reshaped_text = arabic_reshaper.reshape(text)
    bidi_text = get_display(reshaped_text)

    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()

    draw.text(position, bidi_text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def draw_hand_landmarks(frame, result):
    if result and result.hand_landmarks:
        for hand_landmarks in result.hand_landmarks:
            h, w, c = frame.shape
            points = []
            for lm in hand_landmarks:
                cx, cy = int(lm.x * w), int(lm.y * h)
                points.append((cx, cy))
                cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)

            connections = [
                (0, 1), (1, 2), (2, 3), (3, 4),
                (0, 5), (5, 6), (6, 7), (7, 8),
                (5, 9), (9, 10), (10, 11), (11, 12),
                (9, 13), (13, 14), (14, 15), (15, 16),
                (13, 17), (17, 18), (18, 19), (19, 20),
                (0, 17)
            ]
            for p1, p2 in connections:
                cv2.line(frame, points[p1], points[p2], (0, 255, 255), 2)
    return frame

base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    running_mode=vision.RunningMode.VIDEO
)
detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
timestamp_ms = 0
threshold = 0.65

is_recording_session = False
current_word_sequence = []
sentence_words = []
no_hand_counter = 0

# --- ضبط نافذة العرض لتفتح مباشرة بملء الشاشة ---
cv2.namedWindow('Visual Translation System', cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty('Visual Translation System', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

print("--- نظام الجملة المستمرة يعمل. اضغط (Space) لبدء التسجيل العام، افصل بين الكلمات بإنزال يدك قليلاً، ثم اضغط (Space) للإنهاء ---")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape  # أبعاد الإطار الحالية

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    result = detector.detect_for_video(image=mp_image, timestamp_ms=timestamp_ms)
    timestamp_ms += 33

    frame = draw_hand_landmarks(frame, result)

    frame_points = np.zeros(84)
    has_hand = False
    if result and result.hand_landmarks:
        has_hand = True
        temp_points = []
        for hand in result.hand_landmarks:
            wrist = hand[0]
            for lm in hand:
                temp_points.append(lm.x - wrist.x)
                temp_points.append(lm.y - wrist.y)
        if len(temp_points) <= 84:
            frame_points[:len(temp_points)] = temp_points[:len(temp_points)]
        else:
            frame_points = np.array(temp_points[:84])

    if is_recording_session:
        if has_hand:
            current_word_sequence.append(frame_points)
            no_hand_counter = 0
        else:
            if len(current_word_sequence) > 0:
                no_hand_counter += 1
                if no_hand_counter > 15:
                    if len(current_word_sequence) > 5:
                        if len(current_word_sequence) >= 30:
                            indices = np.linspace(0, len(current_word_sequence) - 1, 30).astype(int)
                            model_input = np.array([current_word_sequence[i] for i in indices])
                        else:
                            model_input = np.zeros((30, 84))
                            model_input[:len(current_word_sequence)] = current_word_sequence
                            for i in range(len(current_word_sequence), 30):
                                model_input[i] = current_word_sequence[-1]

                        res = model.predict(np.expand_dims(model_input, axis=0))[0]
                        predicted_index = np.argmax(res)
                        confidence = res[predicted_index]

                        if confidence > threshold:
                            action_key = actions[predicted_index]
                            translated_word = actions_dict[action_key]
                            sentence_words.append(translated_word)
                            print(f"--> تمت إضافة كلمة للجملة: {translated_word}")

                    current_word_sequence = []
                    no_hand_counter = 0

    full_sentence_text = ' '.join(sentence_words) if sentence_words else "لا توجد كلمات مسجلة بعد"

    # واجهة الشاشة (تم تعديل العرض ليكون w ليمتد على كامل عرض الشاشة)
    if is_recording_session:
        cv2.rectangle(frame, (0, 0), (w, 95), (0, 0, 255), -1)
        frame = put_arabic_text(frame, 'التسجيل العام مفتوح (أشر بيدك وافصل بين الكلمات)', (15, 12), 18, (255, 255, 255))
    else:
        cv2.rectangle(frame, (0, 0), (w, 95), (40, 40, 40), -1)
        frame = put_arabic_text(frame, '[Space]: بدء/إنهاء التسجيل | [c]: مسح | [q]: خروج', (15, 12), 18, (0, 255, 255))

    frame = put_arabic_text(frame, f'الجملة: {full_sentence_text}', (15, 52), 20, (255, 255, 0))

    cv2.imshow('Visual Translation System', frame)

    key = cv2.waitKey(10) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c') or key == ord('C'):
        sentence_words = []
        current_word_sequence = []
        is_recording_session = False
        print("--> تم مسح الجملة وإعادة الضبط.")
    elif key == ord(' '):
        if not is_recording_session:
            is_recording_session = True
            sentence_words = []
            current_word_sequence = []
            print("--> 🟢 بدأ تسجيل الجلسة العامة. أشر بالكلمات وافصل بينها برفع يدك قليلاً.")
        else:
            is_recording_session = False
            if len(current_word_sequence) > 5:
                if len(current_word_sequence) >= 30:
                    indices = np.linspace(0, len(current_word_sequence) - 1, 30).astype(int)
                    model_input = np.array([current_word_sequence[i] for i in indices])
                else:
                    model_input = np.zeros((30, 84))
                    model_input[:len(current_word_sequence)] = current_word_sequence
                    for i in range(len(current_word_sequence), 30):
                        model_input[i] = current_word_sequence[-1]

                res = model.predict(np.expand_dims(model_input, axis=0))[0]
                predicted_index = np.argmax(res)
                confidence = res[predicted_index]
                if confidence > threshold:
                    sentence_words.append(actions_dict[actions[predicted_index]])

            print(f"--> 🔴 انتهت الجلسة. الجملة النهائية: {' '.join(sentence_words)}")
            current_word_sequence = []

cap.release()
cv2.destroyAllWindows()