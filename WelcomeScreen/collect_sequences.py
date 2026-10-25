import cv2
import numpy as np
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PIL import Image, ImageDraw, ImageFont
import arabic_reshaper
from bidi.algorithm import get_display

# نحدد الكلمة المستهدفة فقط للتعديل والتسجيل
target_word = 'office'
arabic_word_name = 'مكتب'

no_sequences = 80  # عدد الفيديوهات الجديدة
sequence_length = 45  # عدد الإطارات لكل فيديو
DATA_PATH = os.path.join('Data')  # أو مسار مجلد كلماتك المفردة القديم

# التأكد من إنشاء المجلد الخاص بالكلمة
for sequence in range(no_sequences):
    try:
        os.makedirs(os.path.join(DATA_PATH, target_word, str(sequence)))
    except:
        pass


def put_arabic_text(img, text, position, font_size=30, color=(0, 255, 0)):
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
                (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
                (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
                (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)
            ]
            for p1, p2 in connections:
                cv2.line(frame, points[p1], points[p2], (0, 255, 255), 2)
    return frame


base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")
options = vision.HandLandmarkerOptions(
    base_options=base_options, num_hands=2, running_mode=vision.RunningMode.VIDEO
)
detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
timestamp_ms = 0

print(f"--- بدء إعادة تسجيل وتحديث بيانات كلمة: {arabic_word_name} ---")

for sequence in range(no_sequences):
    # مرحلة الانتظار حتى تستعد للكلمة
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = detector.detect_for_video(image=mp_image, timestamp_ms=timestamp_ms)
        timestamp_ms += 33
        frame = draw_hand_landmarks(frame, result)

        cv2.rectangle(frame, (0, 0), (640, 95), (40, 40, 40), -1)
        frame = put_arabic_text(frame, f'إعادة تسجيل كلمة: {arabic_word_name}', (15, 10), 22, (255, 255, 255))
        frame = put_arabic_text(frame, f'التسجيل رقم: {sequence + 1} / {no_sequences}', (15, 45), 20, (0, 255, 255))
        frame = put_arabic_text(frame, 'اضغط (r) للبدء | (q) للخروج', (15, 75), 18, (255, 255, 0))

        cv2.imshow('Collect Single Word - Office', frame)
        key = cv2.waitKey(10) & 0xFF
        if key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            exit()
        elif key == ord('r'):
            break

    # تسجيل الـ 30 إطاراً للكلمة
    for frame_num in range(sequence_length):
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = detector.detect_for_video(image=mp_image, timestamp_ms=timestamp_ms)
        timestamp_ms += 33
        frame = draw_hand_landmarks(frame, result)

        frame_points = np.zeros(84)
        if result and result.hand_landmarks:
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

        npy_path = os.path.join(DATA_PATH, target_word, str(sequence), f"{frame_num}.npy")
        np.save(npy_path, frame_points)

        cv2.rectangle(frame, (0, 0), (640, 70), (0, 0, 255), -1)
        frame = put_arabic_text(frame, f'جاري تسجيل "مكتب" ({frame_num + 1}/30)', (15, 20), 22, (255, 255, 255))
        cv2.imshow('Collect Single Word - Office', frame)
        cv2.waitKey(10)

print("--- تم الانتهاء من إعادة تسجيل وتحديث بيانات كلمة مكتب بنجاح! ---")
cap.release()
cv2.destroyAllWindows()