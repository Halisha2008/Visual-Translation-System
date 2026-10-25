import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# 1. قائمة الكلمات بأسماء مجلداتها ومعانيها العربية
actions_dict = {
    'Hello': 'مرحباً',
    'no': 'لا',
    'yes': 'نعم',
    'thank_you': 'شكراً',
    'help': 'مساعدة',
    'how': 'كيف',
    'where': 'أين',
    'office': 'مكتب',
    'deputy_manager': 'معاون المدير'
}

actions = np.array(list(actions_dict.keys()))

no_sequences = 60       # عدد الفيديوهات لكل كلمة
sequence_length = 30    # عدد الإطارات في كل فيديو
DATA_PATH = os.path.join('Sign_Video_Data')

# 2. تجهيز البيانات والـ Labels مع طباعة حالة التحميل خطوة بخطوة
print("--- [1/4] جاري تحميل قراءة البيانات من المجلدات ---")
label_map = {label: num for num, label in enumerate(actions)}

sequences, labels = [], []
for action in actions:
    print(f"جاري قراءة بيانات الكلمة: {action} ({actions_dict[action]})")
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            try:
                res = np.load(os.path.join(DATA_PATH, action, str(sequence), f"{frame_num}.npy"))
                window.append(res)
            except Exception as e:
                window.append(np.zeros(84))
        sequences.append(window)
        labels.append(label_map[action])

print("--- [2/4] تحويل البيانات إلى مصفوفات NumPy ---")
X = np.array(sequences)
y = to_categorical(labels).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=44)
print(f"حجم بيانات التدريب (X_train shape): {X_train.shape}")

# 3. بناء هيكل نموذج الـ LSTM
print("--- [3/4] بناء نموذج الـ LSTM ---")
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 84)))
model.add(Dropout(0.2))
model.add(LSTM(128, return_sequences=False, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# 4. بدء التدريب الفعلي
print("--- [4/4] جاري بدء التدريب (Epochs)... ستظهر النتائج الآن تباعاً ---")
model.fit(X_train, y_train, epochs=120, validation_data=(X_test, y_test))

# 5. حفظ الموديل الجديد
model.save('action_model.h5')
print("--- تم تدريب النموذج وحفظه بنجاح باسم action_model.h5 ---")