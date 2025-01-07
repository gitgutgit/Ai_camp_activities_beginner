import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import speech_recognition as sr
import sounddevice as sd
from scipy.io.wavfile import write

# (1) 데이터 정의
data = [
    ("I had a great day today!", "ham"),
    ("Oh no, this is the worst thing ever.", "ham"),
    ("Congratulations! You have won a free trip to Bahamas.", "spam"),
    ("Hmm, the weather is quite nice.", "ham"),
    ("WIN a brand new car by clicking here!", "spam"),
    ("I love spending time with my family.", "ham"),
    ("Limited offer! Click now to get a special discount!", "spam"),
    ("It's really fun to watch movies together.", "ham"),
    ("오늘 같이 놀러갈래? 날씨가 너무 좋아", "ham"),
    ("노트북을 당장 바꾸세요, 지금 구매하세요!", "spam"),
]

# (2) 데이터프레임 생성
df = pd.DataFrame(data, columns=["text", "label"])

# (3) 학습/테스트 세트로 분리
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"],
    test_size=0.2, 
    random_state=42
)

# (4) TF-IDF 벡터라이저 및 모델 생성
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Streamlit UI 시작
st.title("Spam vs Ham Classifier")
st.write("이 앱은 간단한 스팸/햄 분류 모델을 보여줍니다.")

# (5) 테스트 문장과 정확도 계산
st.header("테스트 문장 분류 및 정확도")
test_texts = [
    "Free iPhone giveaway! Hurry up, click the link!",
    "Let's go to a movie this weekend.",
    "Congratulations! You've been selected to win a $1,000 gift card!",
    "Hi jacob, see u at 4pm!",
    "WIN a vacation to Hawaii! Limited offer, click now!",
    "지금 구매하면 특별 할인 혜택을 드립니다! 서두르세요!",
    "Hey, how was your day? Let's catch up soon.",
    "오늘 날씨 좋으면 쇼핑가자",
    "Don't forget to bring the report to the meeting tomorrow.",
    "당신의 계정이 해킹되었습니다. 지금 복구하세요!",
]
test_answers = [
    "spam", "ham", "spam", "ham", "spam", "spam", "ham", "ham", "ham", "spam",
]

# 예측
test_texts_vec = vectorizer.transform(test_texts)
test_predictions = model.predict(test_texts_vec)

# 결과 테이블 출력
results = pd.DataFrame({
    "문장": test_texts,
    "예측 결과": test_predictions,
    "정답": test_answers
})
st.subheader("테스트 결과")
st.table(results)

# 정확도 계산
accuracy = sum([1 if p == a else 0 for p, a in zip(test_predictions, test_answers)]) / len(test_answers) * 100
st.write(f"**테스트 데이터 정확도: {accuracy:.2f}%**")

# (6) 사용자 입력: 직접 문장 입력
st.subheader("직접 문장 입력하여 테스트하기")
user_input = st.text_area("테스트할 문장을 입력하세요:", "")
if st.button("결과 확인"):
    if user_input.strip():
        user_vec = vectorizer.transform([user_input])
        user_prediction = model.predict(user_vec)
        st.write(f"입력한 문장: **'{user_input}'**")
        st.write(f"분류 결과: **{user_prediction[0]}**")
    else:
        st.warning("문장을 입력하거나 음성을 녹음하세요.")

# (7) 음성 입력 기능
st.subheader("음성으로 입력하여 테스트하기")

def record_audio(duration=5, fs=44100):
    st.write("녹음 중입니다...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    write("output.wav", fs, recording)
    st.success("녹음 완료!")
    return "output.wav"

if st.button("음성 입력"):
    file_path = record_audio()
    recognizer = sr.Recognizer()
    with sr.AudioFile(file_path) as source:
        audio = recognizer.record(source)
        try:
            speech_text = recognizer.recognize_google(audio, language="ko")
            st.write(f"인식된 텍스트: **'{speech_text}'**")
            user_vec = vectorizer.transform([speech_text])
            user_prediction = model.predict(user_vec)
            st.write(f"분류 결과: **{user_prediction[0]}**")
        except sr.UnknownValueError:
            st.error("음성을 이해하지 못했습니다.")
        except sr.RequestError:
            st.error("음성 인식 서비스에 문제가 발생했습니다.")
