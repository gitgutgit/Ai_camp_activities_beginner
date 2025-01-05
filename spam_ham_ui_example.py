# streamlit_app.py

import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

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
st.write("이 앱은 간단한 스팸/햄 분류 모델을 보여줍니다. 학생들이 부모님께 모델을 설명하고, 테스트 결과를 직접 확인할 수 있습니다.")

# 사용자 입력
st.header("새로운 문장 테스트하기")
new_texts = [
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
answers = [
    "spam", "ham", "spam", "ham", "spam", "spam", "ham", "ham", "ham", "spam",
]

# 예측
new_texts_vec = vectorizer.transform(new_texts)
predictions = model.predict(new_texts_vec)

# 결과 테이블
st.subheader("결과 보기")
results = pd.DataFrame({
    "문장": new_texts,
    "예측 결과": predictions,
    "정답": answers
})

# 테이블 출력
st.table(results)

# 정확도 계산 및 출력
accuracy = sum([1 if p == a else 0 for p, a in zip(predictions, answers)]) / len(answers) * 100
st.write(f"**Accuracy on new texts: {accuracy:.2f}%**")

# 입력 기능: 새로운 문장 테스트
st.subheader("직접 문장 입력하여 테스트하기")
user_input = st.text_area("테스트할 문장을 입력하세요:", "")
if st.button("결과 확인"):
    user_vec = vectorizer.transform([user_input])
    user_prediction = model.predict(user_vec)
    st.write(f"입력한 문장: **'{user_input}'**")
    st.write(f"분류 결과: **{user_prediction[0]}**")
