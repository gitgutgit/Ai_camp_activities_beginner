import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# 학습용 데이터 (Train Data)
# 틀리게 분류되어있는것도 있으니 고치세요!, 주석으로 된걸 완성해보세요
train_data = [
    ("무료 쿠폰이 준비되었습니다. 지금 클릭하세요!", "spam"),
    ("오늘 같이 저녁 먹으러 갈래요?", "spam"),
    ("지금 바로 구매 시 50% 할인 혜택을 드립니다!", "spam"),
    ("내일 모임에 참석 가능하신가요?", "ham"),
    ("한정 이벤트! 지금 신청하면 특별한 선물을 드립니다!", "spam"),
  #  ("지금 가입하면 무료 포인트를 제공합니다!", ""),
    ("안녕하세요. 오랜만에 연락드려요. 잘 지내시죠?", "ham"),
    ("보고서를 오늘까지 제출 부탁드립니다.", "spam"),
    #("축하합니다! 당첨되셨습니다. 지금 바로 확인하세요!", ""),
    ("오늘 날씨가 너무 좋네요. 산책하러 갈래요?", "ham"),
]

# 테스트용 데이터 (Test Data)
test_data = [
    ("특별 할인 혜택! 지금 구매하세요!", "spam"),
    ("오늘 영화 보러 갈래요?", "ham"),
    ("축하합니다! 무료 여행에 당첨되셨습니다!", "spam"),
    ("다음 회의는 오후 3시입니다.", "ham"),
    ("긴급: 계정을 복구하려면 지금 클릭하세요.", "spam"),
    ("좋은 아침이에요! 오늘 하루도 화이팅!", "ham"),
    ("오늘 만 있는 기회입니다, 당장 구매하세요", "spam"),
    ("축하해 로또에 당첨되었다며! 꼭 밥이라도 사라", "ham"),
]

# 학습 데이터프레임 생성
train_df = pd.DataFrame(train_data, columns=["text", "label"])

# 테스트 데이터프레임 생성
test_df = pd.DataFrame(test_data, columns=["text", "label"])

# TF-IDF 벡터라이저 및 모델 생성
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(train_df["text"])
model = MultinomialNB()
model.fit(X_train_vec, train_df["label"])

# 테스트 데이터 예측 및 정확도 계산
X_test_vec = vectorizer.transform(test_df["text"])
test_predictions = model.predict(X_test_vec)
accuracy = (test_predictions == test_df["label"]).mean() * 100

# Streamlit UI
st.title("스팸/햄 분류기")
st.write("AI 모델을 사용해 한글 문장을 스팸 또는 햄으로 분류합니다.")

# 학습 데이터 표시
st.subheader("학습 데이터 (Train Data)")
st.write("모델이 학습한 데이터를 확인하세요.")
st.table(train_df)

# 테스트 결과 표시
st.subheader("테스트 결과 (Test Data)")
test_results = pd.DataFrame({
    "문장": test_df["text"],
    "AI예측": test_predictions,
    "실제정답": test_df["label"],
})
st.table(test_results)

# 정확도 표시
st.subheader("모델 성능")
st.write(f"**테스트 데이터 정확도: {accuracy:.2f}%**")

# 데이터 입력 및 테스트
st.subheader("직접 문장 입력하여 테스트하기")
user_input = st.text_area("테스트할 문장을 입력하세요:")
if st.button("결과 확인"):
    if user_input.strip():
        user_vec = vectorizer.transform([user_input])
        prediction = model.predict(user_vec)
        st.write(f"입력한 문장: **'{user_input}'**")
        st.write(f"분류 결과: **{prediction[0]}**")
    else:
        st.warning("문장을 입력하세요.")

# 음성 입력 (Extra 활동)
st.subheader("음성으로 입력하여 테스트하기")
st.write("음성 인식 기능은 Extra 활동에서 구현해 보세요.")
