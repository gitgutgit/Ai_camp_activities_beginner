"""
예시: 간단한 스팸/햄 분류 모델 (MultinomialNB + TfidfVectorizer)

- 데이터는 예시를 위해 직접 만든 작은 문장 리스트를 사용합니다.
- 실제로는 SMS Spam Collection 등 공개된 데이터셋을 사용하면 더 풍부한 테스트가 가능합니다.
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# (1) 예시용 데이터프레임 생성
data = {
    'text': [
        "I had a great day today!",
        "Oh no, this is the worst thing ever.",
        "Congratulations! You have won a free trip to Bahamas.",
        "Hmm, the weather is quite nice.",
        "WIN a brand new car by clicking here!",
        "I love spending time with my family.",
        "Limited offer! Click now to get a special discount!",
        "It's really fun to watch movies together.",
        "오늘 같이 놀러갈래? 날씨가 너무 좋아",
        "노트북을 당장 바꾸세요, 지금 구매하세요!"
    ],
    'label': [
        "ham",       # 그냥 일반 메시지
        "ham",
        "spam",      # 스팸 광고
        "ham",
        "spam",
        "ham",
        "spam",
        "ham",
        "ham",
        "spam",
        
    ]
}

df = pd.DataFrame(data)

# (2) 학습/테스트 세트로 분리
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'],
    test_size=0.2, 
    random_state=42
)

# (3) TF-IDF 벡터라이저로 텍스트 -> 숫자 벡터 변환
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# (4) 간단한 모델(나이브베이즈)로 학습
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# (5) 예측 및 평가
y_pred = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# (6) 새 문장 분류 테스트
new_texts = [
    "Free iPhone giveaway! Hurry up, click the link!",
    "Let's go to a movie this weekend.",
    "Hi jacob, see u at 4pm!",
    "오늘 날씨 좋으면 쇼핑가자",
    "당신의 노트북을 바꿀 시간입니다, 지금 바로 구매하세요!"
]


new_texts_vec = vectorizer.transform(new_texts)
predictions = model.predict(new_texts_vec)

print("\n-- Prediction on new texts --")
for txt, pred in zip(new_texts, predictions):
    print(f"'{txt}' => {pred}")
