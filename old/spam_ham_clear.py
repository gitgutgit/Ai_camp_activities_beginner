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

# (1) 튜플 리스트로 데이터 정의
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
    # 활동1. 직접 spam/ ham 구분하기

    #("축하합니다! 당신은 유럽행 비행기를 무료로 탈 기회를 얻으셨습니다. 지금 바로 접속하세요!", "??")
    # ("Let's grab lunch this afternoon. See you at 1pm!", "??")
    # ("당신의 신용카드 한도가 초과되었습니다. 지금 확인하세요!", "??")
    # ("Hi mom, I'll call you later. Love you!", "??")
    # ("지금 가입하면 최대 50% 할인 혜택을 드립니다!", "??")
    # ("How was your day? Did you finish the assignment?", "??")

    # 활동 2. 직접 train 데이터를 더 작성해보기

]

# (2) 데이터프레임 생성
df = pd.DataFrame(data, columns=["text", "label"])

# (3) 학습/테스트 세트로 분리
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"],
    test_size=0.2, 
    random_state=42
)

# (4) TF-IDF 벡터라이저로 텍스트 -> 숫자 벡터 변환
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# (5) 간단한 모델(나이브베이즈)로 학습
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# (6) 예측 및 평가
y_pred = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# (7) 새 문장 분류 테스트
new_texts = [
    # Spam-like 메시지
    "Free iPhone giveaway! Hurry up, click the link!",
    "Congratulations! You've been selected to win a $1,000 gift card!",
    "WIN a vacation to Hawaii! Limited offer, click now!",
    "지금 구매하면 특별 할인 혜택을 드립니다! 서두르세요!",
    "당신의 계정이 해킹되었습니다. 지금 복구하세요!",
    "Special promotion! Buy one, get one free! Don't miss out!",
    "Limited-time deal: Save 50% on your next purchase!",
    "무료 쿠폰을 지금 바로 받으세요!",
    "신용카드 한도가 초과되었습니다. 지금 확인하세요!",

    # Ham-like 메시지
    "Let's go to a movie this weekend.",
    "Hi jacob, see u at 4pm!",
    "오늘 날씨 좋으면 쇼핑가자",
    "Hey, how was your day? Let's catch up soon.",
    "Don't forget to bring the report to the meeting tomorrow.",
    "오늘 저녁 같이 밥 먹으러 갈래?",
    "Can you send me the file by 5 pm?",
    "The meeting has been rescheduled to 3 pm. Let me know if you’re available.",
    "오늘 운동 끝나고 커피 한 잔 어때?",
    "I'll see you at the party later tonight!",
    "What time should I pick you up tomorrow morning?",
    "Have a great weekend! Let’s meet up next week.",
    "Did you finish the homework? Can you share your notes?",
]


new_texts_vec = vectorizer.transform(new_texts)
predictions = model.predict(new_texts_vec)

print("\n-- Prediction on new texts --")
for txt, pred in zip(new_texts, predictions):
    print(f"'{txt}' => {pred}")
