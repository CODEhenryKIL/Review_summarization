import os
import json
import sys
import datetime

from dateutil import parser
from openai import OpenAI

# openai api key 세팅
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
client = OpenAI(api_key=OPENAI_API_KEY)

'''
# openai api key test
compile = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Hi"},],
    temperature=0.0
)
print(compile.choices[0].message.content)
'''
# 리뷰 JSON 파일 로드
review_name = sys.argv[1]
with open(f'./res/{review_name}.json', 'r', encoding='utf-8') as f:
    review_list = json.load(f)

# 전처리 함수
def preprocess_review(review_list):
    
    # 사용할 리스트 선언 및 6개월 이전 날짜 구하기
    good_reviews, bad_reviews = [], []
    current_date = datetime.datetime.now()
    date_boundary = current_date - datetime.timedelta(days=6*30)

    for r in review_list:
        reviews_date_str = r['date']
        try:
            reviews_date = parser.parse(reviews_date_str)
        # 날짜 형식이 다른 경우 현재 날짜로 설정 -> 최근 리뷰의 경우 OO일 전으로 표시됨
        except(ValueError, TypeError):
            reviews_date = current_date
        
        # 6개월 이상 지난 리뷰는 제외
        if reviews_date < date_boundary:
            continue
        
        # 별점이 5점인 리뷰와 5점이 아닌 리뷰로 분리
        if r['stars'] == 5:
            good_reviews.append('[리뷰 시작]' + r['review'] + '[리뷰 끝]')
        else:
            bad_reviews.append('[리뷰 시작]' + r['review'] + '[리뷰 끝]')
    
    # GPT prompt에 맞게 리뷰 리스트를 문자열로 변환
    good_reviews_str = '\n'.join(good_reviews)
    bad_reviews_str = '\n'.join(bad_reviews)

    return good_reviews_str, bad_reviews_str

# 평가 스크립트 - MT-Bench 논문 참고
def eval(reviews, answer_a, answer_b):
    eval_prompt = f"""[System]
Please act as an impartial judge and evaluate the quality of the Korean summaries provided by two
AI assistants to the set of user reviews on accommodations displayed below. You should choose the assistant that
follows the user’s instructions and answers the user’s question better. Your evaluation
should consider factors such as the helpfulness, relevance, accuracy, depth, creativity,
and level of detail of their responses. Begin your evaluation by comparing the two
responses and provide a short explanation. Avoid any position biases and ensure that the
order in which the responses were presented does not influence your decision. Do not allow
the length of the responses to influence your evaluation. Do not favor certain names of
the assistants. Be as objective as possible. After providing your explanation, output your
final verdict by strictly following this format: "[[A]]" if assistant A is better, "[[B]]"
if assistant B is better, and "[[C]]" for a tie.
[User Reviews] 
{reviews}
[The Start of Assistant A’s Answer]
{answer_a}
[The End of Assistant A’s Answer]
[The Start of Assistant B’s Answer]
{answer_b}
[The End of Assistant B’s Answer]"""
    
    completion = client.chat.completions.create(
        model='gpt-4o-mini',
        messages=[{'role': 'user', 'content': eval_prompt}],
        temperature=0.0
    )

    return completion.choices[0].message.content()

# Baseline 모델
prompt_base = f"""아래 숙소 리뷰에 대해 5문장 내로 요약해줘:"""

def baseline_model(reviews, prompt_base, temperature=0.0, model='gpt-3.5-turbo'):
    # 베이스 프롬프트 + 리뷰를 합침
    prompt = prompt + '\n\n' + reviews

    #GPT로 리뷰 요약
    completion = client.chat.completions.create(
        model=model,
        messages=[{'role': 'user', 'content': prompt}],
        temperature=temperature
    )
    return completion.choices[0].message.content()

# 1차 테스트 - 단일 평가
good_reviews, bad_reviews = preprocess_review(review_list)
baseline_good = baseline_model(good_reviews, prompt_base)
test_result = baseline_model(good_reviews, prompt_base, model='gpt-4o-mini')
#print(eval(good_reviews,baseline_good,test_result))

# 2차 테스트 - 대규모 평가



