import os
import json
import sys
import datetime

from dateutil import parser
from openai import OpenAI
from tqdm import tqdm

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

    return completion.choices[0].message.content

# Baseline 모델
prompt_base = f"""아래 숙소 리뷰에 대해 5문장 내로 요약해줘:"""

def baseline_model(reviews, prompt_base, temperature=0.0, model='gpt-3.5-turbo'):
    # 베이스 프롬프트 + 리뷰를 합침
    prompt = prompt_base + '\n\n' + reviews

    #GPT로 리뷰 요약
    completion = client.chat.completions.create(
        model=model,
        messages=[{'role': 'user', 'content': prompt}],
        temperature=temperature
    )
    return completion.choices[0].message.content

# 1차 테스트 - 단일 평가
good_reviews, bad_reviews = preprocess_review(review_list)
baseline_good = baseline_model(good_reviews, prompt_base)
test_result = baseline_model(good_reviews, prompt_base, model='gpt-4o-mini')
#print(eval(good_reviews,baseline_good,test_result))

# 2차 테스트 - 대규모 평가
# eval_count만큼 baseline 모델을 생성
eval_count = 10
baseline = [baseline_model(good_reviews, prompt_base, temperature = 1.0) for _ in range(eval_count)]
gpt_o1_summary = '숲속에 위치한 숙소로 자연 속에서 힐링하기에 최적입니다. 객실 상태가 깔끔하고 침구류가 편안해 쾌적한 숙박을 할 수 있습니다. 스파와 수영장 등 다양한 부대시설이 마련되어 있어 가족 단위뿐만 아니라 연인이나 친구들과 함께 즐기기 좋습니다. 직원들이 매우 친절하고 식당들도 품질이 높아 전반적으로 투숙객들에게 긍정적인 경험을 선사합니다. 사계절 내내 아름다운 풍경을 감상할 수 있어 재방문 의사가 높게 나타나는 인기 숙소로 평가됩니다.'

# 생성된 baseline 모델과 gpt_o1_summary를 비교하여 성능을 평가하는 함수
def eval_batch(reviews, baseline, gpt_o1_summary):
    base_count, o1_count, draw_count = 0, 0, 0
    for i in tqdm(range(len(baseline))):
        completion = eval(reviews, baseline[i], gpt_o1_summary)
        if '[[A]]' in completion:
            base_count += 1
        elif '[[B]]' in completion:
            o1_count += 1
        elif '[[C]]' in completion:
            draw_count += 1
    return base_count, o1_count, draw_count


# 모델 고도화1 - 조건 명시
prompt_update = f"""당신은 요약 전문가 입니다. 사용자 숙소 리뷰들이 주어졌을 때 요약하는 것이 당신의 목표입니다.

요약 결과는 다음 조건들을 충족해야 합니다:
1. 모든 문장은 항상 존댓말로 작성되어야 합니다.
2. 숙소에 대해 소개하는 톤앤매너로 작성해주세요.
    2-1. 좋은 예시
        a) 전반적으로 좋은 숙소였고 방음도 괜찮았다는 평입니다.
        b) 재방문 예정이라는 평들이 존재합니다.
    2-2. 나쁜 예시
        a) 좋은 숙소였고 방음도 괜찮았습니다.
        b) 재방문 예정입니다.
3. 요약 결과는 최소 4문장, 최대 7문장 사이로 작성해주세요.

아래 숙소 리뷰들에 대해 요약해주세요:"""

# 모델 고도화2 - 입력 데이터 품질 증가
def preprocess_review_update(review_list):
    
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

        if len(r['review']) < 30:
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

#모델 고도화3 - Few-shot Prompting
reviews_update, _ = preprocess_review_update(review_list)
summaries_1shot = baseline_model(reviews_update, prompt_update, temperature=0.0,model='gpt-4o-2024-11-20')

# 모델 고도화 1를 적용하지 않은 프롬프트
'''
prompt_1shot = f"""당신은 요약 전문가 입니다. 사용자 숙소 리뷰들이 주어졌을 때 요약하는 것이 당신의 목표입니다.

요약 결과는 다음 조건들을 충족해야 합니다:
1. 모든 문장은 항상 존댓말로 작성되어야 합니다.
2. 숙소에 대해 소개하는 톤앤매너로 작성해주세요.
    2-1. 좋은 예시
        a) 전반적으로 좋은 숙소였고 방음도 괜찮았다는 평입니다.
        b) 재방문 예정이라는 평들이 존재합니다.
    2-2. 나쁜 예시
        a) 좋은 숙소였고 방음도 괜찮았습니다.
        b) 재방문 예정입니다.
3. 요약 결과는 최소 2문장, 최대 5문장 사이로 작성해주세요.

다음은 리뷰들과 요약 예시입니다.
예시 리뷰들:
{reviews_1shot}
예시 요약 결과:
{summaries_1shot}

아래 숙소 리뷰들에 대해 요약해주세요:"""
'''

prompt_1shot = f"""당신은 요약 전문가 입니다. 사용자 숙소 리뷰들이 주어졌을 때 요약하는 것이 당신의 목표입니다.

다음은 리뷰들과 요약 예시입니다.
예시 리뷰들:
{reviews_update}
예시 요약 결과:
{summaries_1shot}

아래 숙소 리뷰들에 대해 요약해주세요:"""

# 검사
summaries = [baseline_model(reviews_update, prompt_1shot, temperature=1.0,model='gpt-4-turbo') for _ in range(eval_count)]
wins, losses, ties = eval_batch(reviews_update, summaries, [gpt_o1_summary for _ in range(len(summaries))])
print(f'Wins: {wins}, Losses: {losses}, Ties: {ties}')