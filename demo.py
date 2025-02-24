import gradio as gr
import os
import json
import pickle
import datetime

from dateutil import parser
from openai import OpenAI


MAPPING = {'a':'./res/reviews.json', 'b':'./res/소노문단양.json'}

# 전처리 함수
def preprocess_review(review_name):

    with open(f'{review_name}', 'r', encoding='utf-8') as f:
        review_list = json.load(f)
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
# 요약 함수
def summarize_review(reviews):
    prompt_base = f"""당신은 요약 전문가 입니다. 사용자 숙소 리뷰들이 주어졌을 때 요약하는 것이 당신의 목표입니다.

다음은 리뷰들과 요약 예시입니다.
예시 리뷰들:
"가족들과 좋은시간 보내고 왔습니다.\n부대시설 좋고 수영장도 좋습니다.\n앞으로도 종종 이용할 것 같습니다!\n\n날씨 좋을 때 한번 더 방문할께요!",
"국내 여행가본 숙소중에 최고였습니다.\n스파, 조식 나무랄게 없었습니다.\n가족들과 소중한 추억 만들었습니다..",
"스파이용할수있어서 좋았구 숙소도 깨끗해서 좋았어요!",
"직원들의 고객응대 수준이 다론 곳과 비교할 수 없을만큼 훌륭했습니다.",
"침구 너무 폭닫하고 좋아서 자는데 가분이 ㅈㅎㅎ았어요. 뷰도 좋고, 직원분들이 너무 친절하셔서 또 오고싶어진 숙소였어요^^",
"룸이 전체적으로 깔끔해서 너무 좋네요 :)\n방3,거실1,화장실3이라서 매우 좋고\n노래방,편의점,bbq,스파 안에서 해결할수있어서 편리하네요",
"제주도 가려다가 결항으로 급하게 당일예약하고 제천 여행갔는데 스파도 좋고 너무 좋았습니다! 다만 스파는 겨울에 가시려면 야외스파가실때 신을 아쿠아 슈즈 꼭 챙기세요^^,,,",
"부산에서 출발해 오후4시반에 체크인.\n12시부터 대기표받아 선착순 룸배정이라네요\n모르고 갔더니 뷰는 포기!\n뒷산배경인데  눈까지 내려 그나마 ...\n룸깨끗.성인자녀들 트윈침대에 만족.\n연박하고 싶다고~\n4시반에 헤브나인스파 입장\n5시반부터 눈이 내리기 시작~\n퇴장 5분전 눈이 절정으로 내리고 뜨거운 스파물거품에 좋은 추억 간직하고 돌아갑니다",
"아이들이 너무 좋아했습니다.\n숙소도 너무 깨끗하고,온천도 너무 좋았습니다.",
"직원친절도 최상.  숙소 정비나 시설이 깨끗하고 뒷산뷰도 조용하고 좋았습니다.",
"추운 겨울 스파도 즐기고 겨울 설경을 보고 싶어서 가족과 함께 방문했습니다~! 눈이 오길 바랐었는데 눈까지 내려서 더욱 멋진 풍경에서 놀 수 있었어요ㅎㅎ 생각보다 커서 이곳 저곳 돌아다니며 물놀이 할 수 있었어요\n\n객실은 포레스트G40 이용했는데, 깨끗하고 괜찮았지만 숙박비에 비해 좀 아쉬운 부분들이 있었어요 \n일단 바닥이 차가운 부분들이 많았고 숙소 들어가자마자 한기가 느껴졌어요 \n난방이 안틀어져있는 상태였어서 들어가자마자 난방 켜고 패딩입고 꽤 오래 있었는데 그래도 춥더라구요ㅠ 침대 없는 방은 온돌방이였는데 춥길래 요랑 이불 깔아놨더니 그 부분만 그나마 조금 따뜻해졌고.. 온도가 쭉쭉 높아져야하는데 너무 시원찮고 난방 이상 있는 거 같아서 프론트에 전화했는데 난방 관리 기사님(?) 불러주셨는데도 추위가 비슷해서 다시 전화드렸고 방 바꿔주셨어요 \n\n똑같은 G40방을 배정 받았는데 난방은 미리 틀어두셨어서 이전과 달리 한기는 덜 느껴졌었어요 \n그래도 바닥은 부분부분 차더라구요ㅠ \n난방이 잘 안돼서 이전 숙소에서 패딩입고 있고 닭강정 시킨 것도 프론트 통화하고 기사님 방문하고 방 옮기느라 제대로 못먹고 시간 버렸는데 룸업그레이드도 없고 바닥 찬기는 거의 비슷했어서 아쉬웠네요ㅠ 룸컨디션 괜찮은지 여쭤보셨는데 시간도 늦었고 또 바꿔도 비슷할 거 같아서 그냥 거기서 묵었어요 그래도 12시 체크아웃으로 늘려주셨던 점은 좋았어요 \n\n바꾼방 침대있는 안방은 따뜻했고, 온돌방은 따뜻해져서 뜨끈하게 잤고 화장실은 추웠고, 거실은 바닥이 차서 약간의 한기가 느껴졌어요 \n\n그래도 직원분들 응대가 좋았고, 스파와 뷰가 넘 좋았어서 언젠가 한 번쯤 또 갈 지도?! \n대신 다음에 오게되면 겨울에는 안 갈 것 같고 신축 건물인 레스트리에 갈 거 같아요",
눈쌓은 마운틴뷰의 객실은 최고였습니다. 객실 내부도 청결했습니다. 다만 4인실인데 이불및베개를 추가로 주셨으나, 마땅히 깔 공간이 협소한점은 아쉬웠습니다. 해당 객실은 2인실로 제한하는게 쾌적한 환경에서 숙박이 가능할것으로 사료됩니다",

예시 요약 결과:
'숲속에 위치한 숙소로 자연 속에서 힐링하기에 최적입니다. 객실 상태가 깔끔하고 침구류가 편안해 쾌적한 숙박을 할 수 있습니다. 스파와 수영장 등 다양한 부대시설이 마련되어 있어 가족 단위뿐만 아니라 연인이나 친구들과 함께 즐기기 좋습니다. 직원들이 매우 친절하고 식당들도 품질이 높아 전반적으로 투숙객들에게 긍정적인 경험을 선사합니다. 사계절 내내 아름다운 풍경을 감상할 수 있어 재방문 의사가 높게 나타나는 인기 숙소로 평가됩니다.'

아래 숙소 리뷰들에 대해 요약해주세요:"""

    prompt = prompt_base + '\n\n' + reviews

    OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
    client = OpenAI(api_key=OPENAI_API_KEY)

    completion = client.chat.completions.create(
        model='gpt-3.5-turbo',
        messages=[{'role': 'user', 'content': prompt}],
        temperature= 0.0
    )
    return completion.choices[0].message.content


def fn(review_name):
    path = MAPPING[review_name]
    reviews_good, reviews_bad = preprocess_review(path)
    
    summary_good = summarize_review(reviews_good)
    summary_bad = summarize_review(reviews_bad)

    return summary_good, summary_bad

def run_demo():
    demo = gr.Interface(
        fn = fn,
        inputs = [gr.Radio(['a', 'b'], label = '숙소')],
        outputs = [gr.Textbox(label = '높은 평점요약'), gr.Textbox(label = '낮은 평점요약')],
    )
    demo.launch()


if __name__ == '__main__':
    run_demo()
#야놀자 프로젝트 끝