from transformers import ElectraTokenizer, ElectraModel
import torch
import faiss
import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# KoELECTRA 모델 및 토크나이저 로드
model_name = "monologg/koelectra-base-v3-discriminator"
tokenizer = ElectraTokenizer.from_pretrained(model_name)
model = ElectraModel.from_pretrained(model_name)

#텍스트를 벡터로 변환하는 함수
def embed_text(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=256)
    outputs = model(**inputs)
    return outputs.last_hidden_state[:,0,:].detach().numpy()

# 문서 예시
documents = [
    "심장판막(cardiac valve, heart valve)은 혈액이 심장의 방을 통해 한 방향으로 흐를 수 있도록 하는 단방향의 판막이다. 4개의 판막이 일반적으로 포유동물의 심장에 존재하며 함께 심장을 통한 혈류의 경로를 결정한다. 심장판막은 판막 양쪽의 혈압 차이에 따라 열리거나 닫힌다.",
    "포유류 심장에는 4개의 판막이 존재한다. 좌심실의 승모판과 우심장의 삼첨판은 위쪽의 심방과 아래쪽의 심실을 분리하는 2개의 방실판막이다. 다른 2개의 판막은 심장에서 나가는 동맥의 입구에 있으며 반달판막이다. 대동맥판은 대동맥 시작 지점에 존재하는 판막이고 허파동맥판은 허파동맥 시작 지점의 판막이다.",
    "판막, 심방, 심실은 심장속막(심내막)으로 둘러싸여 있다. 판막은 심방을 심실과 나누거나 심실을 동맥과 분리한다. 심장판막은 심장골격의 섬유성 고리 주변에 존재한다. 판막은 첨판(leaflets, cusps)으로 구성되어 있다. 승모판은 두 개의 첨판으로 구성되어 있고 나머지 세 판막은 세 개의 첨판으로 구성되어 있다.",
    "간암은 원발성(간에서 시작됨) 또는 이차성(다른 곳에서 간으로 전이된 암을 의미하며 간 전이로 알려짐)이 될 수 있다.[7] 간 전이는 간에서 시작되는 것보다 더 흔하다.[8] 간암은 전 세계적으로 증가하고 있다",
    "원발성 간암은 전 세계에서 6번째로 많이 발생하는 암이자 암으로 인한 사망 원인 4위이다.[11][12] 2018년에는 전 세계적으로 84만1000명이 간암에 걸려 78만2000명이 사망했다.[11] 아시아, 사하라 이남 아프리카 등 B형 및 C형 간염이 흔한 지역에서 간암 발생률이 높다.[8] 남성이 여성보다 간세포암종(HCC)에 더 자주 걸린다.[13] 진단은 55~65세 사이에서 가장 자주 발생한다.",
    "치료는 간이식, 간절제술, 경동맥 화학 색전술, 고주파 열치료법, 알코올 주입술 등이 있다. 세계적으로 간암 발생률이 높은 지역은 아프리카, 대만, 중국, 한국, 동남아시아, 일본 등이 있다. 간암이 낮은 지역은 아메리카지역과 영국, 오스트레일리아, 사우디아라비아, 오만, 아랍에미리트 등 중동 지역 등이 있다.",
    "간암의 주요 원인은 B형 간염, C형 간염 또는 알코올로 인한 간경변이다.[4] 다른 원인으로는 아플라톡신, 비알코올성 지방간 질환, 간 흡충[3] 등이 있다. 가장 흔한 유형은 간세포암으로, 간세포암의 80%를 차지하며 간내 담관암도 있다. 진단은 혈액 검사와 영상의학적으로 뒷받침될 수 있으며 조직 생검으로 확인할 수 있다.",
    "간암의 원인은 매우 다양하기 때문에 간암 예방을 위한 다양한 접근법이 있다. 이러한 노력에는 B형 간염 예방접종[3], B형 간염 치료, C형 간염 치료, 알코올 사용 줄이기, 농업에서 아플라톡신 노출 줄이기, 비만 및 당뇨병 관리 등이 포함된다.[15][16] 만성 간 질환이 있는 사람은 선별 검사를 받는 것이 좋다.[3] 예를 들어, 간세포암종에 걸릴 위험이 있는 만성 간 질환자는 6개월마다 초음파 검사를 받는 것이 좋다",
    "코로나바이러스감염증-19(코로나19, 영어: coronavirus disease 2019, COVID-19) 또는 신종 코로나바이러스 감염증(문화어: 신형코로나비루스감염증)은 SARS-CoV-2가 일으키는 중증 호흡기 증후군이다. 2019년 12월에 중화인민공화국에서 첫 사례가 보고되었고[1] 전 세계적으로 퍼져나가면서 유행병으로 자리잡았다.",
    "코로나-19의 증상은 다양하지만 발열[3][4] , 기침 등이 있다.[5][6] 증상은 바이러스에 감염된지 1~14일 안에 나타난다. 특히 감염된 사람 중 3분의 1은 무증상 감염자로 눈에 띈 증상이 나타나지 않는다.[7] 환자로 분류될 만큼 눈에 띄는 81%의 사람들은 경증에서 중증의 증상이 발생하며, 14%의 사람들은 호흡 곤란, 저산소증 등 증상이 발생하며, 5%의 사람들은 호흡기 부전, 쇼크 등 심각한 증상이 발생한다.[8] 고령자는 심각한 증상이 발생할 확률이 더 높으며, 일부 사람들은 회복후 긴 시간 동안 접한 코로나19 때문에 장기 손상이 관찰되었다.[9] 질병의 장기적인 영향을 더 조사하기 위해 오랜 기간 동안 조사가 이루어지고 있다."
]

# 문서 벡터화
document_embeddings = np.vstack([embed_text(doc) for doc in documents])

# Faiss 인덱스 생성 및 벡터 추가
dimension = document_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(document_embeddings.astype('float32'))

# 검색할 쿼리
query = "코로나-19 증상이 뭐임?"
query_embedding = embed_text(query)[0].astype('float32')

print("Query embedding shape:", query_embedding.shape)
print("Index dimension:", index.d)

# 쿼리 벡터와 가장 유사한 문서 검색
k = 2
D, I, *_ = index.search(np.array([query_embedding]), k)

print("쿼리:", query)
print("가장 유사한 문서:")
for idx, dist in zip(I[0], D[0]):
    print(f"- {documents[idx]} (거리: {dist})")