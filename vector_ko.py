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
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    outputs = model(**inputs)
    return outputs.last_hidden_state[:,0,:].detach().numpy()

# 문서 예시
documents = [
    "파리는 프랑스의 수도입니다.",
    "토마토는 과일이냐고 묻는 사람들이 있는데, 토마토는 야채랍니다.",
    "구글은 미국의 대표적인 기술 회사입니다.",
    "애플은 아이폰을 만드는 회사입니다.",
    "피자는 이탈리아 음식입니다."
]

#도메인적인 지식이 필요함.
for document in documents:
    if(document.find("구글") >= 0 or document.find("애플") >= 0):
        document = "기술 회사인, " + document
    print(document)

# 문서 벡터화
document_embeddings = np.vstack([embed_text(doc) for doc in documents])

# Faiss 인덱스 생성 및 벡터 추가
dimension = document_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(document_embeddings.astype('float32'))

# 검색할 쿼리
query = "기술 회사의 예"
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