from sentence_transformers import SentenceTransformer, util
import torch
import re

model = SentenceTransformer('all-MiniLM-L6-v2')

documents = [
    "A heart valve is a biological one-way valve that allows blood to flow in one direction through the chambers of the heart. Four valves are usually present in a mammalian heart and together they determine the pathway of blood flow through the heart. A heart valve opens or closes according to differential blood pressure on each side.[1][2][3]",
    "The four valves in the mammalian heart are two atrioventricular valves separating the upper atria from the lower ventricles – the mitral valve in the left heart, and the tricuspid valve in the right heart. The other two valves are at the entrance to the arteries leaving the heart these are the semilunar valves – the aortic valve at the aorta, and the pulmonary valve at the pulmonary artery.",
    "A brain tumor occurs when abnormal cells form within the brain.[2] There are two main types of tumors: malignant (cancerous) tumors and benign (non-cancerous) tumors.[2] These can be further classified as primary tumors, which start within the brain, and secondary tumors, which most commonly have spread from tumors located outside the brain, known as brain metastasis tumors.[1] All types of brain tumors may produce symptoms that vary depending on the size of the tumor and the part of the brain that is involved.[2] Where symptoms exist, they may include headaches, seizures, problems with vision, vomiting and mental changes.[1][2][7] Other symptoms may include difficulty walking, speaking, with sensations, or unconsciousness.",
    "Hypertension, also known as high blood pressure, is a long-term medical condition in which the blood pressure in the arteries is persistently elevated.[11] High blood pressure usually does not cause symptoms.[1] It is, however, a major risk factor for stroke, coronary artery disease, heart failure, atrial fibrillation, peripheral arterial disease, vision loss, chronic kidney disease, and dementia.[2][3][4][12] Hypertension is a major cause of premature death worldwide.", 
    "High blood pressure is classified as primary (essential) hypertension or secondary hypertension.[5] About 90–95% of cases are primary, defined as high blood pressure due to nonspecific lifestyle and genetic factors.[5][6] Lifestyle factors that increase the risk include excess salt in the diet, excess body weight, smoking, physical inactivity and alcohol use.[1][5] The remaining 5–10% of cases are categorized as secondary high blood pressure, defined as high blood pressure due to a clearly identifiable cause, such as chronic kidney disease, narrowing of the kidney arteries, an endocrine disorder, or the use of birth control pills.",
]

query = "Tell me about the heart valve"
query = re.sub(r'[^\w\s]', ' ', query)
print(query)

query_embedding = model.encode(query, convert_to_tensor=True)
document_embeddings = model.encode(documents, convert_to_tensor=True)

cosine_scores = util.pytorch_cos_sim(query_embedding, document_embeddings)

top_result = torch.argmax(cosine_scores)

print(cosine_scores)

print(f"쿼리: '{query}'")
print(f"가장 관련성 높은 문서: '{documents[top_result]}'")
