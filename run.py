from sentence_transformers import SentenceTransformer, util
import torch
import re

model = SentenceTransformer('all-MiniLM-L6-v2')

documents = [
    "[Definition]There is a distinction between heart and heart valve disease: congenital and acquired. Congenital heart defects are malformations in the heart septum, great vessels, or heart valves that result in incomplete or obstructed flow of blood. Most cases of pulmonary aortic valve disease are due to congenital lesions, and some aortic and sometimes tricuspid valve diseases are also congenital. These defects are caused by abnormalities in their formation during gestation, improper or insufficient continuation of development or delayed maturation, and may be due to infections, medications, or genetic disease. Acquired heart valve disease is caused by the breakdown or thickening of the heart valves, resulting in decreased heart function. The main cause is rheumatic endocarditis. Rheumatic endocarditis commonly affects the mitral valve, occasionally the aortic valve, and relatively rarely the tricuspid valve.",
    "Syphilis can affect the aortic valve and ascending aorta, but not the other locations. Conditions that can appear in the heart valves include Stenosis: a narrowing or narrowing of the opening of the heart valve due to scarring at the edge of the valve, blocking blood flow Failure (obstruction, regurgitation): the valve is incomplete, allowing blood to flow backwards in one or more chambers Rapid onset mitral valve failure: failure of the mitral valve to close due to heart enlargement",
    "[Definition] A tachycardia attack with typical electrocardiographic findings of PR interval shortening and indistinct QRS complexes associated with arrhythmias, which bypass the atrioventricular node and conduction into the AV node, resulting in premature ventricular excitation. Treatment: It can be treated with medication, radiofrequency ablation or surgery using radiofrequency radiation energy. Prognosis: Depends on the occurrence of atrial flutter, atrial fibrillation (precordial risk), and syncope due to excessive ventricular rate.",
    "Definition : A balloon-like vascular malformation at the site of a cerebral artery bifurcation that ruptures, resulting in intracranial hemorrhage, headache, or neurologic impairment.About 65% of subarachnoid hemorrhages are caused by brain aneurysms (especially strawberry aneurysms). Causes : It was once thought to be caused by congenital weakness of the vessel wall of the cerebral artery bifurcation.", 
    "But recent studies have shown that cerebral aneurysms can also occur when the vessel wall is damaged by continuous hemodynamic stress. Symptoms: The patient complains of an excruciating headache that feels like an explosion, a headache that feels like being hit in the head with a hammer. Ptosis, a dilated pupil or drooping eyelid, may occur without a headache, and may be accompanied by pain in the back of the neck or back pain, and in severe cases, hemiparesis or loss of consciousness. Treatment: CT , cerebral angiography, MR angiography can be used to accurately locate the location of the aneurysm, and then use clips to ligate the aneurysm.",
]

#도메인적인 지식이 필요함.
#for document in documents:
#    if(document.find("google") >= 0 or document.find("apple") >= 0):
#        document = "The technology company, " + document
#    if(document.find("iPhone") >= 0):
#        document = "The techonolgy company, Apple is the company that makes the iPhone and has the world's leading cell phone manufacturing technology."
#    print(document)

query = "Tell me about the symptoms of a brain aneurysm"
query = re.sub(r'[^\w\s]', ' ', query)
print(query)

query_embedding = model.encode(query, convert_to_tensor=True)
document_embeddings = model.encode(documents, convert_to_tensor=True)

cosine_scores = util.pytorch_cos_sim(query_embedding, document_embeddings)

top_result = torch.argmax(cosine_scores)

print(cosine_scores)

print(f"쿼리: '{query}'")
print(f"가장 관련성 높은 문서: '{documents[top_result]}'")