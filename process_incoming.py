from sklearn.metrics.pairwise import cosine_similarity
import joblib
import pandas as pd
import requests
import numpy as np


def create_embedding(text_list):
    r =requests.post ("http://localhost:11434/api/embed", json={
        "model":"bge-m3",
        "input":text_list
    })
    embedding = r.json()["embeddings"]
    return embedding

def inference(prompt , model):
    r =requests.post ("http://localhost:11434/api/generate", json={
        "model":"llama3.2",
        "prompt":prompt,
        "stream":False
    })
    response = r.json()
    print(response)
    return response


df = joblib.load("embeddings.joblib")
incoming_query=input("Ask a question:  ")
question_embedding = create_embedding([incoming_query])[0]
#Find similarities of question embeddings with other embeddings
# print(np.vstack(df["embedding"].values)) 
# print(np.vstack(df["embedding"]).shape)

similarities = cosine_similarity(np.vstack(df["embedding"]),[question_embedding]).flatten()
top_results = 5
max_idx=similarities.argsort()[::-1][0:top_results]
#print(similarities)
print(max_idx)

new_df=df.loc[max_idx]
#print(new_df[["text","source_file","chunk_id"]])
prompt = f''' Here are video chunks containing video title , chunk_id , text at that time
, start and end time in seconds 


{new_df[["text","source_file","chunk_id"]].to_json(orient="records")}



{incoming_query}
User asked this questions related to video chunks , you have to answer where and how much 
content is taught and where it is taught  in which video and what timestamp 
and guide the user to that particular video and timestamp also use these chunks to answer the users
questions , if user asks unrelated questions then tell them you answer questions related to this 
medicine course  



'''
with open ("prompt.txt","w") as f:
    f.write(prompt) 
response = inference(prompt, model="llama3.2")
print(response)

with open("response.txt","w") as f:
    f.write(response)
# for index , item in new_df.iterrows():
    
#     print(index, item["text"], ", chunk_id of the response: ", item["chunk_id"],
#            ", source_file :",item["source_file"], item["start"],item["end"])