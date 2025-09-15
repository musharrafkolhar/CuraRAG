import whisper
import json
model = whisper.load_model("large-v2")
result = model.transcribe(audio= "audios/lecture_6.mp3",
                          language = "en",
                          word_timestamps=False)
print(result["segments"]) 
with open ("output.json", "w") as f:
    json.dump(result, f)

chunks=[]
for segment in result["segments"]:
    chunks.append({"start": segment["start"], "end":segment["end"], 
                  "text": segment["text"]})
    
print(chunks)
with open("output.json","w") as f:
    json.dump(chunks,f)