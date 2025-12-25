import json,torch,torchaudio
import torch.nn.functional as f
from speechbrain.pretrained import EncoderClassifier

device="cuda" if torch.cuda.is_available() else "cpu"

model=EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir=r"D:\timepass proj\ecapa_model",
    run_opts={"device":device})

def get_embed(wav):
    signal,_=torchaudio.load(wav) # tensor[channel,sample],sample
    
    if signal.shape[0]>1:
        signal=signal.mean(dim=0,keepdim=True) # more channel reduce to 1

    with torch.no_grad():
        embed=model.encode_batch(signal.to(device)) # convert to embed
    
    return embed.squeeze().cpu()

with open("voice_json_db.json","r")as file:
    db=json.load(file)

test_embed=get_embed(r"D:\timepass proj\record_9bfe68a485594eb79f3a9650e49b600a.wav")
threshold=0.8

for user,store_embed in db.items():
    scores=[]
    for e in store_embed:
        e=torch.tensor(e)
        score=f.cosine_similarity(test_embed,e,dim=0).item()
        scores.append(score)

    avg_score=sum(scores)/len(scores)
    print(f"{user} similarity:",avg_score)
    if avg_score>threshold:
        print("speaker recognise:",user)
    else:
        print("no recog")