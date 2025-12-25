import torch,torchaudio,json
from speechbrain.pretrained import EncoderClassifier as sb
import warnings

warnings.filterwarnings("ignore",category=UserWarning)
device ="cuda" if torch.cuda.is_available() else "cpu"

model=sb.from_hparams(   source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="ecapa_model",
    run_opts={"device": device})

def embed_audio(wav_path):
    signal,_=torchaudio.load(wav_path)

    if signal.shape[0]>1:
        signal=signal.mean(dim=0,keepdim=True)
    
    with torch.no_grad():
        emb=model.encode_batch(signal)
    return emb.squeeze().tolist()

username="tamine"

file=[r"D:\timepass proj\recording\Recording (7).wav",
r"D:\timepass proj\recording\Recording (8).wav",
r"D:\timepass proj\recording\Recording (9).wav",
r"D:\timepass proj\recording\Recording (10).wav"]

embed=[embed_audio(f) for f in file]
db={username:embed}

with open("voice_json_db.json","w")as f:
    json.dump(db,f,indent=2)

print("done")