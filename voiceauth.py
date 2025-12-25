import sounddevice as sd
import torch, torchaudio, uuid, json, os, random, ssl, smtplib
import torch.nn.functional as F
from speechbrain.pretrained import EncoderClassifier
from scipy.io.wavfile import write
from gdata import g_data

# ================= CONFIG =================
AUDIO_FS = 44100
RECORD_SEC = 8
THRESHOLD = 0.75
VOICE_DB = "voice_db.json"
DEVICE = "cpu"

# ================= LOAD MODEL ONCE =================
model = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="ecapa_model",
    run_opts={"device": DEVICE}
)

# ================= AUDIO =================
def record_audio():
    print("🎙 Recording... stay still")
    audio = sd.rec(int(AUDIO_FS * RECORD_SEC), samplerate=AUDIO_FS, channels=1, dtype="int16")
    sd.wait()
    fname = f"resp_{uuid.uuid4().hex}.wav"
    write(fname, AUDIO_FS, audio)
    return fname

# ================= EMBEDDING =================
def speaker_embed(wavfile):
    signal, _ = torchaudio.load(wavfile)
    if signal.shape[0] > 1:
        signal = signal.mean(dim=0, keepdim=True)

    with torch.no_grad():
        emb = model.encode_batch(signal)

    return emb.squeeze().cpu()

# ================= DATABASE =================
def load_db():
    if not os.path.exists(VOICE_DB):
        return {}
    with open(VOICE_DB, "r") as f:
        return json.load(f)

def save_db(db):
    with open(VOICE_DB, "w") as f:
        json.dump(db, f, indent=2)

# ================= EMAIL =================
def send_otp(email):
    otp = random.randint(10000, 99999)
    msg = f"Your OTP is {otp}"
    context = ssl.create_default_context()

    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
        server.login(g_data["username"], g_data["password"])
        server.sendmail(g_data["username"], email, msg)

    return otp

# ================= SIGNUP =================
def signup():
    email = input("Enter Gmail: ").strip()

    if not email.endswith("@gmail.com"):
        print("❌ Invalid email")
        return

    print("🎙 Recording 3 samples")
    embeds = []

    for i in range(3):
        print(f"Sample {i+1}")
        wav = record_audio()
        embeds.append(speaker_embed(wav).tolist())
        os.remove(wav)

    db = load_db()
    db[email] = embeds
    save_db(db)

    print("✅ Signup complete")

# ================= LOGIN =================
def login():
    db = load_db()
    if not db:
        print("❌ No users found")
        return

    print("🎙 Speak clearly")
    wav = record_audio()
    login_embed = speaker_embed(wav)
    os.remove(wav)

    best_score = 0
    best_user = None

    for email, embeds in db.items():
        for e in embeds:
            e = torch.tensor(e)
            score = F.cosine_similarity(login_embed, e, dim=0).item()
            if score > best_score:
                best_score = score
                best_user = email

    print("Voice score:", round(best_score, 3))

    if best_score < THRESHOLD:
        print("❌ Voice authentication failed")
        return

    print("✅ Voice verified")
    otp = send_otp(best_user)
    user_otp = int(input("Enter OTP: "))

    if user_otp == otp:
        print("🔥 LOGIN SUCCESS")
    else:
        print("❌ OTP incorrect")

# ================= MAIN =================
if __name__ == "__main__":
    print("""
    1. Signup
    2. Login
    """)

    choice = input("Select: ").strip()

    if choice == "1":
        signup()
    elif choice == "2":
        login()
    else:
        print("❌ Invalid option")
