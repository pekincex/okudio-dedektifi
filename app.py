"""
Okudio Okuma Dedektifi — Faz 3.1
"""
from pydub import AudioSegment
import os, io, json, tempfile, datetime, re, requests, base64
import numpy as np
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

app = Flask(__name__, template_folder='templates/templates')
app.config['SECRET_KEY'] = os.environ.get("SECRET_KEY", "okudio-secret-2025")
db_url = os.environ.get("DATABASE_URL", "sqlite:///okudio.db")
if db_url.startswith("postgres://"): db_url = db_url.replace("postgres://", "postgresql://", 1)
app.config['SQLALCHEMY_DATABASE_URI'] = db_url
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

SINIF_REF = {
    "2": {"wpm_min":40,"wpm_max":60,"dogruluk_min":95,"kelime":"40-55","zorluk":"Cok basit cumleler, 4-6 kelimelik cumleler, gunluk bilinen kelimeler"},
    "3": {"wpm_min":70,"wpm_max":90,"dogruluk_min":95,"kelime":"55-75","zorluk":"Basit cumleler, 6-8 kelimelik, biraz daha fazla sifat ve zarf"},
    "4": {"wpm_min":90,"wpm_max":110,"dogruluk_min":97,"kelime":"75-95","zorluk":"Orta zorluk, bilesik cumleler, 8-10 kelimelik cumleler"},
    "5": {"wpm_min":110,"wpm_max":130,"dogruluk_min":98,"kelime":"95-115","zorluk":"Karisik cumleler, soyut kavramlar, 10-12 kelimelik"},
    "6": {"wpm_min":130,"wpm_max":145,"dorluk_min":99,"kelime":"115-135","zorluk":"Uzun ve karisik cumleler, akademik kelimeler"},
    "7": {"wpm_min":145,"wpm_max":160,"dogruluk_min":99,"kelime":"135-155","zorluk":"Edebi dil, mecazlar, uzun paragraflar"},
}

# ─── MODELS ───
class User(UserMixin, db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    role = db.Column(db.String(10), default='student')
    name = db.Column(db.String(120), nullable=False)
    sinif = db.Column(db.String(2), default='2')
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    sessions = db.relationship('ReadingSession', backref='user', lazy=True)
    def set_password(self, pw): self.password_hash = generate_password_hash(pw)
    def check_password(self, pw): return check_password_hash(self.password_hash, pw)

class ReadingSession(db.Model):
    __tablename__ = 'reading_sessions'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    asama = db.Column(db.Integer, default=1)
    referans_metin = db.Column(db.Text)
    transkript = db.Column(db.Text)
    dogruluk = db.Column(db.Float, default=0)
    akicilik = db.Column(db.Float, default=0)
    prozodi = db.Column(db.Float, default=0)
    genel_skor = db.Column(db.Float, default=0)
    wpm = db.Column(db.Integer, default=0)
    hatali_json = db.Column(db.Text, default='[]')
    rapor_json = db.Column(db.Text, default='{}')
    tamamlandi = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)

@login_manager.user_loader
def load_user(id): return User.query.get(int(id))

# ─── GOOGLE STT ───
def google_stt_analiz(ses_dosya_yolu):
    with open(ses_dosya_yolu, "rb") as f:
        audio_b64 = base64.b64encode(f.read()).decode("utf-8")
    resp = requests.post(f"https://speech.googleapis.com/v1p1beta1/speech:recognize?key={GOOGLE_API_KEY}",
        json={"config": {"encoding": "LINEAR16", "sampleRateHertz": 16000, "languageCode": "tr-TR", "enableWordTimeOffsets": True, "enableWordConfidence": True, "model": "default", "useEnhanced": True}, "audio": {"content": audio_b64}}, timeout=120)
    data = resp.json()
    if "error" in data or "results" not in data: return "", []
    transkript = ""; kelimeler = []
    for result in data["results"]:
        alt = result["alternatives"][0]; transkript += alt.get("transcript", "") + " "
        for w in alt.get("words", []):
            kelimeler.append({"kelime": w["word"], "baslangic": round(float(w.get("startTime", "0s").replace("s", "")), 2), "bitis": round(float(w.get("endTime", "0s").replace("s", "")), 2), "confidence": round(w.get("confidence", 0), 3)})
    return transkript.strip(), kelimeler

# ─── IMPROVED COMPARISON (2-pass) ───
def lev_sim(a, b):
    if a == b: return 1.0
    if not a or not b: return 0.0
    la, lb = len(a), len(b)
    m = [[0]*(lb+1) for _ in range(la+1)]
    for i in range(la+1): m[i][0] = i
    for j in range(lb+1): m[0][j] = j
    for i in range(1, la+1):
        for j in range(1, lb+1):
            m[i][j] = m[i-1][j-1] if a[i-1]==b[j-1] else min(m[i-1][j-1],m[i][j-1],m[i-1][j])+1
    return (max(la,lb)-m[la][lb])/max(la,lb)

def kelime_karsilastir(referans_metin, stt_kelimeler):
    ref_words = re.sub(r'[^\w\s]', '', referans_metin).lower().split()
    stt_clean = [re.sub(r'[^\w\s]', '', k["kelime"]).lower() for k in stt_kelimeler]
    stt_used = [False] * len(stt_clean)
    sonuc = [None] * len(ref_words)

    # Pass 1: Sequential greedy match
    stt_idx = 0
    for i, ref_w in enumerate(ref_words):
        best_sim = 0; best_j = -1
        for j in range(stt_idx, min(len(stt_clean), stt_idx + 20)):
            if stt_used[j]: continue
            s = lev_sim(ref_w, stt_clean[j])
            if s > best_sim: best_sim = s; best_j = j
        if best_sim >= 0.8 and best_j >= 0:
            stt_used[best_j] = True
            sonuc[i] = {"kelime": ref_w, "durum": "dogru", "confidence": stt_kelimeler[best_j]["confidence"], "stt": stt_kelimeler[best_j]["kelime"]}
            stt_idx = best_j + 1
        elif best_sim >= 0.4 and best_j >= 0:
            stt_used[best_j] = True
            sonuc[i] = {"kelime": ref_w, "durum": "hatali", "confidence": stt_kelimeler[best_j]["confidence"], "stt": stt_kelimeler[best_j]["kelime"]}
            stt_idx = best_j + 1

    # Pass 2: Global search for unmatched (skipped) words
    for i, ref_w in enumerate(ref_words):
        if sonuc[i] is not None: continue
        best_sim = 0; best_j = -1
        for j in range(len(stt_clean)):
            if stt_used[j]: continue
            s = lev_sim(ref_w, stt_clean[j])
            if s > best_sim: best_sim = s; best_j = j
        if best_sim >= 0.75 and best_j >= 0:
            stt_used[best_j] = True
            sonuc[i] = {"kelime": ref_w, "durum": "dogru" if best_sim >= 0.9 else "hatali", "confidence": stt_kelimeler[best_j]["confidence"], "stt": stt_kelimeler[best_j]["kelime"]}
        else:
            sonuc[i] = {"kelime": ref_w, "durum": "atlanmis", "confidence": 0, "stt": ""}

    dogru = sum(1 for s in sonuc if s["durum"] == "dogru")
    toplam = len(ref_words)
    return {"kelimeler": sonuc, "toplam": toplam, "dogru": dogru, "hatali": sum(1 for s in sonuc if s["durum"] == "hatali"), "atlanmis": sum(1 for s in sonuc if s["durum"] == "atlanmis"), "dogruluk_yuzdesi": round(dogru / toplam * 100, 1) if toplam > 0 else 0}

# ─── PROZODI ───
def prozodi_analiz(ses_dosya_yolu):
    import parselmouth, librosa
    ses = parselmouth.Sound(ses_dosya_yolu); ts = ses.get_total_duration()
    pitch = ses.to_pitch(); pv = pitch.selected_array['frequency']; pv = pv[pv > 0]
    pd = {"ortalama_hz": round(float(np.mean(pv)),1), "std_hz": round(float(np.std(pv)),1), "monotonluk": "Monoton" if float(np.std(pv))<20 else "Normal" if float(np.std(pv))<40 else "Canli"} if len(pv) > 0 else {"ortalama_hz":0,"std_hz":0,"monotonluk":"?"}
    intensity = ses.to_intensity(); iv = intensity.values.T.flatten(); iv = iv[iv>40]
    ed = {"ortalama_db": round(float(np.mean(iv)),1), "std_db": round(float(np.std(iv)),1), "vurgu": "Yetersiz" if float(np.std(iv))<3.5 else "Normal" if float(np.std(iv))<7 else "Guclu"} if len(iv)>0 else {"ortalama_db":0,"std_db":0,"vurgu":"?"}
    y, sr = librosa.load(ses_dosya_yolu, sr=None); sk = librosa.effects.split(y, top_db=30)
    ks=0;dc=0;ul=0
    for i, k in enumerate(sk):
        ks += (k[1]-k[0])/sr
        if i > 0:
            ds = k[0]/sr - sk[i-1][1]/sr
            if ds > 0.15: dc+=1; ul+=1 if ds>1.0 else 0
    ss=ts-ks; so=(ss/ts)*100 if ts>0 else 0
    return {"pitch":pd,"enerji":ed,"sure":{"toplam":round(ts,1),"konusma":round(ks,1),"sessizlik":round(so,1),"duraksama":dc,"uzun":ul,"akicilik":"Akici" if so<20 else "Normal" if so<35 else "Kesik"}}

# ─── CLAUDE ───
def claude_analiz(ref, kar, pz, sinif, ad, asama=1):
    import anthropic
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    sr = SINIF_REF.get(sinif, SINIF_REF["2"])
    hl = ", ".join([f"{k['kelime']}({k['stt']},c:{k['confidence']})" for k in kar["kelimeler"] if k["durum"] != "dogru"][:15])
    prompt = f"""{sinif}. sinif "{ad}" okuma (Asama {asama}).
T:{kar['toplam']} D:{kar['dogru']} H:{kar['hatali']} A:{kar['atlanmis']} %{kar['dogruluk_yuzdesi']}
Hatali: {hl or 'Yok'}
Pitch:{pz['pitch']['ortalama_hz']}Hz std:{pz['pitch']['std_hz']}({pz['pitch']['monotonluk']}), E:{pz['enerji']['std_db']}dB({pz['enerji']['vurgu']}), S:{pz['sure']['toplam']}sn, Sess:%{pz['sure']['sessizlik']}, Dur:{pz['sure']['duraksama']}({pz['sure']['uzun']}uz)
REF WPM:{sr['wpm_min']}-{sr['wpm_max']} min%{sr.get('dogruluk_min',95)} WPM={kar['dogru']}x60/{pz['sure']['konusma']}
SADECE JSON:
{{"genel_skor":0-100,"akicilik_skoru":0-100,"prozodi_skoru":0-100,"seviye":"Baslangic/Gelisen/Yeterli/Ileri","wpm":N,"kaba":{{"toplam":{kar['toplam']},"dogru":{kar['dogru']},"yanlis":{kar['hatali']},"atlanan":{kar['atlanmis']},"wpm":N}},"prozodik_olcek":[{{"m":"Duygu","p":0-4}},{{"m":"Konusma","p":0-4}},{{"m":"Vurgu","p":0-4}},{{"m":"Noktalama","p":0-4}},{{"m":"Anlam","p":0-4}},{{"m":"Bekleme","p":0-4}},{{"m":"Akici","p":0-4}},{{"m":"Gruplama","p":0-4}}],"pt":0-32,"ao":"1 cumle","po":"1 cumle","hk":[{{"k":"x","o":"y","t":"he-ce","a":"anlam","ht":"sub/omi/met"}}],"gy":["2"],"ga":["2"],"on":["3"]}}"""
    response = client.messages.create(model="claude-sonnet-4-20250514", max_tokens=2000, messages=[{"role": "user", "content": prompt}])
    raw = response.content[0].text.strip()
    for p in ["```json","```"]:
        if raw.startswith(p): raw=raw[len(p):]
    if raw.endswith("```"): raw=raw[:-3]
    try: return json.loads(raw.strip())
    except: return {"hata":"Parse hatasi"}

# ─── ROUTES ───
@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('admin_panel' if current_user.role=='admin' else 'student_panel'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET','POST'])
def login():
    if current_user.is_authenticated: return redirect(url_for('index'))
    if request.method=='POST':
        user = User.query.filter_by(username=request.form['username']).first()
        if user and user.check_password(request.form['password']):
            login_user(user); return redirect(url_for('index'))
        flash('Hatali giris','error')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout(): logout_user(); return redirect(url_for('login'))

@app.route('/admin')
@login_required
def admin_panel():
    if current_user.role!='admin': return redirect(url_for('student_panel'))
    students = User.query.filter_by(role='student').order_by(User.name).all()
    sd = []
    for s in students:
        last = ReadingSession.query.filter_by(user_id=s.id).order_by(ReadingSession.created_at.desc()).first()
        total = ReadingSession.query.filter_by(user_id=s.id).count()
        sd.append({"user":s,"last":last,"total":total})
    return render_template('admin.html', students=sd)

@app.route('/admin/ogrenci-ekle', methods=['POST'])
@login_required
def ogrenci_ekle():
    if current_user.role!='admin': return redirect(url_for('index'))
    u=request.form.get('username','').strip();p=request.form.get('password','').strip();n=request.form.get('name','').strip();s=request.form.get('sinif','2')
    if not u or not p or not n: flash('Tum alanlari doldurun','error'); return redirect(url_for('admin_panel'))
    if User.query.filter_by(username=u).first(): flash('Kullanici var','error'); return redirect(url_for('admin_panel'))
    user=User(username=u,name=n,sinif=s,role='student'); user.set_password(p)
    db.session.add(user); db.session.commit(); flash(f'{n} eklendi','success')
    return redirect(url_for('admin_panel'))

@app.route('/admin/ogrenci-sil/<int:uid>', methods=['POST'])
@login_required
def ogrenci_sil(uid):
    if current_user.role!='admin': return redirect(url_for('index'))
    user=User.query.get_or_404(uid); ReadingSession.query.filter_by(user_id=uid).delete()
    db.session.delete(user); db.session.commit(); flash('Silindi','success')
    return redirect(url_for('admin_panel'))

@app.route('/admin/ogrenci/<int:uid>')
@login_required
def ogrenci_detay(uid):
    if current_user.role!='admin': return redirect(url_for('index'))
    return render_template('ogrenci_detay.html', student=User.query.get_or_404(uid), sessions=ReadingSession.query.filter_by(user_id=uid).order_by(ReadingSession.created_at.desc()).all())

@app.route('/ogrenci')
@login_required
def student_panel():
    if current_user.role=='admin': return redirect(url_for('admin_panel'))
    return render_template('student.html', sessions=ReadingSession.query.filter_by(user_id=current_user.id).order_by(ReadingSession.created_at.desc()).limit(20).all())

@app.route('/metin-olustur', methods=['POST'])
@login_required
def metin_olustur():
    import anthropic
    sinif = request.json.get('sinif', current_user.sinif or '2')
    sr = SINIF_REF.get(sinif, SINIF_REF["2"])
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    response = client.messages.create(model="claude-sonnet-4-20250514", max_tokens=400, messages=[{"role": "user", "content": f"""{sinif}. sinif Turkce okuma metni yaz.
{sr['kelime']} kelime. {sr['zorluk']}.
Cocuklara uygun ilgi cekici konu (hayvanlar, doga, macera, bilim, uzay, dostluk).
Her seferinde FARKLI konu. BASLIK YAZMA. Sadece metni yaz, hicbir aciklama ekleme."""}])
    return jsonify({"metin": response.content[0].text.strip()})

@app.route('/asama-metin', methods=['POST'])
@login_required
def asama_metin():
    import anthropic
    sinif = request.json.get('sinif', current_user.sinif or '2')
    hatali = request.json.get('hatali_kelimeler', [])
    sr = SINIF_REF.get(sinif, SINIF_REF["2"])
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    response = client.messages.create(model="claude-sonnet-4-20250514", max_tokens=400, messages=[{"role": "user", "content": f"""{sinif}. sinif okuma metni. MUTLAKA su kelimeleri icersin: {', '.join(hatali[:10])}. {sr['kelime']} kelime. {sr['zorluk']}. BASLIK YAZMA. Sadece metin."""}])
    return jsonify({"metin": response.content[0].text.strip()})

@app.route('/analiz', methods=['POST'])
@login_required
def analiz():
    try:
        if 'ses_dosyasi' not in request.files: return jsonify({"hata":"Dosya yok"}),400
        sf=request.files['ses_dosyasi'];rm=request.form.get('referans_metin','').strip()
        asama=int(request.form.get('asama','1'))
        if not rm: return jsonify({"hata":"Metin yok"}),400
        sx=os.path.splitext(sf.filename)[1] or '.wav'
        with tempfile.NamedTemporaryFile(delete=False,suffix=sx) as tmp: sf.save(tmp.name); tp=tmp.name
        wp=tp+"_c.wav"
        try:
            AudioSegment.from_file(tp).set_frame_rate(16000).set_channels(1).set_sample_width(2).export(wp,format="wav")
            tr,sk=google_stt_analiz(wp)
            if not tr or len(sk)<3: return jsonify({"basarili":False,"yeniden_kayit":True,"sebep":"Ses algilanamadi."})
            kar=kelime_karsilastir(rm,sk)
            pz=prozodi_analiz(wp)
            rp=claude_analiz(rm,kar,pz,current_user.sinif or '2',current_user.name,asama)
            hl=list(set([k["kelime"] for k in kar["kelimeler"] if k["durum"]!="dogru"]))
            tam=kar["dogruluk_yuzdesi"]>=95
            sess=ReadingSession(user_id=current_user.id,asama=asama,referans_metin=rm,transkript=tr,dogruluk=kar["dogruluk_yuzdesi"],akicilik=rp.get("akicilik_skoru",0),prozodi=rp.get("prozodi_skoru",0),genel_skor=rp.get("genel_skor",0),wpm=rp.get("wpm",0),hatali_json=json.dumps(hl,ensure_ascii=False),rapor_json=json.dumps(rp,ensure_ascii=False),tamamlandi=tam)
            db.session.add(sess);db.session.commit()
            return jsonify({"basarili":True,"ogrenci_adi":current_user.name,"sinif":current_user.sinif,"asama":asama,"referans_metin":rm,"transkript":tr,"stt_kelimeler":sk,"karsilastirma":kar,"prozodi":pz,"rapor":rp,"hatali_liste":hl,"tamamlandi":tam})
        finally:
            if os.path.exists(tp): os.unlink(tp)
            if os.path.exists(wp): os.unlink(wp)
    except Exception as e:
        import traceback;traceback.print_exc();return jsonify({"hata":str(e)}),500

with app.app_context():
    db.create_all()
    if not User.query.filter_by(role='admin').first():
        a=User(username='admin',name='Admin',role='admin',sinif='0');a.set_password('admin123')
        db.session.add(a);db.session.commit();print("  Admin: admin/admin123")

if __name__=='__main__':
    app.run(debug=True,port=8080,host='0.0.0.0')
