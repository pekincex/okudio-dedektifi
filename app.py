"""
Okudio Okuma Dedektifi — Faz 3
Auth + PostgreSQL + Admin/Ogrenci Paneli
"""
from pydub import AudioSegment
import os, io, json, tempfile, datetime, re, requests, base64
import numpy as np
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, send_file
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

app = Flask(__name__, template_folder='templates/templates')
app.config['SECRET_KEY'] = os.environ.get("SECRET_KEY", "okudio-secret-key-2025")

# DB — Railway PostgreSQL veya lokal SQLite
db_url = os.environ.get("DATABASE_URL", "sqlite:///okudio.db")
if db_url.startswith("postgres://"):
    db_url = db_url.replace("postgres://", "postgresql://", 1)
app.config['SQLALCHEMY_DATABASE_URI'] = db_url
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

SINIF_REF = {
    "2": {"wpm_min":40,"wpm_max":60,"dogruluk_min":95,"kelime_sayisi":"40-60"},
    "3": {"wpm_min":70,"wpm_max":90,"dogruluk_min":95,"kelime_sayisi":"60-80"},
    "4": {"wpm_min":90,"wpm_max":110,"dogruluk_min":97,"kelime_sayisi":"80-100"},
    "5": {"wpm_min":110,"wpm_max":130,"dogruluk_min":98,"kelime_sayisi":"100-120"},
    "6": {"wpm_min":130,"wpm_max":145,"dogruluk_min":99,"kelime_sayisi":"120-140"},
    "7": {"wpm_min":145,"wpm_max":160,"dogruluk_min":99,"kelime_sayisi":"140-160"},
}

# ─── MODELS ───
class User(UserMixin, db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    role = db.Column(db.String(10), nullable=False, default='student')  # admin / student
    name = db.Column(db.String(120), nullable=False)
    sinif = db.Column(db.String(2), default='2')
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    sessions = db.relationship('ReadingSession', backref='user', lazy=True, order_by='ReadingSession.created_at.desc()')

    def set_password(self, pw):
        self.password_hash = generate_password_hash(pw)
    def check_password(self, pw):
        return check_password_hash(self.password_hash, pw)

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
def load_user(id):
    return User.query.get(int(id))

# ─── ANALYSIS FUNCTIONS ───
def temizle(metin):
    return re.sub(r'[^\w\s]', '', metin).lower().strip()

def google_stt_analiz(ses_dosya_yolu):
    with open(ses_dosya_yolu, "rb") as f:
        audio_b64 = base64.b64encode(f.read()).decode("utf-8")
    payload = {"config": {"encoding": "LINEAR16", "sampleRateHertz": 16000, "languageCode": "tr-TR", "enableWordTimeOffsets": True, "enableWordConfidence": True, "model": "default", "useEnhanced": True}, "audio": {"content": audio_b64}}
    resp = requests.post(f"https://speech.googleapis.com/v1p1beta1/speech:recognize?key={GOOGLE_API_KEY}", json=payload, timeout=120)
    data = resp.json()
    if "error" in data or "results" not in data:
        return "", []
    transkript = ""; kelimeler = []
    for result in data["results"]:
        alt = result["alternatives"][0]; transkript += alt.get("transcript", "") + " "
        for w in alt.get("words", []):
            kelimeler.append({"kelime": w["word"], "baslangic": round(float(w.get("startTime", "0s").replace("s", "")), 2), "bitis": round(float(w.get("endTime", "0s").replace("s", "")), 2), "confidence": round(w.get("confidence", 0), 3)})
    return transkript.strip(), kelimeler

def kelime_karsilastir(referans_metin, stt_kelimeler):
    ref_words = re.sub(r'[^\w\s]', '', referans_metin).lower().split()
    stt_words = [{"kelime_temiz": re.sub(r'[^\w\s]', '', k["kelime"]).lower(), **k} for k in stt_kelimeler]
    def lev_sim(a, b):
        if a == b: return 1.0
        if not a or not b: return 0.0
        m = [[0]*(len(b)+1) for _ in range(len(a)+1)]
        for i in range(len(a)+1): m[i][0] = i
        for j in range(len(b)+1): m[0][j] = j
        for i in range(1, len(a)+1):
            for j in range(1, len(b)+1):
                m[i][j] = m[i-1][j-1] if a[i-1]==b[j-1] else min(m[i-1][j-1],m[i][j-1],m[i-1][j])+1
        return (max(len(a),len(b))-m[len(a)][len(b)])/max(len(a),len(b))
    sonuc = []; stt_idx = 0
    for ref_w in ref_words:
        best_sim = 0; best_j = -1; best_conf = 0
        for j in range(stt_idx, min(len(stt_words), stt_idx + 15)):
            s = lev_sim(ref_w, stt_words[j]["kelime_temiz"])
            if s > best_sim: best_sim = s; best_j = j; best_conf = stt_words[j]["confidence"]
        if best_sim >= 0.85 and best_conf >= 0.5:
            sonuc.append({"kelime": ref_w, "durum": "dogru", "confidence": best_conf, "stt": stt_words[best_j]["kelime"]}); stt_idx = best_j + 1
        elif best_sim >= 0.4:
            sonuc.append({"kelime": ref_w, "durum": "hatali", "confidence": best_conf, "stt": stt_words[best_j]["kelime"]}); stt_idx = best_j + 1
        else:
            sonuc.append({"kelime": ref_w, "durum": "atlanmis", "confidence": 0, "stt": ""})
    dogru = sum(1 for s in sonuc if s["durum"] == "dogru"); toplam = len(ref_words)
    return {"kelimeler": sonuc, "toplam": toplam, "dogru": dogru, "hatali": sum(1 for s in sonuc if s["durum"] == "hatali"), "atlanmis": sum(1 for s in sonuc if s["durum"] == "atlanmis"), "dogruluk_yuzdesi": round(dogru / toplam * 100, 1) if toplam > 0 else 0}

def prozodi_analiz(ses_dosya_yolu):
    import parselmouth, librosa
    ses = parselmouth.Sound(ses_dosya_yolu); toplam_sure = ses.get_total_duration()
    pitch = ses.to_pitch(); pv = pitch.selected_array['frequency']; pv = pv[pv > 0]
    pitch_data = {"ortalama_hz": round(float(np.mean(pv)),1), "std_hz": round(float(np.std(pv)),1), "monotonluk": "Monoton" if float(np.std(pv))<20 else "Normal" if float(np.std(pv))<40 else "Canli"} if len(pv) > 0 else {"ortalama_hz":0,"std_hz":0,"monotonluk":"Olculemedi"}
    intensity = ses.to_intensity(); iv = intensity.values.T.flatten(); iv = iv[iv>40]
    enerji_data = {"ortalama_db": round(float(np.mean(iv)),1), "std_db": round(float(np.std(iv)),1), "vurgu_durumu": "Yetersiz" if float(np.std(iv))<3.5 else "Normal" if float(np.std(iv))<7 else "Guclu"} if len(iv)>0 else {"ortalama_db":0,"std_db":0,"vurgu_durumu":"Olculemedi"}
    y, sr = librosa.load(ses_dosya_yolu, sr=None); sk = librosa.effects.split(y, top_db=30)
    ks = 0; dur_count = 0; uzun = 0
    for i, k in enumerate(sk):
        ks += (k[1]-k[0])/sr
        if i > 0:
            ds = k[0]/sr - sk[i-1][1]/sr
            if ds > 0.15: dur_count += 1; uzun += 1 if ds > 1.0 else 0
    ss = toplam_sure - ks; so = (ss/toplam_sure)*100 if toplam_sure>0 else 0
    return {"pitch": pitch_data, "enerji": enerji_data, "sure": {"toplam_sn": round(toplam_sure,1), "konusma_sn": round(ks,1), "sessizlik_sn": round(ss,1), "sessizlik_orani": round(so,1), "duraksama_sayisi": dur_count, "uzun_duraksama": uzun, "akicilik": "Akici" if so<20 else "Normal" if so<35 else "Kesik kesik"}}

def claude_analiz(referans_metin, karsilastirma, prozodi, sinif, ogrenci_adi, asama=1):
    import anthropic
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    sr = SINIF_REF.get(sinif, SINIF_REF["2"])
    hatali_str = ", ".join([f"{k['kelime']}({k['stt']},c:{k['confidence']})" for k in karsilastirma["kelimeler"] if k["durum"] != "dogru"][:15])
    prompt = f"""{sinif}. sinif "{ogrenci_adi}" okuma yapti (Asama {asama}).
Toplam:{karsilastirma['toplam']}, Dogru:{karsilastirma['dogru']}, Hatali:{karsilastirma['hatali']}, Atlanan:{karsilastirma['atlanmis']}, %{karsilastirma['dogruluk_yuzdesi']}
Hatalilar: {hatali_str or 'Yok'}
Pitch {prozodi['pitch']['ortalama_hz']}Hz std:{prozodi['pitch']['std_hz']}Hz ({prozodi['pitch']['monotonluk']}), Enerji {prozodi['enerji']['std_db']}dB ({prozodi['enerji']['vurgu_durumu']}), Sure {prozodi['sure']['toplam_sn']}sn, Sessizlik %{prozodi['sure']['sessizlik_orani']}, Duraksama {prozodi['sure']['duraksama_sayisi']}({prozodi['sure']['uzun_duraksama']}uzun)
REF WPM:{sr['wpm_min']}-{sr['wpm_max']}, min:%{sr['dogruluk_min']}. Hesapla WPM={karsilastirma['dogru']}x60/{prozodi['sure']['konusma_sn']}
SADECE JSON:
{{"genel_skor":0-100,"akicilik_skoru":0-100,"prozodi_skoru":0-100,"seviye":"Baslangic/Gelisen/Yeterli/Ileri","wpm":sayi,"kaba_degerlendirme":{{"toplam":{karsilastirma['toplam']},"dogru":{karsilastirma['dogru']},"yanlis":{karsilastirma['hatali']},"atlanan":{karsilastirma['atlanmis']},"wpm":sayi}},"prozodik_olcek":[{{"madde":"Duygu yansitma","puan":0-4}},{{"madde":"Konusma dili","puan":0-4}},{{"madde":"Vurgu tonlama","puan":0-4}},{{"madde":"Noktalama uyumu","puan":0-4}},{{"madde":"Anlam vurgusu","puan":0-4}},{{"madde":"Uygun bekleme","puan":0-4}},{{"madde":"Akici okuma","puan":0-4}},{{"madde":"Anlamli gruplama","puan":0-4}}],"prozodik_toplam":0-32,"akicilik_ozeti":"1 cumle","prozodi_ozeti":"1 cumle","hatali_kelimeler":[{{"kelime":"x","okunan":"y","dogru_telaffuz":"he-ce","anlami":"kisa","hata_turu":"substitusyon/omisyon/metatez"}}],"guclu_yonler":["2"],"gelisim_alanlari":["2"],"oneriler":["3"]}}"""
    response = client.messages.create(model="claude-sonnet-4-20250514", max_tokens=2500, messages=[{"role": "user", "content": prompt}])
    raw = response.content[0].text.strip()
    for prefix in ["```json", "```"]:
        if raw.startswith(prefix): raw = raw[len(prefix):]
    if raw.endswith("```"): raw = raw[:-3]
    try: return json.loads(raw.strip())
    except: return {"hata": "Parse hatasi"}


# ─── AUTH ROUTES ───
@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('admin_panel' if current_user.role == 'admin' else 'student_panel'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    if request.method == 'POST':
        user = User.query.filter_by(username=request.form['username']).first()
        if user and user.check_password(request.form['password']):
            login_user(user)
            return redirect(url_for('index'))
        flash('Kullanici adi veya sifre hatali', 'error')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))


# ─── ADMIN ROUTES ───
@app.route('/admin')
@login_required
def admin_panel():
    if current_user.role != 'admin': return redirect(url_for('student_panel'))
    students = User.query.filter_by(role='student').order_by(User.name).all()
    # Her ogrenci icin son skor
    student_data = []
    for s in students:
        last = ReadingSession.query.filter_by(user_id=s.id).order_by(ReadingSession.created_at.desc()).first()
        total = ReadingSession.query.filter_by(user_id=s.id).count()
        student_data.append({"user": s, "last": last, "total": total})
    return render_template('admin.html', students=student_data)

@app.route('/admin/ogrenci-ekle', methods=['POST'])
@login_required
def ogrenci_ekle():
    if current_user.role != 'admin': return redirect(url_for('index'))
    username = request.form.get('username', '').strip()
    password = request.form.get('password', '').strip()
    name = request.form.get('name', '').strip()
    sinif = request.form.get('sinif', '2')
    if not username or not password or not name:
        flash('Tum alanlari doldurun', 'error'); return redirect(url_for('admin_panel'))
    if User.query.filter_by(username=username).first():
        flash('Bu kullanici adi zaten var', 'error'); return redirect(url_for('admin_panel'))
    user = User(username=username, name=name, sinif=sinif, role='student')
    user.set_password(password)
    db.session.add(user); db.session.commit()
    flash(f'{name} eklendi', 'success')
    return redirect(url_for('admin_panel'))

@app.route('/admin/ogrenci-sil/<int:uid>', methods=['POST'])
@login_required
def ogrenci_sil(uid):
    if current_user.role != 'admin': return redirect(url_for('index'))
    user = User.query.get_or_404(uid)
    ReadingSession.query.filter_by(user_id=uid).delete()
    db.session.delete(user); db.session.commit()
    flash(f'{user.name} silindi', 'success')
    return redirect(url_for('admin_panel'))

@app.route('/admin/ogrenci/<int:uid>')
@login_required
def ogrenci_detay(uid):
    if current_user.role != 'admin': return redirect(url_for('index'))
    student = User.query.get_or_404(uid)
    sessions = ReadingSession.query.filter_by(user_id=uid).order_by(ReadingSession.created_at.desc()).all()
    return render_template('ogrenci_detay.html', student=student, sessions=sessions)


# ─── STUDENT ROUTES ───
@app.route('/ogrenci')
@login_required
def student_panel():
    if current_user.role == 'admin': return redirect(url_for('admin_panel'))
    sessions = ReadingSession.query.filter_by(user_id=current_user.id).order_by(ReadingSession.created_at.desc()).limit(20).all()
    return render_template('student.html', sessions=sessions)


# ─── API ROUTES ───
@app.route('/metin-olustur', methods=['POST'])
@login_required
def metin_olustur():
    import anthropic
    sinif = request.json.get('sinif', current_user.sinif or '2')
    sr = SINIF_REF.get(sinif, SINIF_REF["2"])
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    response = client.messages.create(model="claude-sonnet-4-20250514", max_tokens=500, messages=[{"role": "user", "content": f"{sinif}. sinif Turkce okuma metni yaz. {sr['kelime_sayisi']} kelime. Cocuklara uygun, ilgi cekici, her seferinde farkli konu. Sadece metni yaz."}])
    return jsonify({"metin": response.content[0].text.strip()})

@app.route('/asama-metin', methods=['POST'])
@login_required
def asama_metin():
    import anthropic
    sinif = request.json.get('sinif', current_user.sinif or '2')
    hatali = request.json.get('hatali_kelimeler', [])
    sr = SINIF_REF.get(sinif, SINIF_REF["2"])
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    response = client.messages.create(model="claude-sonnet-4-20250514", max_tokens=500, messages=[{"role": "user", "content": f"{sinif}. sinif okuma metni yaz. MUTLAKA su kelimeleri icersin: {', '.join(hatali[:10])}. {sr['kelime_sayisi']} kelime. Sadece metni yaz."}])
    return jsonify({"metin": response.content[0].text.strip()})

@app.route('/analiz', methods=['POST'])
@login_required
def analiz():
    try:
        if 'ses_dosyasi' not in request.files: return jsonify({"hata": "Ses dosyasi yuklenmedi"}), 400
        sf = request.files['ses_dosyasi']; rm = request.form.get('referans_metin', '').strip()
        asama = int(request.form.get('asama', '1'))
        if not rm: return jsonify({"hata": "Referans metin girilmedi"}), 400
        sx = os.path.splitext(sf.filename)[1] or '.wav'
        with tempfile.NamedTemporaryFile(delete=False, suffix=sx) as tmp: sf.save(tmp.name); tp = tmp.name
        wp = tp + "_c.wav"
        try:
            AudioSegment.from_file(tp).set_frame_rate(16000).set_channels(1).set_sample_width(2).export(wp, format="wav")
            transkript, stt_kelimeler = google_stt_analiz(wp)
            if not transkript or len(stt_kelimeler) < 3:
                return jsonify({"basarili": False, "yeniden_kayit": True, "sebep": "Konusma algilanamadi."})
            karsilastirma = kelime_karsilastir(rm, stt_kelimeler)
            pz = prozodi_analiz(wp)
            rapor = claude_analiz(rm, karsilastirma, pz, current_user.sinif or '2', current_user.name, asama)
            hatali_liste = list(set([k["kelime"] for k in karsilastirma["kelimeler"] if k["durum"] != "dogru"]))
            tamamlandi = karsilastirma["dogruluk_yuzdesi"] >= 95

            # DB'ye kaydet
            session_rec = ReadingSession(
                user_id=current_user.id, asama=asama, referans_metin=rm, transkript=transkript,
                dogruluk=karsilastirma["dogruluk_yuzdesi"],
                akicilik=rapor.get("akicilik_skoru", 0), prozodi=rapor.get("prozodi_skoru", 0),
                genel_skor=rapor.get("genel_skor", 0), wpm=rapor.get("wpm", 0),
                hatali_json=json.dumps(hatali_liste, ensure_ascii=False),
                rapor_json=json.dumps(rapor, ensure_ascii=False),
                tamamlandi=tamamlandi
            )
            db.session.add(session_rec); db.session.commit()

            return jsonify({
                "basarili": True, "ogrenci_adi": current_user.name, "sinif": current_user.sinif,
                "asama": asama, "referans_metin": rm, "transkript": transkript,
                "stt_kelimeler": stt_kelimeler, "karsilastirma": karsilastirma, "prozodi": pz,
                "rapor": rapor, "hatali_liste": hatali_liste, "tamamlandi": tamamlandi
            })
        finally:
            if os.path.exists(tp): os.unlink(tp)
            if os.path.exists(wp): os.unlink(wp)
    except Exception as e:
        import traceback; traceback.print_exc(); return jsonify({"hata": str(e)}), 500

@app.route('/api/progress/<int:uid>')
@login_required
def api_progress(uid):
    if current_user.role != 'admin' and current_user.id != uid:
        return jsonify({"hata": "Yetkisiz"}), 403
    sessions = ReadingSession.query.filter_by(user_id=uid).order_by(ReadingSession.created_at).all()
    return jsonify([{"id": s.id, "tarih": s.created_at.strftime("%d.%m"), "dogruluk": s.dogruluk, "akicilik": s.akicilik, "prozodi": s.prozodi, "genel": s.genel_skor, "wpm": s.wpm, "asama": s.asama, "tamamlandi": s.tamamlandi} for s in sessions])


# ─── INIT ───
with app.app_context():
    db.create_all()
    if not User.query.filter_by(role='admin').first():
        admin = User(username='admin', name='Admin', role='admin', sinif='0')
        admin.set_password('admin123')
        db.session.add(admin); db.session.commit()
        print("  Admin olusturuldu: admin / admin123")

if __name__ == '__main__':
    app.run(debug=True, port=8080, host='0.0.0.0')
