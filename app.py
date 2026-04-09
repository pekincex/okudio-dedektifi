"""
Okudio Okuma Dedektifi — Faz 4
Dual STT (Google+Whisper) + Kelime Bazli Ses Analizi + Radar Chart
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
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

app = Flask(__name__, template_folder='templates/templates')
app.config['SECRET_KEY'] = os.environ.get("SECRET_KEY", "okudio-2025")
db_url = os.environ.get("DATABASE_URL", "sqlite:///okudio.db")
if db_url.startswith("postgres://"): db_url = db_url.replace("postgres://", "postgresql://", 1)
app.config['SQLALCHEMY_DATABASE_URI'] = db_url
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

SINIF_REF = {
    "2": {"wpm_min":40,"wpm_max":60,"dogruluk_min":95,"kelime":60,"zorluk":"Cok basit, 4-6 kelimelik cumleler, gunluk kelimeler"},
    "3": {"wpm_min":70,"wpm_max":90,"dogruluk_min":95,"kelime":70,"zorluk":"Basit, 6-8 kelimelik cumleler, sifatlar"},
    "4": {"wpm_min":90,"wpm_max":110,"dogruluk_min":97,"kelime":80,"zorluk":"Orta, bilesik cumleler, 8-10 kelime"},
    "5": {"wpm_min":110,"wpm_max":130,"dogruluk_min":98,"kelime":90,"zorluk":"Karisik cumleler, soyut kavramlar"},
    "6": {"wpm_min":130,"wpm_max":145,"dogruluk_min":99,"kelime":100,"zorluk":"Uzun cumleler, akademik kelimeler"},
    "7": {"wpm_min":145,"wpm_max":160,"dogruluk_min":99,"kelime":110,"zorluk":"Edebi dil, mecazlar"},
}

# ═══ MODELS ═══
class User(UserMixin, db.Model):
    __tablename__='users'
    id=db.Column(db.Integer,primary_key=True);username=db.Column(db.String(80),unique=True,nullable=False)
    password_hash=db.Column(db.String(256),nullable=False);role=db.Column(db.String(10),default='student')
    name=db.Column(db.String(120),nullable=False);sinif=db.Column(db.String(2),default='2')
    created_at=db.Column(db.DateTime,default=datetime.datetime.utcnow)
    sessions=db.relationship('ReadingSession',backref='user',lazy=True)
    def set_password(self,pw):self.password_hash=generate_password_hash(pw)
    def check_password(self,pw):return check_password_hash(self.password_hash,pw)

class ReadingSession(db.Model):
    __tablename__='reading_sessions'
    id=db.Column(db.Integer,primary_key=True);user_id=db.Column(db.Integer,db.ForeignKey('users.id'),nullable=False)
    asama=db.Column(db.Integer,default=1);referans_metin=db.Column(db.Text);transkript=db.Column(db.Text)
    dogruluk=db.Column(db.Float,default=0);akicilik=db.Column(db.Float,default=0);prozodi=db.Column(db.Float,default=0)
    genel_skor=db.Column(db.Float,default=0);wpm=db.Column(db.Integer,default=0)
    hatali_json=db.Column(db.Text,default='[]');rapor_json=db.Column(db.Text,default='{}')
    radar_json=db.Column(db.Text,default='{}')
    tamamlandi=db.Column(db.Boolean,default=False);created_at=db.Column(db.DateTime,default=datetime.datetime.utcnow)

@login_manager.user_loader
def load_user(id):return User.query.get(int(id))


# ═══ GOOGLE STT ═══
def google_stt(wav_path, referans=""):
    with open(wav_path,"rb") as f: audio_b64=base64.b64encode(f.read()).decode("utf-8")
    phrases=list(set(re.sub(r'[^\w\s]','',referans).lower().split()))[:500] if referans else []
    config={"encoding":"LINEAR16","sampleRateHertz":16000,"languageCode":"tr-TR","enableWordTimeOffsets":True,"enableWordConfidence":True,"model":"default","useEnhanced":True}
    if phrases: config["speechContexts"]=[{"phrases":phrases,"boost":15}]
    resp=requests.post(f"https://speech.googleapis.com/v1p1beta1/speech:recognize?key={GOOGLE_API_KEY}",json={"config":config,"audio":{"content":audio_b64}},timeout=120)
    data=resp.json()
    if "error" in data or "results" not in data: return "",[]
    tr="";kl=[]
    for r in data["results"]:
        alt=r["alternatives"][0];tr+=alt.get("transcript","")+" "
        for w in alt.get("words",[]):
            kl.append({"kelime":w["word"],"bas":round(float(w.get("startTime","0s").replace("s","")),2),"bit":round(float(w.get("endTime","0s").replace("s","")),2),"conf":round(w.get("confidence",0),3)})
    return tr.strip(),kl


# ═══ WHISPER STT ═══
def whisper_stt(wav_path):
    from openai import OpenAI
    client=OpenAI(api_key=OPENAI_API_KEY)
    with open(wav_path,"rb") as f:
        resp=client.audio.transcriptions.create(model="whisper-1",file=f,language="tr",response_format="verbose_json",timestamp_granularities=["word"],
            prompt="Ogrenci sesli okuma yapiyor. Duraksama seslerini oldugu gibi yaz.")
    kl=[]
    if hasattr(resp,'words') and resp.words:
        for w in resp.words:
            kl.append({"kelime":w.word,"bas":round(w.start,2),"bit":round(w.end,2)})
    return resp.text,kl


# ═══ DUAL STT MERGE ═══
def dual_stt_analiz(wav_path, referans):
    """Google + Whisper calistir, ikisinin ciktisini birlestir"""
    print("  STT: Google (speech hints)...")
    g_tr,g_kl=google_stt(wav_path,referans)
    print(f"  Google: {len(g_kl)} kelime")

    w_tr,w_kl="",[]
    if OPENAI_API_KEY:
        print("  STT: Whisper...")
        try:
            w_tr,w_kl=whisper_stt(wav_path)
            print(f"  Whisper: {len(w_kl)} kelime")
        except Exception as e:
            print(f"  Whisper hata: {e}")

    # Merge: Google primary, Whisper secondary
    merged_tr = g_tr or w_tr
    return merged_tr, g_kl, w_kl


# ═══ KELIME KARSILASTIRMA (Dual) ═══
def lev_sim(a,b):
    if a==b:return 1.0
    if not a or not b:return 0.0
    la,lb=len(a),len(b)
    if la>20 or lb>20:return 1.0 if a[:5]==b[:5] else 0.0
    m=[[0]*(lb+1) for _ in range(la+1)]
    for i in range(la+1):m[i][0]=i
    for j in range(lb+1):m[0][j]=j
    for i in range(1,la+1):
        for j in range(1,lb+1):
            m[i][j]=m[i-1][j-1] if a[i-1]==b[j-1] else min(m[i-1][j-1],m[i][j-1],m[i-1][j])+1
    return(max(la,lb)-m[la][lb])/max(la,lb)

def kelime_karsilastir(referans, g_kl, w_kl):
    ref_words=re.sub(r'[^\w\s]','',referans).lower().split()
    gc=[re.sub(r'[^\w\s]','',k["kelime"]).lower() for k in g_kl]
    wc=[re.sub(r'[^\w\s]','',k["kelime"]).lower() for k in w_kl]
    g_used=[False]*len(gc);w_used=[False]*len(wc)
    sonuc=[None]*len(ref_words)

    # Pass 1: Sequential match against Google
    gi=0
    for i,rw in enumerate(ref_words):
        bs=0;bj=-1
        for j in range(gi,min(len(gc),gi+20)):
            if g_used[j]:continue
            s=lev_sim(rw,gc[j])
            if s>bs:bs=s;bj=j
        if bs>=0.8 and bj>=0:
            g_used[bj]=True
            sonuc[i]={"kelime":rw,"durum":"dogru","conf":g_kl[bj]["conf"],"stt":g_kl[bj]["kelime"],"bas":g_kl[bj]["bas"],"bit":g_kl[bj]["bit"],"kaynak":"google"}
            gi=bj+1
        elif bs>=0.4 and bj>=0:
            g_used[bj]=True
            sonuc[i]={"kelime":rw,"durum":"hatali","conf":g_kl[bj]["conf"],"stt":g_kl[bj]["kelime"],"bas":g_kl[bj]["bas"],"bit":g_kl[bj]["bit"],"kaynak":"google"}
            gi=bj+1

    # Pass 2: Unmatched → check Whisper
    for i,rw in enumerate(ref_words):
        if sonuc[i] is not None:continue
        bs=0;bj=-1
        for j in range(len(wc)):
            if w_used[j]:continue
            s=lev_sim(rw,wc[j])
            if s>bs:bs=s;bj=j
        if bs>=0.75 and bj>=0:
            w_used[bj]=True
            d="dogru" if bs>=0.9 else "hatali"
            sonuc[i]={"kelime":rw,"durum":d,"conf":0.7,"stt":w_kl[bj]["kelime"],"bas":w_kl[bj]["bas"],"bit":w_kl[bj]["bit"],"kaynak":"whisper"}
        else:
            # Pass 3: Global search in Google
            bs2=0;bj2=-1
            for j in range(len(gc)):
                if g_used[j]:continue
                s=lev_sim(rw,gc[j])
                if s>bs2:bs2=s;bj2=j
            if bs2>=0.7 and bj2>=0:
                g_used[bj2]=True
                sonuc[i]={"kelime":rw,"durum":"dogru" if bs2>=0.9 else "hatali","conf":g_kl[bj2]["conf"],"stt":g_kl[bj2]["kelime"],"bas":g_kl[bj2]["bas"],"bit":g_kl[bj2]["bit"],"kaynak":"google-global"}
            else:
                sonuc[i]={"kelime":rw,"durum":"atlanmis","conf":0,"stt":"","bas":0,"bit":0,"kaynak":""}

    dogru=sum(1 for s in sonuc if s["durum"]=="dogru")
    toplam=len(ref_words)
    return {"kelimeler":sonuc,"toplam":toplam,"dogru":dogru,"hatali":sum(1 for s in sonuc if s["durum"]=="hatali"),"atlanmis":sum(1 for s in sonuc if s["durum"]=="atlanmis"),"dogruluk_yuzdesi":round(dogru/toplam*100,1) if toplam>0 else 0}


# ═══ KELIME BAZLI SES ANALİZİ ═══
def kelime_ses_analizi(wav_path, karsilastirma):
    """Her kelimenin ses segmentini analiz et"""
    import parselmouth
    ses=parselmouth.Sound(wav_path)
    kelime_detay=[]

    for k in karsilastirma["kelimeler"]:
        if k["bas"]==0 and k["bit"]==0:
            kelime_detay.append({"kelime":k["kelime"],"sure_ms":0,"pitch_hz":0,"enerji_db":0,"durum":k["durum"],"anomali":""})
            continue

        sure_ms=round((k["bit"]-k["bas"])*1000)
        beklenen_ms=max(200,len(k["kelime"])*120)  # tahmini beklenen sure

        # Kelime segmentini al
        try:
            seg=ses.extract_part(from_time=k["bas"],to_time=k["bit"],preserve_times=False)
            pitch=seg.to_pitch()
            pv=pitch.selected_array['frequency'];pv=pv[pv>0]
            p_hz=round(float(np.mean(pv)),1) if len(pv)>0 else 0
            intensity=seg.to_intensity()
            iv=intensity.values.T.flatten();iv=iv[iv>30]
            e_db=round(float(np.mean(iv)),1) if len(iv)>0 else 0
        except:
            p_hz=0;e_db=0

        # Anomali tespiti
        anomali=""
        if sure_ms > beklenen_ms * 1.8: anomali="heceleme"
        elif sure_ms > beklenen_ms * 1.4: anomali="yavas"
        elif sure_ms < beklenen_ms * 0.5 and sure_ms > 50: anomali="hizli"

        kelime_detay.append({"kelime":k["kelime"],"sure_ms":sure_ms,"pitch_hz":p_hz,"enerji_db":e_db,"durum":k["durum"],"anomali":anomali,"stt":k.get("stt","")})

    return kelime_detay


# ═══ PROZODİ (global) ═══
def prozodi_analiz(wav_path):
    import parselmouth,librosa
    ses=parselmouth.Sound(wav_path);ts=ses.get_total_duration()
    pitch=ses.to_pitch();pv=pitch.selected_array['frequency'];pv=pv[pv>0]
    pd={"ort":round(float(np.mean(pv)),1),"std":round(float(np.std(pv)),1),"mon":"Monoton" if float(np.std(pv))<20 else "Normal" if float(np.std(pv))<40 else "Canli"} if len(pv)>0 else {"ort":0,"std":0,"mon":"?"}
    intensity=ses.to_intensity();iv=intensity.values.T.flatten();iv=iv[iv>40]
    ed={"ort":round(float(np.mean(iv)),1),"std":round(float(np.std(iv)),1),"vur":"Yetersiz" if float(np.std(iv))<3.5 else "Normal" if float(np.std(iv))<7 else "Guclu"} if len(iv)>0 else {"ort":0,"std":0,"vur":"?"}
    y,sr=librosa.load(wav_path,sr=None);sk=librosa.effects.split(y,top_db=30)
    ks=0;dc=0;ul=0
    for i,k in enumerate(sk):
        ks+=(k[1]-k[0])/sr
        if i>0:
            ds=k[0]/sr-sk[i-1][1]/sr
            if ds>0.15:dc+=1;ul+=1 if ds>1.0 else 0
    ss=ts-ks;so=(ss/ts)*100 if ts>0 else 0
    # Radar icin normalize skorlar (0-100)
    pitch_skor=min(100,max(0,int(pd["std"]*2.5))) # std 0-40 → 0-100
    vurgu_skor=min(100,max(0,int(ed["std"]*14)))   # std 0-7 → 0-100
    akicilik_skor=max(0,int(100-so*2))               # sessizlik %0→100, %50→0
    return {"pitch":pd,"enerji":ed,"sure":{"toplam":round(ts,1),"konusma":round(ks,1),"sessizlik":round(so,1),"duraksama":dc,"uzun":ul,"akicilik":"Akici" if so<20 else "Normal" if so<35 else "Kesik"},"radar_pitch":pitch_skor,"radar_vurgu":vurgu_skor,"radar_akicilik":akicilik_skor}


# ═══ CLAUDE ═══
def claude_analiz(ref,kar,pz,sinif,ad,kelime_detay,asama=1):
    import anthropic
    client=anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    sr=SINIF_REF.get(sinif,SINIF_REF["2"])
    hl=", ".join([f"{k['kelime']}({k['stt']},c:{k['conf']:.1f})" for k in kar["kelimeler"] if k["durum"]!="dogru"][:12])
    anomaliler=[f"{k['kelime']}({k['anomali']},{k['sure_ms']}ms)" for k in kelime_detay if k["anomali"]][:8]
    prompt=f"""{sinif}. sinif "{ad}" okuma (Asama {asama}).
T:{kar['toplam']} D:{kar['dogru']} H:{kar['hatali']} A:{kar['atlanmis']} %{kar['dogruluk_yuzdesi']}
Hatali: {hl or 'Yok'}
Anomaliler: {', '.join(anomaliler) or 'Yok'}
Pitch:{pz['pitch']['ort']}Hz std:{pz['pitch']['std']}({pz['pitch']['mon']}), E:{pz['enerji']['std']}dB({pz['enerji']['vur']}), S:{pz['sure']['toplam']}sn, Sess:%{pz['sure']['sessizlik']}, Dur:{pz['sure']['duraksama']}({pz['sure']['uzun']}uz)
REF WPM:{sr['wpm_min']}-{sr['wpm_max']} min%{sr.get('dogruluk_min',95)} WPM={kar['dogru']}x60/{pz['sure']['konusma']}
SADECE JSON:
{{"genel_skor":0-100,"akicilik_skoru":0-100,"prozodi_skoru":0-100,"seviye":"Baslangic/Gelisen/Yeterli/Ileri","wpm":N,"kaba":{{"toplam":{kar['toplam']},"dogru":{kar['dogru']},"yanlis":{kar['hatali']},"atlanan":{kar['atlanmis']},"wpm":N}},"prozodik_olcek":[{{"m":"Duygu","p":0-4}},{{"m":"Konusma","p":0-4}},{{"m":"Vurgu","p":0-4}},{{"m":"Noktalama","p":0-4}},{{"m":"Anlam","p":0-4}},{{"m":"Bekleme","p":0-4}},{{"m":"Akici","p":0-4}},{{"m":"Gruplama","p":0-4}}],"pt":0-32,"ao":"1 cumle","po":"1 cumle","hk":[{{"k":"x","o":"y","t":"he-ce","a":"anlam","ht":"sub/omi/met"}}],"gy":["2"],"ga":["2"],"on":["3"]}}"""
    resp=client.messages.create(model="claude-sonnet-4-20250514",max_tokens=2000,messages=[{"role":"user","content":prompt}])
    raw=resp.content[0].text.strip()
    for p in ["```json","```"]:
        if raw.startswith(p):raw=raw[len(p):]
    if raw.endswith("```"):raw=raw[:-3]
    try:return json.loads(raw.strip())
    except:return {"hata":"Parse"}


# ═══ OPENAI METİN ÜRETİMİ ═══
def openai_metin_uret(sinif, hatali_kelimeler=None):
    from openai import OpenAI
    client=OpenAI(api_key=OPENAI_API_KEY)
    sr=SINIF_REF.get(sinif,SINIF_REF["2"])
    if hatali_kelimeler:
        prompt=f"{sinif}. sinif Turkce okuma metni yaz. MUTLAKA su kelimeleri icersin: {', '.join(hatali_kelimeler[:10])}. Tam olarak {sr['kelime']} kelime. {sr['zorluk']}. BASLIK YAZMA. Sadece metni yaz."
    else:
        prompt=f"{sinif}. sinif Turkce okuma metni yaz. Tam olarak {sr['kelime']} kelime. {sr['zorluk']}. Cocuklara uygun ilgi cekici konu. Her seferinde FARKLI konu. BASLIK YAZMA. Sadece metni yaz."
    resp=client.chat.completions.create(model="gpt-4o-mini",messages=[{"role":"user","content":prompt}],max_tokens=500)
    return resp.choices[0].message.content.strip()


# ═══ ROUTES ═══
@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('admin_panel' if current_user.role=='admin' else 'student_panel'))
    return redirect(url_for('login'))

@app.route('/login',methods=['GET','POST'])
def login():
    if current_user.is_authenticated:return redirect(url_for('index'))
    if request.method=='POST':
        user=User.query.filter_by(username=request.form['username']).first()
        if user and user.check_password(request.form['password']):login_user(user);return redirect(url_for('index'))
        flash('Hatali giris','error')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():logout_user();return redirect(url_for('login'))

@app.route('/admin')
@login_required
def admin_panel():
    if current_user.role!='admin':return redirect(url_for('student_panel'))
    students=User.query.filter_by(role='student').order_by(User.name).all()
    sd=[{"user":s,"last":ReadingSession.query.filter_by(user_id=s.id).order_by(ReadingSession.created_at.desc()).first(),"total":ReadingSession.query.filter_by(user_id=s.id).count()} for s in students]
    return render_template('admin.html',students=sd)

@app.route('/admin/ogrenci-ekle',methods=['POST'])
@login_required
def ogrenci_ekle():
    if current_user.role!='admin':return redirect(url_for('index'))
    u=request.form.get('username','').strip();p=request.form.get('password','').strip();n=request.form.get('name','').strip();s=request.form.get('sinif','2')
    if not all([u,p,n]):flash('Eksik','error');return redirect(url_for('admin_panel'))
    if User.query.filter_by(username=u).first():flash('Var','error');return redirect(url_for('admin_panel'))
    user=User(username=u,name=n,sinif=s,role='student');user.set_password(p);db.session.add(user);db.session.commit();flash(f'{n} eklendi','success')
    return redirect(url_for('admin_panel'))

@app.route('/admin/ogrenci-sil/<int:uid>',methods=['POST'])
@login_required
def ogrenci_sil(uid):
    if current_user.role!='admin':return redirect(url_for('index'))
    u=User.query.get_or_404(uid);ReadingSession.query.filter_by(user_id=uid).delete();db.session.delete(u);db.session.commit();flash('Silindi','success')
    return redirect(url_for('admin_panel'))

@app.route('/admin/ogrenci/<int:uid>')
@login_required
def ogrenci_detay(uid):
    if current_user.role!='admin':return redirect(url_for('index'))
    return render_template('ogrenci_detay.html',student=User.query.get_or_404(uid),sessions=ReadingSession.query.filter_by(user_id=uid).order_by(ReadingSession.created_at.desc()).all())

@app.route('/admin/oturum/<int:sid>')
@login_required
def oturum_detay(sid):
    if current_user.role!='admin':return redirect(url_for('index'))
    sess=ReadingSession.query.get_or_404(sid);student=User.query.get(sess.user_id)
    return render_template('oturum_detay.html',sess=sess,student=student,rapor=json.loads(sess.rapor_json or '{}'),hatali=json.loads(sess.hatali_json or '[]'),radar=json.loads(sess.radar_json or '{}'))

@app.route('/ogrenci')
@login_required
def student_panel():
    if current_user.role=='admin':return redirect(url_for('admin_panel'))
    return render_template('student.html',sessions=ReadingSession.query.filter_by(user_id=current_user.id).order_by(ReadingSession.created_at.desc()).limit(20).all())

@app.route('/metin-olustur',methods=['POST'])
@login_required
def metin_olustur():
    sinif=request.json.get('sinif',current_user.sinif or '2')
    metin=openai_metin_uret(sinif)
    return jsonify({"metin":metin})

@app.route('/asama-metin',methods=['POST'])
@login_required
def asama_metin():
    sinif=request.json.get('sinif',current_user.sinif or '2')
    hatali=request.json.get('hatali_kelimeler',[])
    metin=openai_metin_uret(sinif,hatali)
    return jsonify({"metin":metin})

@app.route('/analiz',methods=['POST'])
@login_required
def analiz():
    try:
        if 'ses_dosyasi' not in request.files:return jsonify({"hata":"Dosya yok"}),400
        sf=request.files['ses_dosyasi'];rm=request.form.get('referans_metin','').strip()
        asama=int(request.form.get('asama','1'))
        if not rm:return jsonify({"hata":"Metin yok"}),400
        sx=os.path.splitext(sf.filename)[1] or '.wav'
        with tempfile.NamedTemporaryFile(delete=False,suffix=sx) as tmp:sf.save(tmp.name);tp=tmp.name
        wp=tp+"_c.wav"
        try:
            AudioSegment.from_file(tp).set_frame_rate(16000).set_channels(1).set_sample_width(2).export(wp,format="wav")
            print(f"\n  === {current_user.name} Asama {asama} ===")

            # K1: Dual STT
            tr,g_kl,w_kl=dual_stt_analiz(wp,rm)
            if not tr or (len(g_kl)<3 and len(w_kl)<3):
                return jsonify({"basarili":False,"yeniden_kayit":True,"sebep":"Ses algilanamadi."})

            # K2: Dual karsilastirma
            print("  Karsilastirma (3-pass)...")
            kar=kelime_karsilastir(rm,g_kl,w_kl)
            print(f"  %{kar['dogruluk_yuzdesi']} (D:{kar['dogru']} H:{kar['hatali']} A:{kar['atlanmis']})")

            # K3: Kelime bazli ses analizi
            print("  Kelime ses analizi...")
            kd=kelime_ses_analizi(wp,kar)
            anomali_count=sum(1 for k in kd if k["anomali"])
            print(f"  {anomali_count} anomali")

            # K4: Prozodi
            print("  Prozodi...")
            pz=prozodi_analiz(wp)

            # K5: Claude
            print("  Claude...")
            rp=claude_analiz(rm,kar,pz,current_user.sinif or '2',current_user.name,kd,asama)
            print("  OK!")

            hl=list(set([k["kelime"] for k in kar["kelimeler"] if k["durum"]!="dogru"]))
            tam=kar["dogruluk_yuzdesi"]>=95

            # Radar data
            sr=SINIF_REF.get(current_user.sinif,SINIF_REF["2"])
            wpm=rp.get("wpm",0)
            wpm_skor=min(100,max(0,int(wpm/sr["wpm_max"]*100)))
            radar={"dogruluk":kar["dogruluk_yuzdesi"],"akicilik":rp.get("akicilik_skoru",0),"prozodi":rp.get("prozodi_skoru",0),"hiz":wpm_skor,"vurgu":pz["radar_vurgu"],"tonlama":pz["radar_pitch"]}

            sess=ReadingSession(user_id=current_user.id,asama=asama,referans_metin=rm,transkript=tr,dogruluk=kar["dogruluk_yuzdesi"],akicilik=rp.get("akicilik_skoru",0),prozodi=rp.get("prozodi_skoru",0),genel_skor=rp.get("genel_skor",0),wpm=wpm,hatali_json=json.dumps(hl,ensure_ascii=False),rapor_json=json.dumps(rp,ensure_ascii=False),radar_json=json.dumps(radar),tamamlandi=tam)
            db.session.add(sess);db.session.commit()

            return jsonify({"basarili":True,"ogrenci_adi":current_user.name,"sinif":current_user.sinif,"asama":asama,"referans_metin":rm,"transkript":tr,"karsilastirma":kar,"kelime_detay":kd,"prozodi":pz,"rapor":rp,"radar":radar,"hatali_liste":hl,"tamamlandi":tam})
        finally:
            if os.path.exists(tp):os.unlink(tp)
            if os.path.exists(wp):os.unlink(wp)
    except Exception as e:
        import traceback;traceback.print_exc();return jsonify({"hata":str(e)}),500

with app.app_context():
    db.create_all()
    if not User.query.filter_by(role='admin').first():
        a=User(username='admin',name='Admin',role='admin',sinif='0');a.set_password('admin123')
        db.session.add(a);db.session.commit();print("  Admin: admin/admin123")

if __name__=='__main__':
    app.run(debug=True,port=8080,host='0.0.0.0')
