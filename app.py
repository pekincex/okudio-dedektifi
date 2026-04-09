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
    "2": {"wpm_min":40,"wpm_max":60,"dogruluk_min":95,"kelime":60,
        "cumle_uzunluk":"4-6 kelime","yeni_kelime_orani":"%5",
        "tema":"Hayvanlar, aile, okul, arkadaşlık, mevsimler",
        "yapi":"Net giris-gelisme-sonuc. Tek karakter, tek problem. Kronolojik siralama.",
        "kelime_turu":"Somut, günlük kelimeler. Eş anlamlı tekrarlar (sevinmek/mutlu olmak).",
        "ipucu":"Her yeni kelime bağlamda açıklanmalı. Betimleyici sıfatlar: büyük, küçük, güzel.",
        "zorluk":"Cok basit cumleler, tamamen bilinen kelimeler, %5 yeni kelime"},
    "3": {"wpm_min":70,"wpm_max":90,"dogruluk_min":95,"kelime":70,
        "cumle_uzunluk":"6-8 kelime","yeni_kelime_orani":"%7",
        "tema":"Doğa olayları, hayvan yaşamları, duygusal maceralar, basit bilim",
        "yapi":"Net hikaye haritası. Karakter motivasyonu belirgin. Basit diyaloglar.",
        "kelime_turu":"Duygu kelimeleri (endişe, cesaret, merak). Doğa sıfatları (pürüzsüz, devasa).",
        "ipucu":"Satır arası okuma fırsatı: 'Alt dudağı titredi' → üzgün çıkarımı. Sebep-sonuç.",
        "zorluk":"Basit birleşik cümleler, %7 yeni kelime, duygu ve doğa kelimeleri"},
    "4": {"wpm_min":90,"wpm_max":110,"dogruluk_min":97,"kelime":80,
        "cumle_uzunluk":"8-10 kelime","yeni_kelime_orani":"%8",
        "tema":"Buluşlar, keşifler, tarihsel maceralar, ekoloji, mizahi hikayeler",
        "yapi":"Çoklu karakter. Problem-çözüm yapısı belirgin. Kısa paragraflar.",
        "kelime_turu":"Akademik kelimeler (süreç, deney, gözlem). Mecazlar başlangıcı.",
        "ipucu":"Karakter motivasyonu sorgulatıcı. 'Sence neden böyle yaptı?' düşündürücü.",
        "zorluk":"Bileşik cümleler, diyaloglar, %8 yeni kelime, karşılaştırma yapıları"},
    "5": {"wpm_min":110,"wpm_max":130,"dogruluk_min":98,"kelime":90,
        "cumle_uzunluk":"10-12 kelime","yeni_kelime_orani":"%10",
        "tema":"Bilim-kurgu, tarih, coğrafya, sosyal problemler, biyografiler",
        "yapi":"Karmaşık olay örgüsü. Birden fazla problem. Çıkarım gerektiren boşluklar.",
        "kelime_turu":"Soyut kavramlar (adalet, özgürlük, empati). Deyimler bağlamda.",
        "ipucu":"Eş-zıt anlamlıların dansı: aynı kavram farklı kelimelerle tekrarlanmalı.",
        "zorluk":"Karmaşık yapılar, soyut kelimeler, %10 yeni kelime, çıkarım gerektiren"},
    "6": {"wpm_min":130,"wpm_max":145,"dogruluk_min":99,"kelime":100,
        "cumle_uzunluk":"12-14 kelime","yeni_kelime_orani":"%10",
        "tema":"Felsefi sorular, bilimsel keşifler, dünya kültürleri, çevre sorunları",
        "yapi":"Çoklu bakış açısı. Belirsizlik içeren durumlar. Tartışmaya açık sonuçlar.",
        "kelime_turu":"Akademik dil (hipotez, analiz, sentez). Mecaz ve deyimler bolca.",
        "ipucu":"Ana fikir bulma ve karşılaştırma becerileri. Eleştirel düşünme.",
        "zorluk":"Uzun birleşik cümleler, akademik dil, mecazlar, eleştirel düşünme"},
    "7": {"wpm_min":145,"wpm_max":160,"dogruluk_min":99,"kelime":110,
        "cumle_uzunluk":"14-16 kelime","yeni_kelime_orani":"%10",
        "tema":"Edebi metinler, bilimsel makaleler, felsefi hikayeler, toplumsal konular",
        "yapi":"Edebi dil. Betimleyici anlatım. İç monolog. Sembolik anlatım.",
        "kelime_turu":"Edebi kelimeler, atasözleri bağlamda, karmaşık sıfat tamlamaları.",
        "ipucu":"Dilin esnekliği: deyimlerin gerçek/mecaz anlam farkı. İroni.",
        "zorluk":"Edebi dil, sembolik anlatım, karmaşık yapılar, derin çıkarım"},
}

# Türkçe artikülasyon: yaygın ses hataları (S.O.D.A. çerçevesi)
TURK_SES_HATALARI = {
    "r→y": {"tip":"substitusyon","aciklama":"Rötasizm — /r/ yerine /y/","yas":"5-6 yaş altında normal","egzersiz":"dil_siklatma"},
    "r→l": {"tip":"substitusyon","aciklama":"Rötasizm — /r/ yerine /l/","yas":"5-6 yaş altında normal","egzersiz":"dil_siklatma"},
    "k→t": {"tip":"substitusyon","aciklama":"Öne Kayma — /k/ yerine /t/","yas":"4 yaş üstü anormal","egzersiz":"damak_temasi"},
    "g→d": {"tip":"substitusyon","aciklama":"Öne Kayma — /g/ yerine /d/","yas":"4 yaş üstü anormal","egzersiz":"damak_temasi"},
    "s→ş": {"tip":"substitusyon","aciklama":"Peltek /s/ — lateral kayma","yas":"Her yaşta müdahale","egzersiz":"ayna_tisla"},
    "ş→s": {"tip":"substitusyon","aciklama":"Ters peltek","yas":"Her yaşta müdahale","egzersiz":"ayna_tisla"},
    "ç→ş": {"tip":"substitusyon","aciklama":"Affrikat kaybı","yas":"4 yaş üstü anormal","egzersiz":"patlayici_ses"},
    "p→b": {"tip":"substitusyon","aciklama":"Sadalama hatası — tonsuz→tonlu","yas":"4 yaş üstü anormal","egzersiz":"nefes_kontrolu"},
    "t→d": {"tip":"substitusyon","aciklama":"Sadalama hatası — tonsuz→tonlu","yas":"4 yaş üstü anormal","egzersiz":"nefes_kontrolu"},
    "l→y": {"tip":"substitusyon","aciklama":"Lateral kayma — /l/ yerine /y/","yas":"5 yaş üstü anormal","egzersiz":"dil_ucu"},
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
    if phrases: config["speechContexts"]=[{"phrases":phrases,"boost":5}]
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

def dogru_esik(kelime):
    """Kisa kelimeler icin tam eslesme, uzun kelimeler icin daha toleransli"""
    n = len(kelime)
    if n <= 2: return 1.0    # "de","ve","bu" → tam eslesmeli
    if n <= 3: return 0.99   # "bir","iki" → neredeyse tam
    if n <= 5: return 0.90   # "okul","masa" → 1 harf tolerans
    return 0.88              # "kaydıraktan" → 1 harf tolerans

def kelime_karsilastir(referans, g_kl, w_kl):
    ref_words=re.sub(r'[^\w\s]','',referans).lower().split()
    gc=[re.sub(r'[^\w\s]','',k["kelime"]).lower() for k in g_kl]
    wc=[re.sub(r'[^\w\s]','',k["kelime"]).lower() for k in w_kl]
    g_used=[False]*len(gc);w_used=[False]*len(wc)
    sonuc=[None]*len(ref_words)

    # Pass 1: Sequential match against Whisper (UNBIASED - no speech hints)
    wi=0
    for i,rw in enumerate(ref_words):
        esik=dogru_esik(rw)
        bs=0;bj=-1
        for j in range(wi,min(len(wc),wi+20)):
            if w_used[j]:continue
            s=lev_sim(rw,wc[j])
            if s>bs:bs=s;bj=j
        if bs>=esik and bj>=0:
            w_used[bj]=True
            sonuc[i]={"kelime":rw,"durum":"dogru","conf":0.8,"stt":w_kl[bj]["kelime"],"bas":w_kl[bj]["bas"],"bit":w_kl[bj]["bit"],"kaynak":"whisper"}
            wi=bj+1
        elif bs>=0.5 and bj>=0:
            w_used[bj]=True
            sonuc[i]={"kelime":rw,"durum":"hatali","conf":0.5,"stt":w_kl[bj]["kelime"],"bas":w_kl[bj]["bas"],"bit":w_kl[bj]["bit"],"kaynak":"whisper"}
            wi=bj+1

    # Pass 2: Unmatched → check Google (speech hints, for identification only)
    for i,rw in enumerate(ref_words):
        if sonuc[i] is not None:continue
        esik=dogru_esik(rw)
        bs=0;bj=-1
        for j in range(len(gc)):
            if g_used[j]:continue
            s=lev_sim(rw,gc[j])
            if s>bs:bs=s;bj=j
        if bs>=esik and bj>=0:
            g_used[bj]=True
            sonuc[i]={"kelime":rw,"durum":"dogru","conf":g_kl[bj]["conf"],"stt":g_kl[bj]["kelime"],"bas":g_kl[bj]["bas"],"bit":g_kl[bj]["bit"],"kaynak":"google"}
        elif bs>=0.5 and bj>=0:
            g_used[bj]=True
            sonuc[i]={"kelime":rw,"durum":"hatali","conf":g_kl[bj]["conf"],"stt":g_kl[bj]["kelime"],"bas":g_kl[bj]["bas"],"bit":g_kl[bj]["bit"],"kaynak":"google"}
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


# ═══ ARTİKÜLASYON HATA SINIFLANDIRMA ═══
def hata_siniflandir(ref, stt):
    """Kelime çiftini analiz edip hata tipini belirle"""
    if not stt or stt.strip() == "":
        return {"tip": "omisyon", "alt_tip": "kelime_atlama", "detay": "Kelime tamamen atlandı", "ses": "", "pozisyon": ""}
    r = re.sub(r'[^\w]', '', ref).lower()
    s = re.sub(r'[^\w]', '', stt).lower()
    if r == s:
        return {"tip": "dogru", "alt_tip": "", "detay": "", "ses": "", "pozisyon": ""}

    # Metatez: aynı harfler, farklı sıra (toprak→torpak)
    if sorted(r) == sorted(s) and r != s:
        # Hangi pozisyonlar değişmiş
        degisen = [(i, r[i], s[i]) for i in range(len(r)) if r[i] != s[i]]
        return {"tip": "metatez", "alt_tip": "yer_degistirme", "detay": f"Harfler yer değiştirmiş: {'↔'.join([f'{d[1]}/{d[2]}' for d in degisen[:2]])}", "ses": "", "pozisyon": ""}

    # Karakter bazlı hizalama
    # Basit LCS ile farkları bul
    degisimler = []
    eksikler = []
    fazlalar = []

    # İki kelimeyi hizala
    ri = 0; si = 0
    while ri < len(r) and si < len(s):
        if r[ri] == s[si]:
            ri += 1; si += 1
        elif ri + 1 < len(r) and r[ri + 1] == s[si]:
            # ri'deki harf atlanmış (omisyon)
            poz = "bas" if ri == 0 else "son" if ri == len(r) - 1 else "orta"
            eksikler.append({"harf": r[ri], "poz": poz})
            ri += 1
        elif si + 1 < len(s) and r[ri] == s[si + 1]:
            # si'de fazla harf var (ekleme)
            fazlalar.append({"harf": s[si]})
            si += 1
        else:
            # Substitüsyon
            poz = "bas" if ri == 0 else "son" if ri == len(r) - 1 else "orta"
            degisimler.append({"kaynak": r[ri], "hedef": s[si], "poz": poz})
            ri += 1; si += 1

    # Kalan eksikler
    while ri < len(r):
        poz = "son" if ri == len(r) - 1 else "orta"
        eksikler.append({"harf": r[ri], "poz": poz})
        ri += 1
    while si < len(s):
        fazlalar.append({"harf": s[si]})
        si += 1

    # En belirgin hata tipini seç
    if degisimler:
        d = degisimler[0]
        ses_key = f"{d['kaynak']}→{d['hedef']}"
        # Bilinen Türkçe artikülasyon paterni mi?
        turk_pattern = TURK_SES_HATALARI.get(ses_key, None)
        alt_tip = turk_pattern["aciklama"] if turk_pattern else "ses_degistirme"
        return {"tip": "substitusyon", "alt_tip": alt_tip, "detay": f"'{d['kaynak']}' yerine '{d['hedef']}' ({'başta' if d['poz']=='bas' else 'ortada' if d['poz']=='orta' else 'sonda'})", "ses": ses_key, "pozisyon": d["poz"], "turk_pattern": turk_pattern is not None}
    elif eksikler:
        e = eksikler[0]
        return {"tip": "omisyon", "alt_tip": "ses_dusurme", "detay": f"'{e['harf']}' sesi düşürülmüş ({'başta' if e['poz']=='bas' else 'ortada' if e['poz']=='orta' else 'sonda'})", "ses": f"-{e['harf']}", "pozisyon": e["poz"]}
    elif fazlalar:
        f = fazlalar[0]
        return {"tip": "ekleme", "alt_tip": "ses_ekleme", "detay": f"'{f['harf']}' sesi eklenmiş", "ses": f"+{f['harf']}", "pozisyon": ""}
    else:
        return {"tip": "belirsiz", "alt_tip": "", "detay": f"{r} → {s}", "ses": "", "pozisyon": ""}


def hatalari_siniflandir(karsilastirma):
    """Tüm hatalı kelimeleri sınıflandır"""
    hatalar = []
    for k in karsilastirma["kelimeler"]:
        if k["durum"] == "dogru":
            continue
        sinif = hata_siniflandir(k["kelime"], k.get("stt", ""))
        hatalar.append({**sinif, "kelime": k["kelime"], "okunan": k.get("stt", ""), "conf": k.get("conf", 0)})
    return hatalar


# ═══ TEKRARLAYAN HATA PATTERN TESPİTİ ═══
def pattern_tespit(hata_listesi, gecmis_hatalar=None):
    """Tekrarlayan hata kalıplarını tespit et"""
    tip_sayac = {}
    ses_sayac = {}
    poz_sayac = {"bas": 0, "orta": 0, "son": 0}
    alt_tip_sayac = {}

    tum_hatalar = hata_listesi + (gecmis_hatalar or [])

    for h in tum_hatalar:
        tip = h.get("tip", "")
        if tip and tip != "dogru":
            tip_sayac[tip] = tip_sayac.get(tip, 0) + 1

        at = h.get("alt_tip", "")
        if at:
            alt_tip_sayac[at] = alt_tip_sayac.get(at, 0) + 1

        ses = h.get("ses", "")
        if ses:
            ses_sayac[ses] = ses_sayac.get(ses, 0) + 1

        poz = h.get("pozisyon", "")
        if poz in poz_sayac:
            poz_sayac[poz] += 1

    # Pattern'leri belirle
    patterns = []

    # Baskın hata tipi
    if tip_sayac:
        en_sik_tip = max(tip_sayac, key=tip_sayac.get)
        if tip_sayac[en_sik_tip] >= 2:
            tip_aciklama = {"substitusyon": "Ses değiştirme", "omisyon": "Ses düşürme", "metatez": "Harf karıştırma", "ekleme": "Ses ekleme"}
            patterns.append({"pattern": "baskin_hata", "tip": en_sik_tip, "aciklama": f"{tip_aciklama.get(en_sik_tip, en_sik_tip)} hatası baskın ({tip_sayac[en_sik_tip]} kez)", "onem": "yuksek"})

    # Tekrarlayan ses değişimleri
    for ses, sayi in ses_sayac.items():
        if sayi >= 2 and "→" in ses:
            k, h = ses.split("→")
            patterns.append({"pattern": "tekrar_ses", "tip": "substitusyon", "aciklama": f"'{k}' sesini sürekli '{h}' olarak okuyor ({sayi} kez)", "onem": "yuksek", "ses": ses})

    # Pozisyon pattern'i
    toplam_poz = sum(poz_sayac.values())
    if toplam_poz >= 3:
        en_sik_poz = max(poz_sayac, key=poz_sayac.get)
        if poz_sayac[en_sik_poz] >= toplam_poz * 0.6:
            poz_ad = {"bas": "kelimenin başındaki", "orta": "kelimenin ortasındaki", "son": "kelimenin sonundaki"}
            patterns.append({"pattern": "pozisyon", "tip": "pozisyon", "aciklama": f"Hatalar genellikle {poz_ad.get(en_sik_poz, '')} seslerde yoğunlaşıyor", "onem": "orta"})

    return {"tip_dagilimi": tip_sayac, "ses_degisim": ses_sayac, "pozisyon": poz_sayac, "patterns": patterns, "toplam_hata": len(hata_listesi)}


# ═══ PEDAGOJİK ÖNERİ MOTORU ═══
def pedagojik_oneri_uret(patterns, hata_listesi):
    """S.O.D.A. çerçevesine ve Türkçe artikülasyon rehberine göre egzersiz önerileri"""
    oneriler = []
    tip_dag = patterns.get("tip_dagilimi", {})
    ses_dag = patterns.get("ses_degisim", {})

    # ── SPESİFİK SES HATALARI (makaleden) ──
    for ses, sayi in ses_dag.items():
        if "→" not in ses: continue
        k, h = ses.split("→")
        tp = TURK_SES_HATALARI.get(ses, None)

        # R sesi hataları (Rötasizm)
        if k == "r" and h in ("y", "l"):
            oneriler.append({"baslik": "🐴 Dil Kası ve R Sesi Çalışması", "icon": "🐴",
                "aciklama": "Rötasizm: /r/ yerine /"+h+"/ kullanılıyor. Dil ucunun alveol bölgesinde (üst dişlerin arkası) titreşmesi gerekir.",
                "ornek": "1) At tırıs sesi (dil şıklatma) 10 kez. 2) Hızlıca 'Tı-dı-tı-dı' tekrarı — hızlandıkça R sesi ortaya çıkar. 3) Motor sesi 'Vrroooom' ile titreşimi hissettirme. 4) Arı vızıltısı ile dil ucunu uyarma.",
                "hedef": "Dil ucu titreşimini başlatma","tip":"ses_ozel"})
            break
        # K/G öne kayma
        elif (k == "k" and h == "t") or (k == "g" and h == "d"):
            oneriler.append({"baslik": "👅 Damak Teması Çalışması", "icon": "👅",
                "aciklama": f"Öne Kayma: /{k}/ yerine /{h}/ kullanılıyor. Dil sırtı yumuşak damağa (arkaya) temas etmeli.",
                "ornek": f"1) Lolipop/abeslang ile dil ucuna hafifçe basılarak öne gelmesi engellenir. 2) Su gargarası yaptırarak arka damak temasını hissettirme. 3) Minimal çiftler: '{k}edi' mi '{h}edi' mi? Resimli kartlarla ayırt etme.",
                "hedef": "Velar (arka damak) temasını otomatikleştirme","tip":"ses_ozel"})
            break
        # S/Ş peltek
        elif (k == "s" and h == "ş") or (k == "ş" and h == "s"):
            oneriler.append({"baslik": "🐍 Yılan Tıslama Çalışması", "icon": "🐍",
                "aciklama": "Peltek söyleyiş: Dil ucu ön dişlerin arasına fırlıyor. Dişler kapalı kalmalı.",
                "ornek": "1) Ayna karşısında gülümse ve dişleri kenetle. 'İçerideki yılanı tıslat ama kapılar (dişler) kapalı: Sssss' 2) Ön dişlerin arasına yatay pipet yerleştir, ısır ve 'S' de — dil dışarı çıkamaz. 3) S→hece→kelime→cümle hiyerarşisi.",
                "hedef": "Dil ucunu dişlerin arkasında tutma","tip":"ses_ozel"})
            break
        # Sadalama hataları (p→b, t→d)
        elif tp and "sadalama" in (tp.get("aciklama","")).lower():
            oneriler.append({"baslik": "💨 Nefes Kontrolü Çalışması", "icon": "💨",
                "aciklama": f"Sadalama hatası: Tonsuz /{k}/ yerine tonlu /{h}/. Nefes kontrolü eksik.",
                "ornek": f"1) Elin ağzın önünde tutarak '{k}' dediğinde nefes hisset, '{h}' dediğinde hissetme farkı. 2) Fısıltı ile '{k}' sesi çalışması. 3) Kelime hiyerarşisi: {k}a → {k}al → {k}alem → '{k}alemi al'",
                "hedef": "Tonsuz-tonlu ses ayrımını öğrenme","tip":"ses_ozel"})
            break

    # ── GENEL TİP BAZLI ÖNERİLER ──
    if tip_dag.get("substitusyon", 0) >= 1 and not any(o["tip"]=="ses_ozel" for o in oneriler):
        ses_ornekler = [f"'{k}' ve '{h}'" for ses, sayi in ses_dag.items() if "→" in ses for k, h in [ses.split("→")]]
        oneriler.append({"baslik": "🎯 Minimal Çiftler Terapisi", "icon": "🎯",
            "aciklama": "Karıştırılan seslerin anlam farkı yarattığını resimli kartlarla gösterin.",
            "ornek": f"Sesler: {', '.join(ses_ornekler[:3])}. 'Bana KEDİ'yi göster... şimdi TEDİ'yi göster' oyunu.",
            "hedef": "İşitsel ayırt etme becerisi","tip":"substitusyon"})

    if tip_dag.get("omisyon", 0) >= 1:
        oneriler.append({"baslik": "🤖 Robot Okuma + Ritim Tutma", "icon": "🤖",
            "aciklama": "Ses yutma: Hız, motor beceri kapasitesini aşıyor. Hızı yavaşlatıp her heceyi vurgulayın.",
            "ornek": "1) Her hece için masaya bir kez vur: A-yak-ka-bı (4 vuruş). 2) 'Robot gibi konuş' oyunu — abartılı ağız hareketleri. 3) Noktalama işaretlerinde duraksayarak okuma.",
            "hedef": "Kas hafızası güçlendirme, hız kontrolü","tip":"omisyon"})

    if tip_dag.get("ekleme", 0) >= 1:
        oneriler.append({"baslik": "✂️ Ses Sayma Oyunu", "icon": "✂️",
            "aciklama": "Fazla ses ekleme: Çocuk kelimeye olmayan sesler ekliyor (ör: spor→sipor, tren→tiren).",
            "ornek": "Kelimeyi yavaşça söyle ve her sesi parmakla say. S-P-O-R = 4 parmak, SI-PO-R ≠ doğru!",
            "hedef": "Doğru ses sayısı farkındalığı","tip":"ekleme"})

    if tip_dag.get("metatez", 0) >= 1:
        oneriler.append({"baslik": "🧩 Harf Dizme Oyunu", "icon": "🧩",
            "aciklama": "Harf sırası karışıyor. Mıknatıslı/karton harflerle doğru sıraya dizme.",
            "ornek": "T-O-P-R-A-K harflerini karıştır, doğru sıraya koy. Doğruysa alkışla!",
            "hedef": "Kelime içi ses sıralamasını pekiştirme","tip":"metatez"})

    # ── İŞİTSEL BOMBARDIMAN (her durumda, hata varsa) ──
    if hata_listesi:
        hatali_kelimeler = list(set([h.get("kelime","") for h in hata_listesi if h.get("kelime")]))[:5]
        oneriler.append({"baslik": "👂 İşitsel Bombardıman", "icon": "👂",
            "aciklama": "Hatalı kelimeleri vurgulu ve yavaş bir şekilde çocuğa bol bol dinletin. 'Hayır öyle değil' demeden, doğru modeli verin.",
            "ornek": f"Zorlanılan kelimeler: {', '.join(hatali_kelimeler)}. Bu kelimeleri içeren kısa bir masal vurgulu okuyun.",
            "hedef": "Doğru ses modelini içselleştirme","tip":"genel"})

    return oneriler


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
    prompt=f"""Sen cocuk okuma gucluğu ve akici okuma uzmanisin. MEB Prozodik Okuma Olcegi ve S.O.D.A. artikulasyon cercevesini biliyorsun.

{sinif}. sinif "{ad}" sesli okuma analizi (Asama {asama}).
Toplam:{kar['toplam']} Dogru:{kar['dogru']} Hatali:{kar['hatali']} Atlanan:{kar['atlanmis']} Dogruluk:%{kar['dogruluk_yuzdesi']}
Hatali kelimeler: {hl or 'Yok'}
Ses anomalileri (heceleme/yavas): {', '.join(anomaliler) or 'Yok'}
Akustik: Pitch:{pz['pitch']['ort']}Hz std:{pz['pitch']['std']}({pz['pitch']['mon']}), Enerji std:{pz['enerji']['std']}dB({pz['enerji']['vur']}), Sure:{pz['sure']['toplam']}sn, Sessizlik:%{pz['sure']['sessizlik']}, Duraksama:{pz['sure']['duraksama']}({pz['sure']['uzun']}uzun)
Sinif ref WPM:{sr['wpm_min']}-{sr['wpm_max']}, min dogruluk:%{sr.get('dogruluk_min',95)}

HATALI KELIMELER ICIN: Her hata icin S.O.D.A. siniflandirmasi yap (sub=ses degistirme, omi=ses dusurme, met=harf karistirma, ekl=ses ekleme). Kelimenin anlamini cocuga uygun acikla. Dogru telaffuzu hece hece goster.

PROZODIK OLCEK: Pitch std degerine bak — std<15 ise monoton (dusuk puan), std 15-30 normal, std>30 canli. Duraksama sayisi ve sessizlik yuzdesi akicilik icin onemli. Puan ALGORITMIK olsun, rastgele degil.

SADECE JSON:
{{"genel_skor":0-100,"akicilik_skoru":0-100,"prozodi_skoru":0-100,"seviye":"Baslangic/Gelisen/Yeterli/Ileri","wpm":N,"kaba":{{"toplam":{kar['toplam']},"dogru":{kar['dogru']},"yanlis":{kar['hatali']},"atlanan":{kar['atlanmis']},"wpm":N}},"prozodik_olcek":[{{"m":"Duygu yansitma","p":0-4}},{{"m":"Konusma dilini kullanma","p":0-4}},{{"m":"Vurgu ve tonlama","p":0-4}},{{"m":"Noktalamaya uyum","p":0-4}},{{"m":"Anlam birimleri vurgusu","p":0-4}},{{"m":"Uygun yerde bekleme","p":0-4}},{{"m":"Akici okuma","p":0-4}},{{"m":"Anlamli gruplama","p":0-4}}],"pt":0-32,"ao":"akicilik ozeti 1 cumle","po":"prozodi ozeti 1 cumle","hk":[{{"k":"hatali_kelime","o":"cocugun_okudugu","t":"he-ce-le-me","a":"cocuga_uygun_anlam","ht":"sub/omi/met/ekl"}}],"gy":["guclu yonler"],"ga":["gelisim alanlari"],"on":["cocuga ozel oneriler"]}}"""
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

    pedagoji = f"""PEDAGOJIK KURALLAR (kesinlikle uy):
1. %5-10 KURALI: Kelimelerin %90-95'i cocugun bildigi kelimeler olmali. Sadece %{sr.get('yeni_kelime_orani','5')} yeni/hedef kelime icersin.
2. BAGLAMSAL IPUCU: Her yeni kelime baglamda anlasilmali. Kelimeyi kullanmadan once veya sonra anlamini fisildayan bir cumle koy. KOTU: "Cocuk mutereddit gorunuyordu." IYI: "Iceri girsem mi girmesem mi diye dusundu, bir turlu karar veremedi."
3. ES ANLAMLI DANS: Ayni kavrami farkli kelimelerle tekrarla (sevinmek → mutlu olmak → nesesiyle ziplamak).
4. HIKAYE HARITASI: {sr.get('yapi','')}
5. CUMLE UZUNLUGU: {sr.get('cumle_uzunluk','4-8')} kelime arasi cumleler.
6. KELIME TURLERI: {sr.get('kelime_turu','')}
7. CIKARIM FIRSATI: {sr.get('ipucu','')}"""

    if hatali_kelimeler:
        prompt=f"""{sinif}. sinif Turkce okuma metni yaz.
MUTLAKA su kelimeleri dogal olarak icersin: {', '.join(hatali_kelimeler[:10])}
Tam olarak {sr['kelime']} kelime.
Tema secenekleri: {sr.get('tema','')}
{pedagoji}
BASLIK YAZMA. ACIKLAMA EKLEME. Sadece metni yaz."""
    else:
        prompt=f"""{sinif}. sinif Turkce okuma metni yaz.
Tam olarak {sr['kelime']} kelime.
Tema secenekleri: {sr.get('tema','')}
Her seferinde FARKLI bir tema sec.
{pedagoji}
BASLIK YAZMA. ACIKLAMA EKLEME. Sadece metni yaz."""

    resp=client.chat.completions.create(model="gpt-4o-mini",messages=[{"role":"user","content":prompt}],max_tokens=600,temperature=0.9)
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

            # K6: Artikülasyon hata sınıflandırma
            print("  Hata siniflandirma...")
            hatalar=hatalari_siniflandir(kar)
            print(f"  {len(hatalar)} hata siniflandirildi")

            # K7: Pattern tespiti (TÜM GEÇMİŞ VERİLERDEN)
            print("  Pattern tespiti (kumulatif)...")
            gecmis_hatalar=[]
            gecmis_sessions=ReadingSession.query.filter_by(user_id=current_user.id).order_by(ReadingSession.created_at.desc()).limit(20).all()
            for gs in gecmis_sessions:
                try:
                    gh=json.loads(gs.hatali_json or '[]')
                    for kelime in gh:
                        gecmis_hatalar.append({"tip":"belirsiz","kelime":kelime,"ses":"","pozisyon":"","detay":""})
                except: pass
            patterns=pattern_tespit(hatalar, gecmis_hatalar)
            print(f"  {len(patterns['patterns'])} pattern ({len(gecmis_hatalar)} gecmis hata dahil)")

            # K8: Pedagojik öneri
            ped_oneriler=pedagojik_oneri_uret(patterns, hatalar)
            print(f"  {len(ped_oneriler)} oneri uretildi")

            # Günlük başarı sayısı
            bugun=datetime.datetime.utcnow().date()
            basarili_bugun=ReadingSession.query.filter(ReadingSession.user_id==current_user.id,ReadingSession.tamamlandi==True,db.func.date(ReadingSession.created_at)==bugun).count()

            # Radar data
            sr=SINIF_REF.get(current_user.sinif,SINIF_REF["2"])
            wpm=rp.get("wpm",0)
            wpm_skor=min(100,max(0,int(wpm/sr["wpm_max"]*100)))
            radar={"dogruluk":kar["dogruluk_yuzdesi"],"akicilik":rp.get("akicilik_skoru",0),"prozodi":rp.get("prozodi_skoru",0),"hiz":wpm_skor,"vurgu":pz["radar_vurgu"],"tonlama":pz["radar_pitch"]}

            sess=ReadingSession(user_id=current_user.id,asama=asama,referans_metin=rm,transkript=tr,dogruluk=kar["dogruluk_yuzdesi"],akicilik=rp.get("akicilik_skoru",0),prozodi=rp.get("prozodi_skoru",0),genel_skor=rp.get("genel_skor",0),wpm=wpm,hatali_json=json.dumps(hl,ensure_ascii=False),rapor_json=json.dumps(rp,ensure_ascii=False),radar_json=json.dumps(radar),tamamlandi=tam)
            db.session.add(sess);db.session.commit()

            return jsonify({"basarili":True,"ogrenci_adi":current_user.name,"sinif":current_user.sinif,"asama":asama,"referans_metin":rm,"transkript":tr,"karsilastirma":kar,"kelime_detay":kd,"prozodi":pz,"rapor":rp,"radar":radar,"hatali_liste":hl,"tamamlandi":tam,"hatalar":hatalar,"patterns":patterns,"ped_oneriler":ped_oneriler,"basarili_bugun":basarili_bugun})
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
