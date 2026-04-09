"""
Okudio Okuma Dedektifi — Faz 2 (Adaptif)
Google Cloud STT + Parselmouth + Claude
"""
from pydub import AudioSegment
import os, io, json, tempfile, datetime, re, requests, base64
import numpy as np
from flask import Flask, render_template, request, jsonify, send_file

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

SINIF_REF = {
    "2": {"wpm_min":40,"wpm_max":60,"dogruluk_min":95,"kelime_sayisi":"40-60"},
    "3": {"wpm_min":70,"wpm_max":90,"dogruluk_min":95,"kelime_sayisi":"60-80"},
    "4": {"wpm_min":90,"wpm_max":110,"dogruluk_min":97,"kelime_sayisi":"80-100"},
    "5": {"wpm_min":110,"wpm_max":130,"dogruluk_min":98,"kelime_sayisi":"100-120"},
    "6": {"wpm_min":130,"wpm_max":145,"dogruluk_min":99,"kelime_sayisi":"120-140"},
    "7": {"wpm_min":145,"wpm_max":160,"dogruluk_min":99,"kelime_sayisi":"140-160"},
}

def temizle(metin):
    return re.sub(r'[^\w\s]', '', metin).lower().strip()

def google_stt_analiz(ses_dosya_yolu):
    with open(ses_dosya_yolu, "rb") as f:
        audio_b64 = base64.b64encode(f.read()).decode("utf-8")
    payload = {
        "config": {
            "encoding": "LINEAR16", "sampleRateHertz": 16000, "languageCode": "tr-TR",
            "enableWordTimeOffsets": True, "enableWordConfidence": True, "model": "default", "useEnhanced": True
        },
        "audio": {"content": audio_b64}
    }
    resp = requests.post(f"https://speech.googleapis.com/v1p1beta1/speech:recognize?key={GOOGLE_API_KEY}", json=payload, timeout=120)
    data = resp.json()
    if "error" in data:
        print(f"  STT HATA: {data['error']}"); return "", []
    if "results" not in data:
        return "", []
    transkript = ""; kelimeler = []
    for result in data["results"]:
        alt = result["alternatives"][0]
        transkript += alt.get("transcript", "") + " "
        for w in alt.get("words", []):
            start = w.get("startTime", "0s").replace("s", "")
            end = w.get("endTime", "0s").replace("s", "")
            kelimeler.append({"kelime": w["word"], "baslangic": round(float(start), 2), "bitis": round(float(end), 2), "confidence": round(w.get("confidence", 0), 3)})
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
        search_end = min(len(stt_words), stt_idx + 15)
        for j in range(stt_idx, search_end):
            s = lev_sim(ref_w, stt_words[j]["kelime_temiz"])
            if s > best_sim: best_sim = s; best_j = j; best_conf = stt_words[j]["confidence"]
        if best_sim >= 0.85 and best_conf >= 0.5:
            sonuc.append({"kelime": ref_w, "durum": "dogru", "confidence": best_conf, "stt": stt_words[best_j]["kelime"]})
            stt_idx = best_j + 1
        elif best_sim >= 0.4:
            sonuc.append({"kelime": ref_w, "durum": "hatali", "confidence": best_conf, "stt": stt_words[best_j]["kelime"]})
            stt_idx = best_j + 1
        else:
            sonuc.append({"kelime": ref_w, "durum": "atlanmis", "confidence": 0, "stt": ""})
    dogru = sum(1 for s in sonuc if s["durum"] == "dogru")
    hatali = sum(1 for s in sonuc if s["durum"] == "hatali")
    atlanmis = sum(1 for s in sonuc if s["durum"] == "atlanmis")
    toplam = len(ref_words)
    return {"kelimeler": sonuc, "toplam": toplam, "dogru": dogru, "hatali": hatali, "atlanmis": atlanmis, "dogruluk_yuzdesi": round(dogru / toplam * 100, 1) if toplam > 0 else 0}

def prozodi_analiz(ses_dosya_yolu):
    import parselmouth, librosa
    ses = parselmouth.Sound(ses_dosya_yolu)
    toplam_sure = ses.get_total_duration()
    pitch = ses.to_pitch()
    pv = pitch.selected_array['frequency']; pv = pv[pv > 0]
    if len(pv) > 0:
        pitch_data = {"ortalama_hz": round(float(np.mean(pv)),1), "std_hz": round(float(np.std(pv)),1), "monotonluk": "Monoton" if float(np.std(pv))<20 else "Normal" if float(np.std(pv))<40 else "Canli"}
    else:
        pitch_data = {"ortalama_hz":0,"std_hz":0,"monotonluk":"Olculemedi"}
    intensity = ses.to_intensity()
    iv = intensity.values.T.flatten(); iv = iv[iv>40]
    enerji_data = {"ortalama_db": round(float(np.mean(iv)),1), "std_db": round(float(np.std(iv)),1), "vurgu_durumu": "Yetersiz" if float(np.std(iv))<3.5 else "Normal" if float(np.std(iv))<7 else "Guclu"} if len(iv)>0 else {"ortalama_db":0,"std_db":0,"vurgu_durumu":"Olculemedi"}
    y, sr = librosa.load(ses_dosya_yolu, sr=None)
    sk = librosa.effects.split(y, top_db=30)
    ks = 0; duraksamalar = []; uzun = 0
    for i, k in enumerate(sk):
        ks += (k[1]-k[0])/sr
        if i > 0:
            ds = k[0]/sr - sk[i-1][1]/sr
            if ds > 0.15: duraksamalar.append(round(ds*1000,0)); uzun += 1 if ds > 1.0 else 0
    ss = toplam_sure - ks; so = (ss/toplam_sure)*100 if toplam_sure>0 else 0
    return {"pitch": pitch_data, "enerji": enerji_data, "sure": {"toplam_sn": round(toplam_sure,1), "konusma_sn": round(ks,1), "sessizlik_sn": round(ss,1), "sessizlik_orani": round(so,1), "duraksama_sayisi": len(duraksamalar), "uzun_duraksama": uzun, "akicilik": "Akici" if so<20 else "Normal" if so<35 else "Kesik kesik"}}

def claude_analiz(referans_metin, karsilastirma, prozodi, sinif, ogrenci_adi, asama=1):
    import anthropic
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    sr = SINIF_REF.get(sinif, SINIF_REF["2"])
    hatali_kelimeler = [k for k in karsilastirma["kelimeler"] if k["durum"] != "dogru"]
    hatali_str = ", ".join([f"{k['kelime']}(okunan:{k['stt']}, conf:{k['confidence']})" for k in hatali_kelimeler[:15]])
    prompt = f"""{sinif}. sinif "{ogrenci_adi}" sesli okuma yapti (Asama {asama}).

SONUCLAR: Toplam:{karsilastirma['toplam']}, Dogru:{karsilastirma['dogru']}, Hatali:{karsilastirma['hatali']}, Atlanan:{karsilastirma['atlanmis']}, Dogruluk:%{karsilastirma['dogruluk_yuzdesi']}
Hatalilar: {hatali_str or 'Yok'}
AKUSTIK: Pitch {prozodi['pitch']['ortalama_hz']}Hz std:{prozodi['pitch']['std_hz']}Hz ({prozodi['pitch']['monotonluk']}), Enerji {prozodi['enerji']['std_db']}dB ({prozodi['enerji']['vurgu_durumu']}), Sure {prozodi['sure']['toplam_sn']}sn, Sessizlik %{prozodi['sure']['sessizlik_orani']}, Duraksama {prozodi['sure']['duraksama_sayisi']}({prozodi['sure']['uzun_duraksama']}uzun)
REF: WPM:{sr['wpm_min']}-{sr['wpm_max']}, min dogruluk:%{sr['dogruluk_min']}
WPM hesapla: {karsilastirma['dogru']}x60/{prozodi['sure']['konusma_sn']}

SADECE JSON:
{{
  "genel_skor":0-100,"akicilik_skoru":0-100,"prozodi_skoru":0-100,
  "seviye":"Baslangic/Gelisen/Yeterli/Ileri","wpm":sayi,
  "kaba_degerlendirme":{{"toplam":{karsilastirma['toplam']},"dogru":{karsilastirma['dogru']},"yanlis":{karsilastirma['hatali']},"atlanan":{karsilastirma['atlanmis']},"wpm":sayi}},
  "prozodik_olcek":[{{"madde":"Duygu yansitma","puan":0-4}},{{"madde":"Konusma dili","puan":0-4}},{{"madde":"Vurgu tonlama","puan":0-4}},{{"madde":"Noktalama uyumu","puan":0-4}},{{"madde":"Anlam vurgusu","puan":0-4}},{{"madde":"Uygun bekleme","puan":0-4}},{{"madde":"Akici okuma","puan":0-4}},{{"madde":"Anlamli gruplama","puan":0-4}}],
  "prozodik_toplam":0-32,
  "akicilik_ozeti":"1-2 cumle","prozodi_ozeti":"1-2 cumle",
  "hatali_kelimeler":[{{"kelime":"x","okunan":"y","dogru_telaffuz":"he-ce","anlami":"kisa","hata_turu":"substitusyon/omisyon/metatez"}}],
  "guclu_yonler":["2 madde"],"gelisim_alanlari":["2 madde"],
  "oneriler":["ogretmene 1","veliye 1","cocuga 1"]
}}"""
    response = client.messages.create(model="claude-sonnet-4-20250514", max_tokens=2500, messages=[{"role": "user", "content": prompt}])
    raw = response.content[0].text.strip()
    if raw.startswith("```json"): raw = raw[7:]
    if raw.startswith("```"): raw = raw[3:]
    if raw.endswith("```"): raw = raw[:-3]
    try: return json.loads(raw.strip())
    except Exception as e:
        print(f"  JSON HATASI: {e}\n  HAM: {raw[:300]}"); return {"hata": "Parse hatasi"}


@app.route('/')
def index(): return render_template('index.html')

@app.route('/metin-olustur', methods=['POST'])
def metin_olustur():
    import anthropic
    sinif = request.json.get('sinif', '2')
    sr = SINIF_REF.get(sinif, SINIF_REF["2"])
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    response = client.messages.create(model="claude-sonnet-4-20250514", max_tokens=500, messages=[{"role": "user", "content": f"""{sinif}. sinif seviyesinde Turkce okuma metni yaz. {sr['kelime_sayisi']} kelime. Cocuklara uygun ilgi cekici konu (hayvanlar, doga, macera, bilim, uzay, dostluk vb). Her seferinde FARKLI konu. Sadece metni yaz."""}])
    return jsonify({"metin": response.content[0].text.strip()})

@app.route('/asama-metin', methods=['POST'])
def asama_metin():
    """Hatali kelimelerden yeni metin olustur"""
    import anthropic
    sinif = request.json.get('sinif', '2')
    hatali = request.json.get('hatali_kelimeler', [])
    asama = request.json.get('asama', 2)
    sr = SINIF_REF.get(sinif, SINIF_REF["2"])
    kelime_listesi = ", ".join(hatali[:10])
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    response = client.messages.create(model="claude-sonnet-4-20250514", max_tokens=500, messages=[{"role": "user", "content": f"""{sinif}. sinif seviyesinde Turkce okuma metni yaz.
Bu metin MUTLAKA su kelimeleri icersin: {kelime_listesi}
Bu kelimeler cocugun onceki okumada zorlandigi kelimeler. Metinde bu kelimeleri dogal sekilde kullan.
{sr['kelime_sayisi']} kelime arasi. Sadece metni yaz, aciklama ekleme."""}])
    return jsonify({"metin": response.content[0].text.strip()})

@app.route('/analiz', methods=['POST'])
def analiz():
    try:
        if 'ses_dosyasi' not in request.files: return jsonify({"hata": "Ses dosyasi yuklenmedi"}), 400
        sf = request.files['ses_dosyasi']
        rm = request.form.get('referans_metin', '').strip()
        ogrenci_adi = request.form.get('ogrenci_adi', '').strip() or 'Belirtilmedi'
        sinif = request.form.get('sinif', '2').strip()
        asama = int(request.form.get('asama', '1'))
        if not rm: return jsonify({"hata": "Referans metin girilmedi"}), 400
        sx = os.path.splitext(sf.filename)[1] or '.wav'
        with tempfile.NamedTemporaryFile(delete=False, suffix=sx) as tmp: sf.save(tmp.name); tp = tmp.name
        wp = tp + "_c.wav"
        try:
            AudioSegment.from_file(tp).set_frame_rate(16000).set_channels(1).set_sample_width(2).export(wp, format="wav")
            print(f"\n  === {ogrenci_adi} ({sinif}. sinif) Asama {asama} ===")
            print("  K1: Google STT..."); transkript, stt_kelimeler = google_stt_analiz(wp)
            if not transkript or len(stt_kelimeler) < 3:
                return jsonify({"basarili": False, "yeniden_kayit": True, "sebep": "Konusma algilanamadi."})
            print("  K2: Karsilastirma..."); karsilastirma = kelime_karsilastir(rm, stt_kelimeler)
            print(f"  Dogruluk: %{karsilastirma['dogruluk_yuzdesi']}")
            print("  K3: Prozodi..."); pz = prozodi_analiz(wp)
            print("  K4: Claude..."); rapor = claude_analiz(rm, karsilastirma, pz, sinif, ogrenci_adi, asama); print("  OK!")
            # Hatali kelimeleri ayikla (sonraki asama icin)
            hatali_liste = list(set([k["kelime"] for k in karsilastirma["kelimeler"] if k["durum"] != "dogru"]))
            return jsonify({
                "basarili": True, "ogrenci_adi": ogrenci_adi, "sinif": sinif, "asama": asama,
                "referans_metin": rm, "transkript": transkript, "stt_kelimeler": stt_kelimeler,
                "karsilastirma": karsilastirma, "prozodi": pz, "rapor": rapor,
                "hatali_liste": hatali_liste,
                "tamamlandi": karsilastirma["dogruluk_yuzdesi"] >= 95
            })
        finally:
            if os.path.exists(tp): os.unlink(tp)
            if os.path.exists(wp): os.unlink(wp)
    except Exception as e:
        import traceback; traceback.print_exc(); return jsonify({"hata": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=8080, host='0.0.0.0')
