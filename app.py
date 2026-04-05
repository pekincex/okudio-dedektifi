"""
Okudio Okuma Dedektifi — Pedagojik Ses Analiz Motoru
====================================================
Whisper API (STT) + Parselmouth (Prozodi) + Claude API (Pedagojik Analiz)

Kullanım:
  1. API key'lerini aşağıya yaz
  2. Terminal'de: python3 app.py
  3. Tarayıcıda: http://localhost:5000
"""
from pydub import AudioSegment
import os
import io
import json
import tempfile
import datetime
import numpy as np
from flask import Flask, render_template, request, jsonify, send_file

# ── API KEY'LERİNİ BURAYA YAZ ──────────────────────────────────
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
# ────────────────────────────────────────────────────────────────

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max upload


# ═══════════════════════════════════════════════════════════════
# KATMAN 1: WHISPER API — Sesten Metne + Kelime Zamanlaması
# ═══════════════════════════════════════════════════════════════
def whisper_analiz(ses_dosya_yolu):
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)

    with open(ses_dosya_yolu, "rb") as audio:
        response = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio,
            language="tr",
            response_format="verbose_json",
            timestamp_granularities=["word"]
        )

    transkript = response.text
    kelimeler = []
    if hasattr(response, 'words') and response.words:
        for w in response.words:
            kelimeler.append({
                "kelime": w.word,
                "baslangic": round(w.start, 2),
                "bitis": round(w.end, 2),
                "sure": round(w.end - w.start, 2)
            })

    return transkript, kelimeler


# ═══════════════════════════════════════════════════════════════
# KATMAN 2: PARSELMOUTH — Prozodi Analizi
# ═══════════════════════════════════════════════════════════════
def prozodi_analiz(ses_dosya_yolu):
    import parselmouth
    import librosa

    ses = parselmouth.Sound(ses_dosya_yolu)
    toplam_sure = ses.get_total_duration()

    # ── PITCH (F0) ──
    pitch = ses.to_pitch()
    pitch_values = pitch.selected_array['frequency']
    pitch_voiced = pitch_values[pitch_values > 0]

    if len(pitch_voiced) > 0:
        pitch_data = {
            "ortalama_hz": round(float(np.mean(pitch_voiced)), 1),
            "min_hz": round(float(np.min(pitch_voiced)), 1),
            "max_hz": round(float(np.max(pitch_voiced)), 1),
            "std_hz": round(float(np.std(pitch_voiced)), 1),
            "aralik_hz": round(float(np.max(pitch_voiced) - np.min(pitch_voiced)), 1),
            "monotonluk": "Monoton" if float(np.std(pitch_voiced)) < 20 else "Normal" if float(np.std(pitch_voiced)) < 40 else "Cok canli"
        }
    else:
        pitch_data = {"ortalama_hz": 0, "std_hz": 0, "monotonluk": "Olculemedi"}

    # ── ENERJİ ──
    intensity = ses.to_intensity()
    int_values = intensity.values.T.flatten()
    int_values = int_values[int_values > 40]

    if len(int_values) > 0:
        enerji_data = {
            "ortalama_db": round(float(np.mean(int_values)), 1),
            "std_db": round(float(np.std(int_values)), 1),
            "vurgu_durumu": "Yetersiz" if float(np.std(int_values)) < 3.5 else "Normal" if float(np.std(int_values)) < 7 else "Guclu"
        }
    else:
        enerji_data = {"ortalama_db": 0, "std_db": 0, "vurgu_durumu": "Olculemedi"}

    # ── DURAKSAMA ──
    y, sr = librosa.load(ses_dosya_yolu, sr=None)
    sesli_kisimlar = librosa.effects.split(y, top_db=30)

    konusulan_sure = 0
    duraksamalar = []
    for i, kisim in enumerate(sesli_kisimlar):
        konusulan_sure += (kisim[1] - kisim[0]) / sr
        if i > 0:
            onceki_bitis = sesli_kisimlar[i - 1][1] / sr
            simdiki_baslangic = kisim[0] / sr
            duraksama_suresi = simdiki_baslangic - onceki_bitis
            if duraksama_suresi > 0.15:
                duraksamalar.append({
                    "zaman": round(onceki_bitis, 2),
                    "sure_ms": round(duraksama_suresi * 1000, 0)
                })

    sessizlik_suresi = toplam_sure - konusulan_sure
    sessizlik_orani = (sessizlik_suresi / toplam_sure) * 100 if toplam_sure > 0 else 0

    return {
        "pitch": pitch_data,
        "enerji": enerji_data,
        "duraksama": {
            "toplam_sure_sn": round(toplam_sure, 1),
            "konusma_suresi_sn": round(konusulan_sure, 1),
            "sessizlik_suresi_sn": round(sessizlik_suresi, 1),
            "sessizlik_orani": round(sessizlik_orani, 1),
            "duraksama_sayisi": len(duraksamalar),
            "duraksamalar": duraksamalar[:20],
            "akicilik_durumu": "Akici" if sessizlik_orani < 20 else "Normal" if sessizlik_orani < 35 else "Kesik kesik"
        }
    }


# ═══════════════════════════════════════════════════════════════
# KATMAN 3: CLAUDE API — Pedagojik Yorumlama
# ═══════════════════════════════════════════════════════════════
def claude_pedagojik_analiz(referans_metin, whisper_transkript, whisper_kelimeler, prozodi_verileri):
    import anthropic
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    prompt = f"""Sen bir Turkce okuma egitimi uzmanisin. Ilkokul duzeyinde bir ogrencinin sesli okuma performansini analiz ediyorsun.

## Referans Metin (Cocugun okumasi gereken):
{referans_metin}

## Whisper STT Ciktisi (Cocugun gercekte okudugu):
{whisper_transkript}

## Kelime Bazli Zamanlama Verileri:
{json.dumps(whisper_kelimeler, ensure_ascii=False, indent=2)}

## Prozodi (Akustik) Analiz Verileri:
{json.dumps(prozodi_verileri, ensure_ascii=False, indent=2)}

## Gorevin:
Asagidaki basliklarda kapsamli bir pedagojik analiz raporu uret. Yanitini JSON formatinda ver.

JSON yapisi su olmali:
{{
  "genel_skor": 0-100 arasi genel okuma skoru,
  "dogruluk_skoru": 0-100 arasi dogruluk skoru,
  "akicilik_skoru": 0-100 arasi akicilik skoru,
  "prozodi_skoru": 0-100 arasi prozodi skoru,
  "seviye": "Baslangic / Gelisen / Yeterli / Ileri" (NAEP rubrigine gore),
  "okuma_hizi_wpm": dakikadaki dogru kelime sayisi (hesapla),
  "kelime_analizi": [
    {{"kelime": "referans kelime", "durum": "dogru/yanlis/atlandi/eklendi", "cocugun_okudugu": "...", "not": "kisa aciklama"}}
  ],
  "dogruluk_ozeti": "Dogruluk hakkinda 2-3 cumle
