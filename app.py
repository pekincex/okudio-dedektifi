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
  "dogruluk_ozeti": "Dogruluk hakkinda 2-3 cumlelik degerlendirme",
  "akicilik_ozeti": "Akicilik hakkinda 2-3 cumlelik degerlendirme",
  "prozodi_ozeti": "Prozodi/tonlama hakkinda 2-3 cumlelik degerlendirme (pitch ve enerji verilerini yorumla)",
  "guclu_yonler": ["cocugun iyi yaptigi seyler listesi"],
  "gelisim_alanlari": ["cocugun gelistirilmesi gereken alanlar"],
  "oneriler": ["ogretmene/veliye ozel oneriler"],
  "dikkat_edilecek_kelimeler": ["cocugun zorlandigi kelime listesi"]
}}

Onemli notlar:
- Referans metni Whisper ciktisiyla kelime kelime karsilastir.
- Whisper bazen cocugun dogru okudugu kelimeleri bile yanlis yazabilir, bunu goz onunde bulundur.
- Prozodi verilerindeki pitch standart sapmasi dusukse monoton okuma, yuksekse canli okuma demektir.
- Duraksama orani %35 uzeriyse cocuk kesik kesik okuyor demektir.
- Ilkokul 2. sinif seviyesine gore degerlendir.
- Sadece JSON dondur, baska metin ekleme.
- JSON icinde Turkce karakterler kullan (ş, ç, ğ, ü, ö, ı)."""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4000,
        messages=[{"role": "user", "content": prompt}]
    )

    raw_text = response.content[0].text
    cleaned = raw_text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    if cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {"hata": "Claude yaniti parse edilemedi", "ham_yanit": raw_text}


# ═══════════════════════════════════════════════════════════════
# PDF RAPOR ÜRETME
# ═══════════════════════════════════════════════════════════════
def pdf_rapor_uret(rapor_data):
    """
    Analiz sonuçlarını profesyonel PDF'e dönüştürür.
    Türkçe karakter desteği için DejaVu font kullanır.
    """
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        HRFlowable, KeepTogether
    )
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    # ── Font ayarı (Türkçe karakter desteği) ──
    font_name = "Helvetica"
    font_bold = "Helvetica-Bold"

    # DejaVu font varsa kullan (Türkçe ş, ç, ğ, ü, ö, ı desteği)
    dejavu_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        "/Library/Fonts/DejaVuSans.ttf",
        os.path.expanduser("~/Library/Fonts/DejaVuSans.ttf"),
    ]
    dejavu_bold_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
        "/Library/Fonts/DejaVuSans-Bold.ttf",
        os.path.expanduser("~/Library/Fonts/DejaVuSans-Bold.ttf"),
    ]

    for p in dejavu_paths:
        if os.path.exists(p):
            pdfmetrics.registerFont(TTFont("DejaVu", p))
            font_name = "DejaVu"
            break
    for p in dejavu_bold_paths:
        if os.path.exists(p):
            pdfmetrics.registerFont(TTFont("DejaVuBold", p))
            font_bold = "DejaVuBold"
            break

    # ── Renkler ──
    NAVY = colors.HexColor("#1A2332")
    BLUE = colors.HexColor("#2563EB")
    LIGHT_BLUE = colors.HexColor("#EFF6FF")
    GREEN = colors.HexColor("#059669")
    LIGHT_GREEN = colors.HexColor("#ECFDF5")
    ORANGE = colors.HexColor("#D97706")
    LIGHT_ORANGE = colors.HexColor("#FFFBEB")
    RED = colors.HexColor("#DC2626")
    LIGHT_RED = colors.HexColor("#FEF2F2")
    GRAY = colors.HexColor("#6B7280")
    LIGHT_GRAY = colors.HexColor("#F3F4F6")
    WHITE = colors.white

    rapor = rapor_data.get("rapor", {})
    prozodi = rapor_data.get("prozodi", {})
    whisper = rapor_data.get("whisper", {})

    # ── Stiller ──
    s_title = ParagraphStyle("Title", fontName=font_bold, fontSize=22, textColor=NAVY,
                             spaceAfter=4, leading=28)
    s_subtitle = ParagraphStyle("Subtitle", fontName=font_name, fontSize=10, textColor=GRAY,
                                spaceAfter=16)
    s_heading = ParagraphStyle("Heading", fontName=font_bold, fontSize=13, textColor=NAVY,
                               spaceBefore=18, spaceAfter=8, leading=18)
    s_body = ParagraphStyle("Body", fontName=font_name, fontSize=9.5, textColor=colors.HexColor("#374151"),
                            leading=15, spaceAfter=4)
    s_small = ParagraphStyle("Small", fontName=font_name, fontSize=8, textColor=GRAY, leading=12)
    s_label = ParagraphStyle("Label", fontName=font_bold, fontSize=8, textColor=GRAY,
                             leading=11, spaceAfter=2)

    # ── PDF Buffer ──
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        topMargin=20*mm, bottomMargin=15*mm,
        leftMargin=18*mm, rightMargin=18*mm
    )
    elements = []
    pw = A4[0] - 36*mm  # page width minus margins

    # ── Tarih ──
    tarih = datetime.datetime.now().strftime("%d.%m.%Y %H:%M")

    # ════════════════════ BAŞLIK ════════════════════
    elements.append(Paragraph("Okudio Okuma Analiz Raporu", s_title))
    elements.append(Paragraph(f"Olusturulma Tarihi: {tarih}  |  Whisper + Parselmouth + Claude", s_subtitle))
    elements.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#E5E7EB")))
    elements.append(Spacer(1, 8))

    # ════════════════════ SKOR KARTLARI ════════════════════
    elements.append(Paragraph("Performans Ozeti", s_heading))

    def skor_renk(v):
        if v >= 70: return GREEN
        if v >= 40: return ORANGE
        return RED

    def skor_bg(v):
        if v >= 70: return LIGHT_GREEN
        if v >= 40: return LIGHT_ORANGE
        return LIGHT_RED

    gs = rapor.get("genel_skor", 0)
    ds = rapor.get("dogruluk_skoru", 0)
    aks = rapor.get("akicilik_skoru", 0)
    ps = rapor.get("prozodi_skoru", 0)

    score_data = [[
        Paragraph(f"<font size='20' color='{skor_renk(gs).hexval()}'>{gs}</font><br/>"
                  f"<font size='7' color='#6B7280'>Genel Skor</font>", s_body),
        Paragraph(f"<font size='20' color='{skor_renk(ds).hexval()}'>{ds}</font><br/>"
                  f"<font size='7' color='#6B7280'>Dogruluk</font>", s_body),
        Paragraph(f"<font size='20' color='{skor_renk(aks).hexval()}'>{aks}</font><br/>"
                  f"<font size='7' color='#6B7280'>Akicilik</font>", s_body),
        Paragraph(f"<font size='20' color='{skor_renk(ps).hexval()}'>{ps}</font><br/>"
                  f"<font size='7' color='#6B7280'>Prozodi</font>", s_body),
    ]]

    score_table = Table(score_data, colWidths=[pw/4]*4, rowHeights=[50])
    score_table.setStyle(TableStyle([
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BACKGROUND', (0,0), (0,0), skor_bg(gs)),
        ('BACKGROUND', (1,0), (1,0), skor_bg(ds)),
        ('BACKGROUND', (2,0), (2,0), skor_bg(aks)),
        ('BACKGROUND', (3,0), (3,0), skor_bg(ps)),
        ('ROUNDEDCORNERS', [6,6,6,6]),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor("#E5E7EB")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#E5E7EB")),
    ]))
    elements.append(score_table)

    seviye = rapor.get("seviye", "")
    wpm = rapor.get("okuma_hizi_wpm", "")
    elements.append(Spacer(1, 6))
    elements.append(Paragraph(
        f"<font color='#2563EB'><b>Seviye:</b> {seviye}   |   "
        f"<b>Okuma Hizi:</b> {wpm} kelime/dk</font>", s_body
    ))

    # ════════════════════ METİN KARŞILAŞTIRMA ════════════════════
    elements.append(Paragraph("Metin Karsilastirma", s_heading))

    ref_metin = rapor_data.get("referans_metin", "")
    stt_metin = whisper.get("transkript", "")

    comp_data = [
        [Paragraph("<b>Referans Metin</b>", s_label),
         Paragraph("<b>Whisper Transkripti</b>", s_label)],
        [Paragraph(ref_metin[:500], s_small),
         Paragraph(stt_metin[:500], s_small)]
    ]
    comp_table = Table(comp_data, colWidths=[pw/2]*2)
    comp_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (0,-1), LIGHT_GRAY),
        ('BACKGROUND', (1,0), (1,-1), LIGHT_BLUE),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor("#E5E7EB")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#E5E7EB")),
    ]))
    elements.append(comp_table)

    # ════════════════════ PEDAGOJİK DEĞERLENDİRME ════════════════════
    elements.append(Paragraph("Pedagojik Degerlendirme", s_heading))

    for label, key in [("Dogruluk", "dogruluk_ozeti"), ("Akicilik", "akicilik_ozeti"), ("Prozodi", "prozodi_ozeti")]:
        val = rapor.get(key, "")
        if val:
            elements.append(Paragraph(f"<b>{label}:</b> {val}", s_body))
            elements.append(Spacer(1, 4))

    # ════════════════════ PROZODİ VERİLERİ ════════════════════
    elements.append(Paragraph("Prozodi ve Akustik Veriler", s_heading))

    pitch = prozodi.get("pitch", {})
    enerji = prozodi.get("enerji", {})
    dur = prozodi.get("duraksama", {})

    proz_data = [
        ["Metrik", "Deger", "Durum"],
        ["Pitch Ortalamasi", f"{pitch.get('ortalama_hz', '-')} Hz", pitch.get('monotonluk', '')],
        ["Tonlama Canliligi (Std)", f"{pitch.get('std_hz', '-')} Hz", "Yuksek = canli okuma"],
        ["Pitch Araligi", f"{pitch.get('aralik_hz', '-')} Hz", f"Min: {pitch.get('min_hz','-')} / Max: {pitch.get('max_hz','-')}"],
        ["Vurgu Gucu", f"{enerji.get('std_db', '-')} dB", enerji.get('vurgu_durumu', '')],
        ["Toplam Sure", f"{dur.get('toplam_sure_sn', '-')} sn", f"Konusma: {dur.get('konusma_suresi_sn','-')} sn"],
        ["Duraksama Sayisi", str(dur.get('duraksama_sayisi', 0)), f"Sessizlik: %{dur.get('sessizlik_orani','-')}"],
        ["Akicilik Durumu", dur.get('akicilik_durumu', '-'), ""],
    ]
    proz_table = Table(proz_data, colWidths=[pw*0.35, pw*0.30, pw*0.35])
    proz_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), NAVY),
        ('TEXTCOLOR', (0,0), (-1,0), WHITE),
        ('FONTNAME', (0,0), (-1,0), font_bold),
        ('FONTSIZE', (0,0), (-1,-1), 8.5),
        ('FONTNAME', (0,1), (-1,-1), font_name),
        ('BACKGROUND', (0,1), (-1,-1), WHITE),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [WHITE, LIGHT_GRAY]),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#E5E7EB")),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
    ]))
    elements.append(proz_table)

    # ════════════════════ KELİME BAZLI ANALİZ ════════════════════
    kelime_analizi = rapor.get("kelime_analizi", [])
    if kelime_analizi:
        elements.append(Paragraph("Kelime Bazli Analiz", s_heading))

        k_header = [["#", "Hedef Kelime", "Okunan", "Durum", "Not"]]
        k_rows = []
        for i, k in enumerate(kelime_analizi[:60]):  # Max 60 kelime
            durum_text = k.get("durum", "")
            k_rows.append([
                str(i+1),
                k.get("kelime", ""),
                k.get("cocugun_okudugu", "-"),
                durum_text,
                k.get("not", "")[:40]
            ])

        k_table = Table(k_header + k_rows, colWidths=[pw*0.06, pw*0.22, pw*0.22, pw*0.14, pw*0.36])

        k_style = [
            ('BACKGROUND', (0,0), (-1,0), NAVY),
            ('TEXTCOLOR', (0,0), (-1,0), WHITE),
            ('FONTNAME', (0,0), (-1,0), font_bold),
            ('FONTSIZE', (0,0), (-1,-1), 8),
            ('FONTNAME', (0,1), (-1,-1), font_name),
            ('GRID', (0,0), (-1,-1), 0.3, colors.HexColor("#E5E7EB")),
            ('TOPPADDING', (0,0), (-1,-1), 4),
            ('BOTTOMPADDING', (0,0), (-1,-1), 4),
            ('LEFTPADDING', (0,0), (-1,-1), 5),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [WHITE, LIGHT_GRAY]),
        ]

        # Durum renklendirme
        for i, k in enumerate(kelime_analizi[:60]):
            durum = k.get("durum", "")
            row = i + 1
            if durum == "dogru":
                k_style.append(('TEXTCOLOR', (3, row), (3, row), GREEN))
            elif durum == "yanlis":
                k_style.append(('TEXTCOLOR', (3, row), (3, row), RED))
            elif durum == "atlandi":
                k_style.append(('TEXTCOLOR', (3, row), (3, row), ORANGE))

        k_table.setStyle(TableStyle(k_style))
        elements.append(k_table)

    # ════════════════════ GÜÇLÜ YÖNLER & GELİŞİM ════════════════════
    guclu = rapor.get("guclu_yonler", [])
    gelisim = rapor.get("gelisim_alanlari", [])
    oneriler = rapor.get("oneriler", [])

    if guclu or gelisim:
        elements.append(Paragraph("Guclu Yonler ve Gelisim Alanlari", s_heading))

        gg_data = [
            [Paragraph("<b>Guclu Yonler</b>", s_label),
             Paragraph("<b>Gelisim Alanlari</b>", s_label)],
            [Paragraph("<br/>".join([f"+ {g}" for g in guclu]) if guclu else "-", s_small),
             Paragraph("<br/>".join([f"- {g}" for g in gelisim]) if gelisim else "-", s_small)]
        ]
        gg_table = Table(gg_data, colWidths=[pw/2]*2)
        gg_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (0,-1), LIGHT_GREEN),
            ('BACKGROUND', (1,0), (1,-1), LIGHT_ORANGE),
            ('VALIGN', (0,0), (-1,-1), 'TOP'),
            ('TOPPADDING', (0,0), (-1,-1), 6),
            ('BOTTOMPADDING', (0,0), (-1,-1), 8),
            ('LEFTPADDING', (0,0), (-1,-1), 8),
            ('RIGHTPADDING', (0,0), (-1,-1), 8),
            ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor("#E5E7EB")),
            ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#E5E7EB")),
        ]))
        elements.append(gg_table)

    if oneriler:
        elements.append(Paragraph("Ogretmen / Veli Onerileri", s_heading))
        for o in oneriler:
            elements.append(Paragraph(f"  {o}", s_body))

    # ── Zorlanılan Kelimeler ──
    dikkat = rapor.get("dikkat_edilecek_kelimeler", [])
    if dikkat:
        elements.append(Spacer(1, 8))
        elements.append(Paragraph(
            f"<b>Zorlanilan Kelimeler:</b> {', '.join(dikkat)}", s_body
        ))

    # ── Footer ──
    elements.append(Spacer(1, 16))
    elements.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#E5E7EB")))
    elements.append(Paragraph(
        f"Bu rapor Okudio Okuma Dedektifi tarafindan otomatik olarak olusturulmustur. | {tarih}",
        ParagraphStyle("Footer", fontName=font_name, fontSize=7, textColor=GRAY, alignment=1)
    ))

    # ── Build ──
    doc.build(elements)
    buffer.seek(0)
    return buffer


# ═══════════════════════════════════════════════════════════════
# FLASK ROTALARI
# ═══════════════════════════════════════════════════════════════
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analiz', methods=['POST'])
def analiz():
    try:
        if 'ses_dosyasi' not in request.files:
            return jsonify({"hata": "Ses dosyasi yuklenmedi"}), 400

        ses_dosyasi = request.files['ses_dosyasi']
        referans_metin = request.form.get('referans_metin', '').strip()

        if not referans_metin:
            return jsonify({"hata": "Referans metin girilmedi"}), 400
        if ses_dosyasi.filename == '':
            return jsonify({"hata": "Dosya secilmedi"}), 400

        suffix = os.path.splitext(ses_dosyasi.filename)[1] or '.wav'
        
        # Orijinal dosyayı kaydet
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            ses_dosyasi.save(tmp.name)
            tmp_path = tmp.name
        
        # Dönüştürülecek .wav dosyasının yolu
        wav_path = tmp_path + "_converted.wav"

        try:
            print(f"  Pydub ile dönüştürülüyor: {suffix} -> .wav")
            # Ses dosyasını wav formatına çevir
            ses = AudioSegment.from_file(tmp_path)
            ses.export(wav_path, format="wav")

            print("  Katman 1: Whisper analizi basliyor...")
            transkript, kelimeler = whisper_analiz(wav_path)
            print(f"  Transkript: {transkript[:80]}...")

            print("  Katman 2: Prozodi analizi basliyor...")
            prozodi = prozodi_analiz(wav_path)
            print(f"  Pitch: {prozodi['pitch'].get('ortalama_hz', '?')} Hz")

            print("  Katman 3: Claude pedagojik analiz basliyor...")
            rapor = claude_pedagojik_analiz(referans_metin, transkript, kelimeler, prozodi)
            print("  Rapor uretildi!")

            return jsonify({
                "basarili": True,
                "referans_metin": referans_metin,
                "whisper": {"transkript": transkript, "kelimeler": kelimeler},
                "prozodi": prozodi,
                "rapor": rapor
            })

        finally:
            # Temizlik - Sunucuda çöp dosya birikmemesi için ikisini de siliyoruz
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            if os.path.exists(wav_path):
                os.unlink(wav_path)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"hata": str(e)}), 500


@app.route('/pdf-indir', methods=['POST'])
def pdf_indir():
    """Analiz sonuçlarını PDF olarak indirir."""
    try:
        rapor_data = request.get_json()
        if not rapor_data:
            return jsonify({"hata": "Rapor verisi eksik"}), 400

        pdf_buffer = pdf_rapor_uret(rapor_data)

        tarih = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        filename = f"Okudio_Rapor_{tarih}.pdf"

        return send_file(
            pdf_buffer,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=filename
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"hata": str(e)}), 500


if __name__ == '__main__':
    print("\n" + "="*55)
    print("  Okudio Okuma Dedektifi")
    print("  Whisper + Parselmouth + Claude")
    print("="*55)
    print(f"\n  Tarayicida ac: http://localhost:5000")
    print(f"  Durdurmak icin: Ctrl+C\n")
    app.run(debug=True, port=8080, host='0.0.0.0')
