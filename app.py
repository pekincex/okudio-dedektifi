"""
Okudio Okuma Dedektifi — Pedagojik Ses Analiz Motoru
====================================================
Whisper API (STT) + Parselmouth (Prozodi) + Claude API (Pedagojik Analiz)
"""
from pydub import AudioSegment
import os
import io
import json
import tempfile
import datetime
import numpy as np
from flask import Flask, render_template, request, jsonify, send_file

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024


def whisper_analiz(ses_dosya_yolu):
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    with open(ses_dosya_yolu, "rb") as audio:
        response = client.audio.transcriptions.create(
            model="whisper-1", file=audio, language="tr",
            response_format="verbose_json", timestamp_granularities=["word"],
            prompt="Ogrenci sesli okuma yapiyor. ee, ii, hmm, hih, sey gibi duraksama seslerini ve tekrar okunan kelimeleri oldugu gibi yaz."
        )
    transkript = response.text
    kelimeler = []
    if hasattr(response, 'words') and response.words:
        for w in response.words:
            kelimeler.append({"kelime": w.word, "baslangic": round(w.start, 2), "bitis": round(w.end, 2), "sure": round(w.end - w.start, 2)})
    return transkript, kelimeler


def prozodi_analiz(ses_dosya_yolu):
    import parselmouth
    import librosa
    ses = parselmouth.Sound(ses_dosya_yolu)
    toplam_sure = ses.get_total_duration()
    pitch = ses.to_pitch()
    pitch_values = pitch.selected_array['frequency']
    pitch_voiced = pitch_values[pitch_values > 0]
    if len(pitch_voiced) > 0:
        son_ceyrek = pitch_voiced[int(len(pitch_voiced)*0.75):]
        ilk_ceyrek = pitch_voiced[:max(1, int(len(pitch_voiced)*0.25))]
        pitch_data = {
            "ortalama_hz": round(float(np.mean(pitch_voiced)), 1),
            "min_hz": round(float(np.min(pitch_voiced)), 1),
            "max_hz": round(float(np.max(pitch_voiced)), 1),
            "std_hz": round(float(np.std(pitch_voiced)), 1),
            "aralik_hz": round(float(np.max(pitch_voiced) - np.min(pitch_voiced)), 1),
            "pitch_degisim_yonu": "dusus" if np.mean(son_ceyrek) < np.mean(ilk_ceyrek) else "yatis",
            "monotonluk": "Monoton" if float(np.std(pitch_voiced)) < 20 else "Normal" if float(np.std(pitch_voiced)) < 40 else "Cok canli"
        }
    else:
        pitch_data = {"ortalama_hz": 0, "std_hz": 0, "monotonluk": "Olculemedi", "pitch_degisim_yonu": "belirsiz"}
    intensity = ses.to_intensity()
    int_values = intensity.values.T.flatten()
    int_values = int_values[int_values > 40]
    if len(int_values) > 0:
        enerji_data = {"ortalama_db": round(float(np.mean(int_values)), 1), "std_db": round(float(np.std(int_values)), 1),
                       "vurgu_durumu": "Yetersiz" if float(np.std(int_values)) < 3.5 else "Normal" if float(np.std(int_values)) < 7 else "Guclu"}
    else:
        enerji_data = {"ortalama_db": 0, "std_db": 0, "vurgu_durumu": "Olculemedi"}
    y, sr = librosa.load(ses_dosya_yolu, sr=None)
    sesli_kisimlar = librosa.effects.split(y, top_db=30)
    konusulan_sure = 0
    duraksamalar = []
    uzun_duraksama_sayisi = 0
    for i, kisim in enumerate(sesli_kisimlar):
        konusulan_sure += (kisim[1] - kisim[0]) / sr
        if i > 0:
            onceki_bitis = sesli_kisimlar[i - 1][1] / sr
            simdiki_baslangic = kisim[0] / sr
            duraksama_suresi = simdiki_baslangic - onceki_bitis
            if duraksama_suresi > 0.15:
                duraksamalar.append({"zaman": round(onceki_bitis, 2), "sure_ms": round(duraksama_suresi * 1000, 0)})
                if duraksama_suresi > 1.0:
                    uzun_duraksama_sayisi += 1
    sessizlik_suresi = toplam_sure - konusulan_sure
    sessizlik_orani = (sessizlik_suresi / toplam_sure) * 100 if toplam_sure > 0 else 0
    return {
        "pitch": pitch_data, "enerji": enerji_data,
        "duraksama": {
            "toplam_sure_sn": round(toplam_sure, 1), "konusma_suresi_sn": round(konusulan_sure, 1),
            "sessizlik_suresi_sn": round(sessizlik_suresi, 1), "sessizlik_orani": round(sessizlik_orani, 1),
            "duraksama_sayisi": len(duraksamalar), "uzun_duraksama_sayisi": uzun_duraksama_sayisi,
            "duraksamalar": duraksamalar[:20],
            "akicilik_durumu": "Akici" if sessizlik_orani < 20 else "Normal" if sessizlik_orani < 35 else "Kesik kesik"
        }
    }


def kayit_kalitesi_kontrol(transkript, kelimeler, prozodi, referans_metin):
    ref_kelime_sayisi = len(referans_metin.strip().split())
    stt_kelime_sayisi = len(transkript.strip().split()) if transkript.strip() else 0
    toplam_sure = prozodi.get("duraksama", {}).get("toplam_sure_sn", 0)
    sessizlik_orani = prozodi.get("duraksama", {}).get("sessizlik_orani", 0)
    if stt_kelime_sayisi < 3:
        return False, "Kayitte neredeyse hic ses algilanamadi. Mikrofona yakin ve net bir sekilde okumayı deneyelim!"
    if toplam_sure < 5:
        return False, "Kayit cok kisa oldu ({}sn). En az 5 saniye kayit yapmamiz lazim!".format(round(toplam_sure, 1))
    if ref_kelime_sayisi > 5 and stt_kelime_sayisi < ref_kelime_sayisi * 0.15:
        return False, "Kayittaki ses referans metinle neredeyse hic uyusmuyor. Dogru metni mi okuduk?"
    if sessizlik_orani > 80:
        return False, "Kaydin buyuk bolumu sessizlik. Mikrofon cok uzakta kalmis olabilir."
    ref_kelimeler = set(referans_metin.lower().split())
    stt_kelimeler = set(transkript.lower().split())
    if ref_kelime_sayisi > 5 and len(ref_kelimeler.intersection(stt_kelimeler)) < 2:
        return False, "Kayittaki sesleri metinle eslestiremedik. Sakin bir ortamda tekrar deneyelim!"
    return True, ""


def claude_pedagojik_analiz(referans_metin, whisper_transkript, whisper_kelimeler, prozodi_verileri):
    import anthropic
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    prompt = f"""Sen Turkiye'de ilkokul duzeyinde akici okuma ve prozodi konusunda uzmanlasmis bir egitim bilimcisin.

REFERANS METIN:
{referans_metin}

WHISPER STT CIKTISI:
{whisper_transkript}

KELIME ZAMANLAMA:
{json.dumps(whisper_kelimeler, ensure_ascii=False, indent=2)}

PROZODI VERILERI:
{json.dumps(prozodi_verileri, ensure_ascii=False, indent=2)}

ANALIZ KILAVUZU:

A. PROZODI HATALARI - Su turleri tespit et:
- Tekduze/Monoton Okuma: pitch std_hz < 20 ise gosterge. Robotik, duygusuz okuma.
- Yanlis Vurgu/Tonlama: Soru cumlelerinin duz okunmasi, vurgularin yanlis yere konulmasi.
- Noktalama Yok Sayma: Virgul/noktada durmadan okuma. Duraksama zamanlarini noktalamayla karsilastir.
- Ifade Eksikligi: Metindeki duyguyu yansitamama.

B. AKICILIK HATALARI:
- Heceleme: Kelimeleri "ki-tap" gibi heceleyerek okuma. Kelime suresi > 1sn ise isaret.
- Tek Tek Okuma: Kelimeler arasi asiri duraksama.
- Anlamsiz Duraksama: Cumleyi yanlis yerden bolme.
- Yavas Okuma: 2. sinif icin 40-60 kelime/dk beklenir.
- Uzun Duraksama: 1sn+ duraksamalar sozcuk tanima guclugu.

C. DOGRULUK HATALARI:
- Hece Atlama: Kelimenin parcasini okumama.
- Kelime Ekleme/Cikarma: Metinde olmayan kelime okuma veya atlama.
- Yanlis Okuma: Kelimeyi baska kelimeyle degistirme.
- Tekrar Okuma: Ayni kelimeyi defalarca okuma.

JSON CIKTI (sadece JSON, baska hicbir sey yazma):
{{
  "genel_skor": 0-100,
  "dogruluk_skoru": 0-100,
  "akicilik_skoru": 0-100,
  "prozodi_skoru": 0-100,
  "seviye": "Baslangic / Gelisen / Yeterli / Ileri",
  "okuma_hizi_wpm": sayi,
  "kelime_analizi": [
    {{"kelime": "hedef", "durum": "dogru/yanlis/atlandi/eklendi/tekrar", "cocugun_okudugu": "...", "not": "pedagojik aciklama"}}
  ],
  "prozodi_hatalari": [
    {{"tur": "monoton_okuma/yanlis_vurgu/noktalama_yok_sayma/ifade_eksikligi", "aciklama": "...", "cumle_veya_bolum": "..."}}
  ],
  "akicilik_hatalari": [
    {{"tur": "heceleme/tek_tek_okuma/anlamsiz_duraksama/yavas_okuma/uzun_duraksama", "aciklama": "...", "zaman_araligi": "..."}}
  ],
  "dogruluk_hatalari": [
    {{"tur": "hece_atlama/kelime_ekleme/kelime_cikarma/yanlis_okuma/tekrar_okuma", "hedef_kelime": "...", "okunan": "...", "aciklama": "..."}}
  ],
  "dogruluk_ozeti": "2-3 cumle",
  "akicilik_ozeti": "2-3 cumle",
  "prozodi_ozeti": "2-3 cumle - pitch ve enerji verilerini yorumla",
  "guclu_yonler": ["en az 2 madde"],
  "gelisim_alanlari": ["en az 2 madde"],
  "oneriler": ["ogretmene 2 oneri", "veliye 1 oneri", "cocuga 1 motivasyon cumlesi"],
  "dikkat_edilecek_kelimeler": ["zorlanilan kelimeler"]
}}

NOT: Kelime suresi > 1sn = heceleme isareti. pitch std < 20 = monoton. sessizlik > %35 = kesik kesik. Turkce karakterler kullan."""

    response = client.messages.create(
        model="claude-sonnet-4-20250514", max_tokens=4000,
        messages=[{"role": "user", "content": prompt}]
    )
    raw_text = response.content[0].text
    cleaned = raw_text.strip()
    if cleaned.startswith("```json"): cleaned = cleaned[7:]
    if cleaned.startswith("```"): cleaned = cleaned[3:]
    if cleaned.endswith("```"): cleaned = cleaned[:-3]
    cleaned = cleaned.strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {"hata": "Claude yaniti parse edilemedi", "ham_yanit": raw_text}


def pdf_rapor_uret(rapor_data):
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.units import mm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    font_name = "Helvetica"
    font_bold = "Helvetica-Bold"
    for p in ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/dejavu/DejaVuSans.ttf", "/Library/Fonts/DejaVuSans.ttf", os.path.expanduser("~/Library/Fonts/DejaVuSans.ttf")]:
        if os.path.exists(p):
            pdfmetrics.registerFont(TTFont("DejaVu", p)); font_name = "DejaVu"; break
    for p in ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf", "/Library/Fonts/DejaVuSans-Bold.ttf", os.path.expanduser("~/Library/Fonts/DejaVuSans-Bold.ttf")]:
        if os.path.exists(p):
            pdfmetrics.registerFont(TTFont("DejaVuBold", p)); font_bold = "DejaVuBold"; break
    NAVY = colors.HexColor("#1A2332"); GREEN = colors.HexColor("#059669"); LIGHT_GREEN = colors.HexColor("#ECFDF5")
    ORANGE = colors.HexColor("#D97706"); LIGHT_ORANGE = colors.HexColor("#FFFBEB"); RED = colors.HexColor("#DC2626")
    LIGHT_RED = colors.HexColor("#FEF2F2"); GRAY = colors.HexColor("#6B7280"); LIGHT_GRAY = colors.HexColor("#F3F4F6")
    LIGHT_BLUE = colors.HexColor("#EFF6FF"); WHITE = colors.white
    rapor = rapor_data.get("rapor", {}); prozodi = rapor_data.get("prozodi", {}); whisper = rapor_data.get("whisper", {})
    s_title = ParagraphStyle("Title", fontName=font_bold, fontSize=22, textColor=NAVY, spaceAfter=4, leading=28)
    s_subtitle = ParagraphStyle("Subtitle", fontName=font_name, fontSize=10, textColor=GRAY, spaceAfter=16)
    s_heading = ParagraphStyle("Heading", fontName=font_bold, fontSize=13, textColor=NAVY, spaceBefore=18, spaceAfter=8, leading=18)
    s_body = ParagraphStyle("Body", fontName=font_name, fontSize=9.5, textColor=colors.HexColor("#374151"), leading=15, spaceAfter=4)
    s_small = ParagraphStyle("Small", fontName=font_name, fontSize=8, textColor=GRAY, leading=12)
    s_label = ParagraphStyle("Label", fontName=font_bold, fontSize=8, textColor=GRAY, leading=11, spaceAfter=2)
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=20*mm, bottomMargin=15*mm, leftMargin=18*mm, rightMargin=18*mm)
    elements = []; pw = A4[0] - 36*mm
    tarih = datetime.datetime.now().strftime("%d.%m.%Y %H:%M")
    elements.append(Paragraph("Okudio Okuma Analiz Raporu", s_title))
    elements.append(Paragraph(f"Tarih: {tarih} | Whisper + Parselmouth + Claude", s_subtitle))
    elements.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#E5E7EB"))); elements.append(Spacer(1, 8))
    elements.append(Paragraph("Performans Ozeti", s_heading))
    def sr(v):
        if v >= 70: return GREEN
        if v >= 40: return ORANGE
        return RED
    def sb(v):
        if v >= 70: return LIGHT_GREEN
        if v >= 40: return LIGHT_ORANGE
        return LIGHT_RED
    gs=rapor.get("genel_skor",0); ds=rapor.get("dogruluk_skoru",0); aks=rapor.get("akicilik_skoru",0); ps=rapor.get("prozodi_skoru",0)
    sd=[[Paragraph(f"<font size='20' color='{sr(gs).hexval()}'>{gs}</font><br/><font size='7' color='#6B7280'>Genel</font>",s_body),
         Paragraph(f"<font size='20' color='{sr(ds).hexval()}'>{ds}</font><br/><font size='7' color='#6B7280'>Dogruluk</font>",s_body),
         Paragraph(f"<font size='20' color='{sr(aks).hexval()}'>{aks}</font><br/><font size='7' color='#6B7280'>Akicilik</font>",s_body),
         Paragraph(f"<font size='20' color='{sr(ps).hexval()}'>{ps}</font><br/><font size='7' color='#6B7280'>Prozodi</font>",s_body)]]
    st=Table(sd, colWidths=[pw/4]*4, rowHeights=[50])
    st.setStyle(TableStyle([('ALIGN',(0,0),(-1,-1),'CENTER'),('VALIGN',(0,0),(-1,-1),'MIDDLE'),
        ('BACKGROUND',(0,0),(0,0),sb(gs)),('BACKGROUND',(1,0),(1,0),sb(ds)),('BACKGROUND',(2,0),(2,0),sb(aks)),('BACKGROUND',(3,0),(3,0),sb(ps)),
        ('BOX',(0,0),(-1,-1),0.5,colors.HexColor("#E5E7EB")),('INNERGRID',(0,0),(-1,-1),0.5,colors.HexColor("#E5E7EB"))]))
    elements.append(st)
    elements.append(Spacer(1,6))
    elements.append(Paragraph(f"<font color='#2563EB'><b>Seviye:</b> {rapor.get('seviye','')} | <b>Hiz:</b> {rapor.get('okuma_hizi_wpm','')} kelime/dk</font>", s_body))
    elements.append(Paragraph("Metin Karsilastirma", s_heading))
    ct=Table([[Paragraph("<b>Referans</b>",s_label),Paragraph("<b>Transkript</b>",s_label)],[Paragraph(rapor_data.get("referans_metin","")[:500],s_small),Paragraph(whisper.get("transkript","")[:500],s_small)]],colWidths=[pw/2]*2)
    ct.setStyle(TableStyle([('BACKGROUND',(0,0),(0,-1),LIGHT_GRAY),('BACKGROUND',(1,0),(1,-1),LIGHT_BLUE),('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6),('LEFTPADDING',(0,0),(-1,-1),8),('BOX',(0,0),(-1,-1),0.5,colors.HexColor("#E5E7EB")),('INNERGRID',(0,0),(-1,-1),0.5,colors.HexColor("#E5E7EB"))]))
    elements.append(ct)
    elements.append(Paragraph("Pedagojik Degerlendirme", s_heading))
    for l,k in [("Dogruluk","dogruluk_ozeti"),("Akicilik","akicilik_ozeti"),("Prozodi","prozodi_ozeti")]:
        v=rapor.get(k,"")
        if v: elements.append(Paragraph(f"<b>{l}:</b> {v}",s_body)); elements.append(Spacer(1,4))
    for ht,hb in [("prozodi_hatalari","Prozodi Hatalari"),("akicilik_hatalari","Akicilik Hatalari"),("dogruluk_hatalari","Dogruluk Hatalari")]:
        hh=rapor.get(ht,[])
        if hh:
            elements.append(Paragraph(hb, s_heading))
            for h in hh[:10]: elements.append(Paragraph(f"<b>[{h.get('tur','')}]</b> {h.get('aciklama','')}",s_body))
    pitch=prozodi.get("pitch",{}); enerji=prozodi.get("enerji",{}); dur=prozodi.get("duraksama",{})
    elements.append(Paragraph("Akustik Veriler", s_heading))
    pd=[["Metrik","Deger","Durum"],["Pitch",f"{pitch.get('ortalama_hz','-')} Hz",pitch.get('monotonluk','')],["Tonlama Std",f"{pitch.get('std_hz','-')} Hz",""],["Vurgu",f"{enerji.get('std_db','-')} dB",enerji.get('vurgu_durumu','')],["Sure",f"{dur.get('toplam_sure_sn','-')} sn",f"Konusma: {dur.get('konusma_suresi_sn','-')} sn"],["Duraksamalar",str(dur.get('duraksama_sayisi',0)),f"Uzun: {dur.get('uzun_duraksama_sayisi',0)}"],["Akicilik",dur.get('akicilik_durumu','-'),f"Sessizlik: %{dur.get('sessizlik_orani','-')}"]]
    pt=Table(pd,colWidths=[pw*0.35,pw*0.30,pw*0.35])
    pt.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),NAVY),('TEXTCOLOR',(0,0),(-1,0),WHITE),('FONTNAME',(0,0),(-1,0),font_bold),('FONTSIZE',(0,0),(-1,-1),8.5),('FONTNAME',(0,1),(-1,-1),font_name),('ROWBACKGROUNDS',(0,1),(-1,-1),[WHITE,LIGHT_GRAY]),('GRID',(0,0),(-1,-1),0.5,colors.HexColor("#E5E7EB")),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]))
    elements.append(pt)
    ka=rapor.get("kelime_analizi",[])
    if ka:
        elements.append(Paragraph("Kelime Analizi", s_heading))
        kh=[["#","Hedef","Okunan","Durum","Not"]]
        kr=[[str(i+1),k.get("kelime",""),k.get("cocugun_okudugu","-"),k.get("durum",""),k.get("not","")[:40]] for i,k in enumerate(ka[:60])]
        kt=Table(kh+kr,colWidths=[pw*0.06,pw*0.20,pw*0.20,pw*0.14,pw*0.40])
        ks=[('BACKGROUND',(0,0),(-1,0),NAVY),('TEXTCOLOR',(0,0),(-1,0),WHITE),('FONTNAME',(0,0),(-1,0),font_bold),('FONTSIZE',(0,0),(-1,-1),8),('FONTNAME',(0,1),(-1,-1),font_name),('GRID',(0,0),(-1,-1),0.3,colors.HexColor("#E5E7EB")),('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),4),('ROWBACKGROUNDS',(0,1),(-1,-1),[WHITE,LIGHT_GRAY])]
        for i,k in enumerate(ka[:60]):
            d=k.get("durum",""); r=i+1
            if d=="dogru": ks.append(('TEXTCOLOR',(3,r),(3,r),GREEN))
            elif d=="yanlis": ks.append(('TEXTCOLOR',(3,r),(3,r),RED))
            elif d=="atlandi": ks.append(('TEXTCOLOR',(3,r),(3,r),ORANGE))
        kt.setStyle(TableStyle(ks)); elements.append(kt)
    gu=rapor.get("guclu_yonler",[]); ge=rapor.get("gelisim_alanlari",[]); on=rapor.get("oneriler",[])
    if gu or ge:
        elements.append(Paragraph("Guclu Yonler ve Gelisim", s_heading))
        gg=[[Paragraph("<b>Guclu</b>",s_label),Paragraph("<b>Gelisim</b>",s_label)],[Paragraph("<br/>".join([f"+ {g}" for g in gu]) if gu else "-",s_small),Paragraph("<br/>".join([f"- {g}" for g in ge]) if ge else "-",s_small)]]
        gt=Table(gg,colWidths=[pw/2]*2)
        gt.setStyle(TableStyle([('BACKGROUND',(0,0),(0,-1),LIGHT_GREEN),('BACKGROUND',(1,0),(1,-1),LIGHT_ORANGE),('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),8),('LEFTPADDING',(0,0),(-1,-1),8),('BOX',(0,0),(-1,-1),0.5,colors.HexColor("#E5E7EB")),('INNERGRID',(0,0),(-1,-1),0.5,colors.HexColor("#E5E7EB"))]))
        elements.append(gt)
    if on:
        elements.append(Paragraph("Oneriler", s_heading))
        for o in on: elements.append(Paragraph(f"  {o}",s_body))
    di=rapor.get("dikkat_edilecek_kelimeler",[])
    if di: elements.append(Spacer(1,8)); elements.append(Paragraph(f"<b>Zorlanilan Kelimeler:</b> {', '.join(di)}",s_body))
    elements.append(Spacer(1,16)); elements.append(HRFlowable(width="100%",thickness=0.5,color=colors.HexColor("#E5E7EB")))
    elements.append(Paragraph(f"Okudio Okuma Dedektifi | {tarih}",ParagraphStyle("Footer",fontName=font_name,fontSize=7,textColor=GRAY,alignment=1)))
    doc.build(elements); buffer.seek(0); return buffer


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analiz', methods=['POST'])
def analiz():
    try:
        if 'ses_dosyasi' not in request.files: return jsonify({"hata": "Ses dosyasi yuklenmedi"}), 400
        ses_dosyasi = request.files['ses_dosyasi']
        referans_metin = request.form.get('referans_metin', '').strip()
        if not referans_metin: return jsonify({"hata": "Referans metin girilmedi"}), 400
        if ses_dosyasi.filename == '': return jsonify({"hata": "Dosya secilmedi"}), 400
        suffix = os.path.splitext(ses_dosyasi.filename)[1] or '.wav'
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            ses_dosyasi.save(tmp.name); tmp_path = tmp.name
        wav_path = tmp_path + "_converted.wav"
        try:
            ses = AudioSegment.from_file(tmp_path); ses.export(wav_path, format="wav")
            print("  Katman 1: Whisper...")
            transkript, kelimeler = whisper_analiz(wav_path)
            print(f"  Transkript: {transkript[:80]}...")
            print("  Katman 2: Prozodi...")
            prozodi = prozodi_analiz(wav_path)
            print(f"  Pitch: {prozodi['pitch'].get('ortalama_hz', '?')} Hz")
            gecerli, sebep = kayit_kalitesi_kontrol(transkript, kelimeler, prozodi, referans_metin)
            if not gecerli:
                print(f"  Kayit reddedildi: {sebep}")
                return jsonify({"basarili": False, "yeniden_kayit": True, "sebep": sebep})
            print("  Katman 3: Claude...")
            rapor = claude_pedagojik_analiz(referans_metin, transkript, kelimeler, prozodi)
            print("  Rapor uretildi!")
            return jsonify({"basarili": True, "referans_metin": referans_metin, "whisper": {"transkript": transkript, "kelimeler": kelimeler}, "prozodi": prozodi, "rapor": rapor})
        finally:
            if os.path.exists(tmp_path): os.unlink(tmp_path)
            if os.path.exists(wav_path): os.unlink(wav_path)
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"hata": str(e)}), 500


@app.route('/pdf-indir', methods=['POST'])
def pdf_indir():
    try:
        rapor_data = request.get_json()
        if not rapor_data: return jsonify({"hata": "Rapor verisi eksik"}), 400
        pdf_buffer = pdf_rapor_uret(rapor_data)
        tarih = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        return send_file(pdf_buffer, mimetype='application/pdf', as_attachment=True, download_name=f"Okudio_Rapor_{tarih}.pdf")
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"hata": str(e)}), 500


if __name__ == '__main__':
    print("\n" + "="*55)
    print("  Okudio Okuma Dedektifi")
    print("="*55)
    print(f"\n  http://localhost:8080\n")
    app.run(debug=True, port=8080, host='0.0.0.0')
