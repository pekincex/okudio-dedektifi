"""
Okudio Okuma Dedektifi v3 — MEB Prozodik Okuma Olcegi Referansli
Whisper + Parselmouth + Claude
"""
from pydub import AudioSegment
import os, io, json, tempfile, datetime
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
            prompt="Ogrenci sesli okuma yapiyor. ee, ii, hmm, hih, sey gibi duraksama seslerini, tekrar okunan kelimeleri ve heceleyerek okunan kelimeleri oldugu gibi yaz. Hicbir duzeltme yapma."
        )
    transkript = response.text
    kelimeler = []
    if hasattr(response, 'words') and response.words:
        for w in response.words:
            kelimeler.append({"kelime": w.word, "baslangic": round(w.start, 2), "bitis": round(w.end, 2), "sure": round(w.end - w.start, 2)})
    return transkript, kelimeler


def prozodi_analiz(ses_dosya_yolu):
    import parselmouth, librosa
    ses = parselmouth.Sound(ses_dosya_yolu)
    toplam_sure = ses.get_total_duration()
    pitch = ses.to_pitch()
    pv = pitch.selected_array['frequency']
    pv = pv[pv > 0]
    if len(pv) > 0:
        q1 = pv[:max(1,int(len(pv)*0.25))]; q4 = pv[int(len(pv)*0.75):]
        pitch_data = {"ortalama_hz": round(float(np.mean(pv)),1), "min_hz": round(float(np.min(pv)),1), "max_hz": round(float(np.max(pv)),1), "std_hz": round(float(np.std(pv)),1), "aralik_hz": round(float(np.max(pv)-np.min(pv)),1), "pitch_degisim_yonu": "dusus" if np.mean(q4)<np.mean(q1) else "yatis", "monotonluk": "Monoton" if float(np.std(pv))<20 else "Normal" if float(np.std(pv))<40 else "Cok canli"}
    else:
        pitch_data = {"ortalama_hz":0,"std_hz":0,"monotonluk":"Olculemedi","pitch_degisim_yonu":"belirsiz"}
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
            if ds > 0.15:
                duraksamalar.append({"zaman": round(sk[i-1][1]/sr,2), "sure_ms": round(ds*1000,0)})
                if ds > 1.0: uzun += 1
    ss = toplam_sure - ks; so = (ss/toplam_sure)*100 if toplam_sure>0 else 0
    return {"pitch": pitch_data, "enerji": enerji_data, "duraksama": {"toplam_sure_sn": round(toplam_sure,1), "konusma_suresi_sn": round(ks,1), "sessizlik_suresi_sn": round(ss,1), "sessizlik_orani": round(so,1), "duraksama_sayisi": len(duraksamalar), "uzun_duraksama_sayisi": uzun, "duraksamalar": duraksamalar[:20], "akicilik_durumu": "Akici" if so<20 else "Normal" if so<35 else "Kesik kesik"}}


def kayit_kalitesi_kontrol(transkript, kelimeler, prozodi, referans_metin):
    rks = len(referans_metin.strip().split()); sks = len(transkript.strip().split()) if transkript.strip() else 0
    ts = prozodi.get("duraksama",{}).get("toplam_sure_sn",0); so = prozodi.get("duraksama",{}).get("sessizlik_orani",0)
    if sks < 3: return False, "Kayitte neredeyse hic ses algilanamadi. Mikrofona yakin ve net okumayı deneyelim!"
    if ts < 5: return False, "Kayit cok kisa ({}sn). En az 5 saniye okumamiz lazim!".format(round(ts,1))
    if rks > 5 and sks < rks * 0.15: return False, "Kayittaki ses referans metinle uyusmuyor. Dogru metni mi okuduk?"
    if so > 80: return False, "Kaydin buyuk bolumu sessizlik. Mikrofon uzakta kalmis olabilir."
    rk = set(referans_metin.lower().split()); sk = set(transkript.lower().split())
    if rks > 5 and len(rk.intersection(sk)) < 2: return False, "Sesleri metinle eslestiremedik. Sakin ortamda tekrar deneyelim!"
    return True, ""


def claude_pedagojik_analiz(referans_metin, whisper_transkript, whisper_kelimeler, prozodi_verileri):
    import anthropic
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    prompt = f"""Sen Turkiye'de ilkokul duzeyinde akici okuma ve prozodi alaninda uzmanlasmis bir egitim bilimcisin.

REFERANSLAR:
- MEB Ozel Egitim ve Rehberlik Hizmetleri Genel Mudurlugu, Akici Okuma Ogretmen Kilavuzu (2025)
- Keskin & Bastug (2011) Prozodik Okuma Olcegi
- Aytac, A. (2017). Prozodi ile okudugunu anlama arasindaki iliski. Ana Dili Egitimi Dergisi, 5(2), 102-113.
- NAEP Oral Reading Fluency Scale

REFERANS METIN:
{referans_metin}

WHISPER STT:
{whisper_transkript}

KELIME ZAMANLAMA (ilk 30):
{json.dumps(whisper_kelimeler[:30], ensure_ascii=False)}

PROZODI VERILERI:
{json.dumps(prozodi_verileri, ensure_ascii=False)}

PUANLAMA METODOLOJISI — MEB Prozodik Okuma Olcegi (0-4 puan, 8 madde):
1. Metindeki duyguları okumasına yansıtır (heyecan, mutluluk, endise vb.)
2. Gunluk konusma dilinde okur (ritim ve duyguyla)
3. Vurgu ve tonlamaları metnin gereken yerlerinde kullanır
4. Ses tonunu noktalama isaretlerine gore ayarlar
5. Anlami vurgulanmak istenen sozcukler uygun tonlamayla okunur
6. Okuma sirasinda uygun surelerde bekler
7. Akıcı okur
8. Anlamli gruplamalar yaparak okur (tek nefeste okunan sozcukler birbiriyle alakali)

Her madde 0-4 arasi puanlanir. Toplam max 32 puan. Prozodi skorunu buna gore hesapla.

DOGRULUK HESAPLAMA:
- Dogru okunan sozcuk sayisi / Toplam sozcuk sayisi x 100
- %95+ = Bagimsiz duzey, %90-94 = Ogretimsel duzey, <%90 = Endiselendirici duzey

AKICILIK HESAPLAMA (WPM):
- Formul: (Dogru okunan sozcuk sayisi x 60) / Okuma suresi (saniye)
- 2. sinif beklentisi: 40-60 dogru sozcuk/dakika

ANALIZ CIKTISI (sadece JSON):
{{
  "genel_skor": 0-100,
  "dogruluk_skoru": 0-100,
  "akicilik_skoru": 0-100,
  "prozodi_skoru": 0-100,
  "seviye": "Baslangic/Gelisen/Yeterli/Ileri",
  "okuma_hizi_wpm": sayi,

  "puanlama_metodolojisi": {{
    "dogruluk_aciklama": "Dogruluk skoru nasil hesaplandi (dogru/toplam sozcuk, yuzde)",
    "akicilik_aciklama": "WPM nasil hesaplandi, sinif duzeyine gore degerlendirme",
    "prozodi_aciklama": "MEB Prozodik Okuma Olcegi 8 maddesi uzerinden degerlendirme",
    "prozodik_okuma_olcegi": [
      {{"madde": "Duyguları yansıtma", "puan": 0-4, "aciklama": "..."}},
      {{"madde": "Gunluk konusma dili", "puan": 0-4, "aciklama": "..."}},
      {{"madde": "Vurgu ve tonlama", "puan": 0-4, "aciklama": "..."}},
      {{"madde": "Noktalama isaretleri", "puan": 0-4, "aciklama": "..."}},
      {{"madde": "Anlam vurgusu", "puan": 0-4, "aciklama": "..."}},
      {{"madde": "Uygun bekleme", "puan": 0-4, "aciklama": "..."}},
      {{"madde": "Akici okuma", "puan": 0-4, "aciklama": "..."}},
      {{"madde": "Anlamli gruplama", "puan": 0-4, "aciklama": "..."}}
    ],
    "referanslar": ["MEB Akici Okuma Ogretmen Kilavuzu (2025)", "Keskin & Bastug (2011)", "NAEP Oral Reading Fluency Scale"]
  }},

  "kelime_analizi": [
    {{"kelime": "hedef", "durum": "dogru/yanlis/atlandi/eklendi/tekrar", "cocugun_okudugu": "...", "not": "pedagojik aciklama"}}
  ],

  "hatali_kelime_detaylari": [
    {{
      "kelime": "hatali okunan kelime",
      "cocugun_okudugu": "cocugun soyledigi",
      "dogru_telaffuz": "dogru telaffuz sekli (heceleriyle: orn ki-tap)",
      "anlami": "kelimenin sozluk anlami (kisa)",
      "hata_turu": "hece_atlama/yanlis_okuma/tekrar/heceleme/kelime_degistirme",
      "oneri": "bu kelimeyi calisirken ne yapilmali"
    }}
  ],

  "metin_karsilastirma": [
    {{"kelime": "referans kelime", "durum": "eslesme/uyumsuz/okunmadi", "eslesen_stt": "whisper karsiligi veya null"}}
  ],

  "prozodi_hatalari": [{{"tur": "...", "aciklama": "...", "cumle_veya_bolum": "..."}}],
  "akicilik_hatalari": [{{"tur": "...", "aciklama": "...", "zaman_araligi": "..."}}],
  "dogruluk_hatalari": [{{"tur": "...", "hedef_kelime": "...", "okunan": "...", "aciklama": "..."}}],

  "dogruluk_ozeti": "2-3 cumle",
  "akicilik_ozeti": "2-3 cumle",
  "prozodi_ozeti": "2-3 cumle - pitch/enerji verilerini yorumla",
  "guclu_yonler": ["en az 2 madde"],
  "gelisim_alanlari": ["en az 2 madde"],
  "oneriler": ["ogretmene 2 oneri", "veliye 1 oneri", "cocuga 1 motivasyon cumlesi"],
  "dikkat_edilecek_kelimeler": ["kelimeler"]
}}

ONEMLI:
- metin_karsilastirma dizisinde referans metindeki HER kelime icin bir giris olustur. Durum: "eslesme" (whisperda da var), "uyumsuz" (whisper farkli okumus), "okunmadi" (whisperda yok).
- hatali_kelime_detaylari dizisinde yanlis/atlanan her kelime icin anlam ve dogru telaffuz ver.
- Kelime suresi > 1sn = heceleme isareti. pitch std < 20 = monoton. sessizlik > %35 = kesik kesik.
- Sadece JSON dondur. Turkce karakterler kullan."""

    response = client.messages.create(model="claude-sonnet-4-20250514", max_tokens=8000, messages=[{"role": "user", "content": prompt}])
    raw = response.content[0].text.strip()
    if raw.startswith("```json"): raw = raw[7:]
    if raw.startswith("```"): raw = raw[3:]
    if raw.endswith("```"): raw = raw[:-3]
    try:
        return json.loads(raw.strip())
    except Exception as e:
        print(f"  JSON HATASI: {e}")
        print(f"  HAM YANIT: {raw[:500]}")
        return {"hata": "Claude yanit parse edilemedi", "ham_yanit": raw[:500]}


def pdf_rapor_uret(rapor_data):
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.units import mm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    fn = "Helvetica"; fb = "Helvetica-Bold"
    for p in ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf","/usr/share/fonts/dejavu/DejaVuSans.ttf","/Library/Fonts/DejaVuSans.ttf",os.path.expanduser("~/Library/Fonts/DejaVuSans.ttf")]:
        if os.path.exists(p): pdfmetrics.registerFont(TTFont("DejaVu",p)); fn="DejaVu"; break
    for p in ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf","/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf","/Library/Fonts/DejaVuSans-Bold.ttf",os.path.expanduser("~/Library/Fonts/DejaVuSans-Bold.ttf")]:
        if os.path.exists(p): pdfmetrics.registerFont(TTFont("DejaVuBold",p)); fb="DejaVuBold"; break
    NV=colors.HexColor("#1A2332");GR=colors.HexColor("#059669");LG=colors.HexColor("#ECFDF5");OR=colors.HexColor("#D97706");LO=colors.HexColor("#FFFBEB");RD=colors.HexColor("#DC2626");LR=colors.HexColor("#FEF2F2");GY=colors.HexColor("#6B7280");LGY=colors.HexColor("#F3F4F6");LB=colors.HexColor("#EFF6FF");WH=colors.white
    r=rapor_data.get("rapor",{});pr=rapor_data.get("prozodi",{});wh=rapor_data.get("whisper",{})
    st=ParagraphStyle("T",fontName=fb,fontSize=22,textColor=NV,spaceAfter=4,leading=28)
    ss=ParagraphStyle("S",fontName=fn,fontSize=10,textColor=GY,spaceAfter=16)
    sh=ParagraphStyle("H",fontName=fb,fontSize=13,textColor=NV,spaceBefore=18,spaceAfter=8,leading=18)
    sb=ParagraphStyle("B",fontName=fn,fontSize=9.5,textColor=colors.HexColor("#374151"),leading=15,spaceAfter=4)
    sm=ParagraphStyle("M",fontName=fn,fontSize=8,textColor=GY,leading=12)
    sl=ParagraphStyle("L",fontName=fb,fontSize=8,textColor=GY,leading=11,spaceAfter=2)
    buf=io.BytesIO(); doc=SimpleDocTemplate(buf,pagesize=A4,topMargin=20*mm,bottomMargin=15*mm,leftMargin=18*mm,rightMargin=18*mm)
    el=[]; pw=A4[0]-36*mm; t=datetime.datetime.now().strftime("%d.%m.%Y %H:%M")
    el.append(Paragraph("Okudio Okuma Analiz Raporu",st)); el.append(Paragraph(f"Tarih: {t} | MEB Prozodik Okuma Olcegi Referansli",ss))
    el.append(HRFlowable(width="100%",thickness=1,color=colors.HexColor("#E5E7EB"))); el.append(Spacer(1,8))
    def sc(v): return GR if v>=70 else OR if v>=40 else RD
    def bg(v): return LG if v>=70 else LO if v>=40 else LR
    gs=r.get("genel_skor",0);ds=r.get("dogruluk_skoru",0);ak=r.get("akicilik_skoru",0);ps=r.get("prozodi_skoru",0)
    el.append(Paragraph("Performans Ozeti",sh))
    sd=[[Paragraph(f"<font size='20' color='{sc(gs).hexval()}'>{gs}</font><br/><font size='7' color='#6B7280'>Genel</font>",sb),Paragraph(f"<font size='20' color='{sc(ds).hexval()}'>{ds}</font><br/><font size='7' color='#6B7280'>Dogruluk</font>",sb),Paragraph(f"<font size='20' color='{sc(ak).hexval()}'>{ak}</font><br/><font size='7' color='#6B7280'>Akicilik</font>",sb),Paragraph(f"<font size='20' color='{sc(ps).hexval()}'>{ps}</font><br/><font size='7' color='#6B7280'>Prozodi</font>",sb)]]
    tt=Table(sd,colWidths=[pw/4]*4,rowHeights=[50]); tt.setStyle(TableStyle([('ALIGN',(0,0),(-1,-1),'CENTER'),('VALIGN',(0,0),(-1,-1),'MIDDLE'),('BACKGROUND',(0,0),(0,0),bg(gs)),('BACKGROUND',(1,0),(1,0),bg(ds)),('BACKGROUND',(2,0),(2,0),bg(ak)),('BACKGROUND',(3,0),(3,0),bg(ps)),('BOX',(0,0),(-1,-1),0.5,colors.HexColor("#E5E7EB")),('INNERGRID',(0,0),(-1,-1),0.5,colors.HexColor("#E5E7EB"))])); el.append(tt)
    el.append(Spacer(1,6)); el.append(Paragraph(f"<font color='#2563EB'><b>Seviye:</b> {r.get('seviye','')} | <b>Hiz:</b> {r.get('okuma_hizi_wpm','')} kelime/dk</font>",sb))
    # Metodoloji
    pm = r.get("puanlama_metodolojisi",{})
    if pm:
        el.append(Paragraph("Puanlama Metodolojisi",sh))
        for k in ["dogruluk_aciklama","akicilik_aciklama","prozodi_aciklama"]:
            v=pm.get(k,"")
            if v: el.append(Paragraph(f"<b>{k.replace('_',' ').title()}:</b> {v}",sb))
        poo = pm.get("prozodik_okuma_olcegi",[])
        if poo:
            el.append(Spacer(1,4)); el.append(Paragraph("<b>Prozodik Okuma Olcegi (MEB, 0-4 puan):</b>",sb))
            pd=[["Madde","Puan","Aciklama"]]+[[p.get("madde",""),str(p.get("puan","")),p.get("aciklama","")] for p in poo]
            pt=Table(pd,colWidths=[pw*0.30,pw*0.10,pw*0.60]); pt.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),NV),('TEXTCOLOR',(0,0),(-1,0),WH),('FONTNAME',(0,0),(-1,0),fb),('FONTSIZE',(0,0),(-1,-1),8),('FONTNAME',(0,1),(-1,-1),fn),('GRID',(0,0),(-1,-1),0.3,colors.HexColor("#E5E7EB")),('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),4),('ROWBACKGROUNDS',(0,1),(-1,-1),[WH,LGY])])); el.append(pt)
    # Metin Karsilastirma
    el.append(Paragraph("Metin Karsilastirma",sh))
    ct=Table([[Paragraph("<b>Referans</b>",sl),Paragraph("<b>Transkript</b>",sl)],[Paragraph(rapor_data.get("referans_metin","")[:500],sm),Paragraph(wh.get("transkript","")[:500],sm)]],colWidths=[pw/2]*2)
    ct.setStyle(TableStyle([('BACKGROUND',(0,0),(0,-1),LGY),('BACKGROUND',(1,0),(1,-1),LB),('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6),('LEFTPADDING',(0,0),(-1,-1),8),('BOX',(0,0),(-1,-1),0.5,colors.HexColor("#E5E7EB")),('INNERGRID',(0,0),(-1,-1),0.5,colors.HexColor("#E5E7EB"))])); el.append(ct)
    # Degerlendirme
    el.append(Paragraph("Pedagojik Degerlendirme",sh))
    for l,k in [("Dogruluk","dogruluk_ozeti"),("Akicilik","akicilik_ozeti"),("Prozodi","prozodi_ozeti")]:
        v=r.get(k,"")
        if v: el.append(Paragraph(f"<b>{l}:</b> {v}",sb)); el.append(Spacer(1,4))
    # Hatali Kelime Detaylari
    hkd = r.get("hatali_kelime_detaylari",[])
    if hkd:
        el.append(Paragraph("Hatali Kelime Detaylari",sh))
        hd=[["Kelime","Okunan","Dogru Telaffuz","Anlami","Oneri"]]+[[h.get("kelime",""),h.get("cocugun_okudugu",""),h.get("dogru_telaffuz",""),h.get("anlami","")[:30],h.get("oneri","")[:30]] for h in hkd[:15]]
        ht=Table(hd,colWidths=[pw*0.15,pw*0.15,pw*0.20,pw*0.25,pw*0.25]); ht.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),NV),('TEXTCOLOR',(0,0),(-1,0),WH),('FONTNAME',(0,0),(-1,0),fb),('FONTSIZE',(0,0),(-1,-1),7.5),('FONTNAME',(0,1),(-1,-1),fn),('GRID',(0,0),(-1,-1),0.3,colors.HexColor("#E5E7EB")),('TOPPADDING',(0,0),(-1,-1),3),('BOTTOMPADDING',(0,0),(-1,-1),3),('ROWBACKGROUNDS',(0,1),(-1,-1),[WH,LGY])])); el.append(ht)
    # Hata Kategorileri
    for ht_key,hb in [("prozodi_hatalari","Prozodi Hatalari"),("akicilik_hatalari","Akicilik Hatalari"),("dogruluk_hatalari","Dogruluk Hatalari")]:
        hh=r.get(ht_key,[])
        if hh:
            el.append(Paragraph(hb,sh))
            for h in hh[:10]: el.append(Paragraph(f"<b>[{h.get('tur','')}]</b> {h.get('aciklama','')}",sb))
    # Akustik
    pi=pr.get("pitch",{}); en=pr.get("enerji",{}); du=pr.get("duraksama",{})
    el.append(Paragraph("Akustik Veriler",sh))
    avd=[["Metrik","Deger","Durum"],["Pitch",f"{pi.get('ortalama_hz','-')} Hz",pi.get('monotonluk','')],["Tonlama Std",f"{pi.get('std_hz','-')} Hz",""],["Vurgu",f"{en.get('std_db','-')} dB",en.get('vurgu_durumu','')],["Sure",f"{du.get('toplam_sure_sn','-')} sn",f"Konusma: {du.get('konusma_suresi_sn','-')} sn"],["Duraksamalar",str(du.get('duraksama_sayisi',0)),f"Uzun: {du.get('uzun_duraksama_sayisi',0)}"],["Akicilik",du.get('akicilik_durumu','-'),f"Sessizlik: %{du.get('sessizlik_orani','-')}"]]
    avt=Table(avd,colWidths=[pw*0.35,pw*0.30,pw*0.35]); avt.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),NV),('TEXTCOLOR',(0,0),(-1,0),WH),('FONTNAME',(0,0),(-1,0),fb),('FONTSIZE',(0,0),(-1,-1),8.5),('FONTNAME',(0,1),(-1,-1),fn),('ROWBACKGROUNDS',(0,1),(-1,-1),[WH,LGY]),('GRID',(0,0),(-1,-1),0.5,colors.HexColor("#E5E7EB")),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)])); el.append(avt)
    # Guclu/Gelisim
    gu=r.get("guclu_yonler",[]); ge=r.get("gelisim_alanlari",[]); on=r.get("oneriler",[])
    if gu or ge:
        el.append(Paragraph("Guclu Yonler ve Gelisim",sh))
        gg=[[Paragraph("<b>Guclu</b>",sl),Paragraph("<b>Gelisim</b>",sl)],[Paragraph("<br/>".join([f"+ {g}" for g in gu]) if gu else "-",sm),Paragraph("<br/>".join([f"- {g}" for g in ge]) if ge else "-",sm)]]
        gt=Table(gg,colWidths=[pw/2]*2); gt.setStyle(TableStyle([('BACKGROUND',(0,0),(0,-1),LG),('BACKGROUND',(1,0),(1,-1),LO),('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),8),('LEFTPADDING',(0,0),(-1,-1),8),('BOX',(0,0),(-1,-1),0.5,colors.HexColor("#E5E7EB")),('INNERGRID',(0,0),(-1,-1),0.5,colors.HexColor("#E5E7EB"))])); el.append(gt)
    if on:
        el.append(Paragraph("Oneriler",sh))
        for o in on: el.append(Paragraph(f"  {o}",sb))
    di=r.get("dikkat_edilecek_kelimeler",[])
    if di: el.append(Spacer(1,8)); el.append(Paragraph(f"<b>Zorlanilan:</b> {', '.join(di)}",sb))
    el.append(Spacer(1,16)); el.append(HRFlowable(width="100%",thickness=0.5,color=colors.HexColor("#E5E7EB")))
    el.append(Paragraph(f"Okudio Okuma Dedektifi | MEB Prozodik Okuma Olcegi Referansli | {t}",ParagraphStyle("F",fontName=fn,fontSize=7,textColor=GY,alignment=1)))
    doc.build(el); buf.seek(0); return buf


@app.route('/')
def index(): return render_template('index.html')

@app.route('/analiz', methods=['POST'])
def analiz():
    try:
        if 'ses_dosyasi' not in request.files: return jsonify({"hata":"Ses dosyasi yuklenmedi"}),400
        sf=request.files['ses_dosyasi']; rm=request.form.get('referans_metin','').strip()
        if not rm: return jsonify({"hata":"Referans metin girilmedi"}),400
        if sf.filename=='': return jsonify({"hata":"Dosya secilmedi"}),400
        sx=os.path.splitext(sf.filename)[1] or '.wav'
        with tempfile.NamedTemporaryFile(delete=False,suffix=sx) as tmp: sf.save(tmp.name); tp=tmp.name
        wp=tp+"_c.wav"
        try:
            AudioSegment.from_file(tp).export(wp,format="wav")
            print("  K1: Whisper..."); tr,kl=whisper_analiz(wp); print(f"  STT: {tr[:80]}...")
            print("  K2: Prozodi..."); pz=prozodi_analiz(wp); print(f"  Pitch: {pz['pitch'].get('ortalama_hz','?')} Hz")
            ok,sb=kayit_kalitesi_kontrol(tr,kl,pz,rm)
            if not ok: print(f"  Red: {sb}"); return jsonify({"basarili":False,"yeniden_kayit":True,"sebep":sb})
            print("  K3: Claude..."); rp=claude_pedagojik_analiz(rm,tr,kl,pz); print("  Rapor OK!")
            return jsonify({"basarili":True,"referans_metin":rm,"whisper":{"transkript":tr,"kelimeler":kl},"prozodi":pz,"rapor":rp})
        finally:
            if os.path.exists(tp): os.unlink(tp)
            if os.path.exists(wp): os.unlink(wp)
    except Exception as e:
        import traceback; traceback.print_exc(); return jsonify({"hata":str(e)}),500

@app.route('/pdf-indir', methods=['POST'])
def pdf_indir():
    try:
        rd=request.get_json()
        if not rd: return jsonify({"hata":"Veri eksik"}),400
        pb=pdf_rapor_uret(rd); return send_file(pb,mimetype='application/pdf',as_attachment=True,download_name=f"Okudio_Rapor_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.pdf")
    except Exception as e:
        import traceback; traceback.print_exc(); return jsonify({"hata":str(e)}),500

if __name__=='__main__':
    print("\n  Okudio Okuma Dedektifi v3\n  http://localhost:8080\n")
    app.run(debug=True,port=8080,host='0.0.0.0')
