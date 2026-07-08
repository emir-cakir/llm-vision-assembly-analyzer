"""
STAGE 4b - inference_report.py'nin uretttigi video_analysis_report.json'i
OpenAI GPT-4o-mini'ye verip dogal dilde bir verimlilik raporu yazdirir.

Kurulum (bir kere):
    pip install openai

Kullanim:
    OPENAI_API_KEY ortam degiskenini ayarla, sonra:
    python generate_llm_report.py --json_path video_analysis_report.json

API key'ini kod icine YAZMA (guvensiz) - ortam degiskeni olarak ver:
    Windows (PowerShell):  $env:OPENAI_API_KEY = "sk-..."
    Sonra ayni terminalde:  python generate_llm_report.py ...
"""

import argparse
import json
import os
import sys


def build_prompt(report_data: dict) -> str:
    video_key = report_data["video_key"]
    # Tek kameralik (eski) format 'cam' kullanir, coklu kamera formati
    # 'cameras_used' + 'furniture' kullanir - ikisini de destekle.
    if "cam" in report_data:
        cam_desc = f"kamera: {report_data['cam']}"
    else:
        cams = ", ".join(report_data.get("cameras_used", []))
        cam_desc = f"{len(report_data.get('cameras_used', []))} kameranin birlesimi ({cams})"
    segments = report_data["segments"]

    real_segments = [s for s in segments if s["label"] != "NA"]
    na_total = sum(s["duration_sec"] for s in segments if s["label"] == "NA")
    total_duration = segments[-1]["end_sec"] - segments[0]["start_sec"] if segments else 0

    # --- On-siniflandirma + gruplama: ayni etiket (orn. 'align part') videoda
    # birden fazla kez gectiginde, LLM'e her tekil olayi degil, ETIKET
    # BAZINDA ozetlenmis (kac kez, ortalama fark) bir liste veriyoruz. Bu hem
    # raporu kisa/okunakli tutar hem de yon/isaret hatasini onler, cunku
    # yon zaten bizim tarafimizdan hesaplanip metne gomulu. ---
    from collections import defaultdict

    by_label = defaultdict(list)
    for s in real_segments:
        if s["pct_diff_vs_standard"] is not None:
            by_label[s["label"]].append(s)

    def summarize(items):
        n = len(items)
        total_dur = sum(s["duration_sec"] for s in items)
        avg_diff = sum(s["pct_diff_vs_standard"] for s in items) / n
        # camera_agreement sadece coklu-kamera (v2) formatinda var - eski
        # tek-kamera formatinda yoksa None dondur, prompt'ta atlanir.
        agreements = [s["camera_agreement"] for s in items if "camera_agreement" in s]
        avg_agreement = sum(agreements) / len(agreements) if agreements else None
        return n, total_dur, avg_diff, avg_agreement

    LOW_AGREEMENT_THRESHOLD = 0.7

    slow, fast, normal = [], [], []
    for label, items in by_label.items():
        n, total_dur, avg_diff, avg_agreement = summarize(items)
        tekrar = f"{n} kez, toplam {total_dur:.1f}sn" if n > 1 else f"{total_dur:.1f}sn"
        low_conf_note = ""
        if avg_agreement is not None and avg_agreement < LOW_AGREEMENT_THRESHOLD:
            low_conf_note = f" [DIKKAT: dusuk kesinlik, kameralar arasi uyum sadece %{avg_agreement*100:.0f}]"
        if avg_diff >= 30:
            slow.append(f"- {label} ({tekrar}): ortalama standarttan YUZDE {avg_diff:.0f} "
                        f"DAHA UZUN surdu (yavas){low_conf_note}")
        elif avg_diff <= -30:
            fast.append(f"- {label} ({tekrar}): ortalama standarttan YUZDE {abs(avg_diff):.0f} "
                        f"DAHA KISA surdu (hizli){low_conf_note}")
        else:
            normal.append(f"- {label} ({tekrar}): standarda yakin (ortalama fark: {avg_diff:+.0f}%)")

    slow_text = "\n".join(slow) if slow else "(yok)"
    fast_text = "\n".join(fast) if fast else "(yok)"
    normal_text = "\n".join(normal) if normal else "(yok)"
    total_steps = len(real_segments)

    prompt = f"""Sen bir montaj/verimlilik analistisin. Asagida, bir IKEA mobilya montaj
videosu icin ONCEDEN SINIFLANDIRILMIS ve ETIKET BAZINDA OZETLENMIS aksiyon tipleri
listesi var (ayni aksiyon tipi videoda birden fazla kez tekrarlanmis olabilir, bu
durumda kac kez tekrarlandigi ve ortalama sure farki verildi). Bu siniflandirma
(yavas/hizli/normal) zaten hesaplanmis ve KESIN dogrudur - sen tekrar hesaplama veya
yorumlama yapma, sadece bu bilgiyi akici Turkce bir metne don.

Video: {video_key} ({cam_desc})
Toplam sure: {total_duration:.1f} saniye, toplam {total_steps} adim (bazi aksiyon
tipleri birden fazla kez tekrarlandi)
Toplam bos/gecis suresi: {na_total:.1f} saniye

STANDARTTAN BELIRGIN YAVAS AKSIYON TIPLERI (mutlaka olumsuz/dikkat cekici sekilde belirt):
{slow_text}

STANDARTTAN BELIRGIN HIZLI AKSIYON TIPLERI (olumlu sekilde belirt):
{fast_text}

STANDARDA YAKIN AKSIYON TIPLERI (kisaca gecebilirsin, detaya girme):
{normal_text}

Gorevin: Yukaridaki ONCEDEN SINIFLANDIRILMIS ve OZETLENMIS bilgiyi kullanarak, DUZ
AKICI PARAGRAFLAR halinde (madde isareti, liste, kalin/markdown baslik KULLANMADAN)
kisa bir verimlilik raporu yaz. Rapor su akista olsun:
1. Kisa bir giris cumlesi (toplam sure, kac adim)
2. Bir paragrafta, yavas olan aksiyon tiplerinden bahset - "su adimlar standarttan
   belirgin sekilde yavas olmustur" tarzinda baslayip, her birini dogal bir cumle
   akisi icinde (madde isareti degil) yuzdesiyle birlikte anlat, ve HER biri icin
   olasi bir sebep tahmini ekle (orn. "...muhtemelen parcanin hizalanmasinda
   zorlanildigi icin olabilir"). Eger bir aksiyonun yaninda "[DIKKAT: dusuk kesinlik...]"
   notu varsa, o aksiyondan bahsederken bunu dogal bir dille belirt (orn. "...ancak bu
   tahminde kameralar arasinda tam bir gorus birligi olmadigi icin bu bulguyu ihtiyatla
   degerlendirmek gerekir"). Bu notu SADECE ilgili aksiyon icin kullan, digerlerine
   ekleme.
3. Bir paragrafta, hizli/iyi giden aksiyon tiplerinden kisaca ve olumlu bahset. Eger
   "hizli" listesi bossa, bunu OLUMSUZ bir sonuc olarak yorumlama veya "genel
   verimlilik olumsuz etkilendi" gibi veriyle desteklenmeyen bir genelleme YAPMA;
   bunun yerine "standarda yakin" listesindeki adimlari kisaca ve notr/olumlu bir
   dille belirt (o liste de bossa, bu paragrafi kisa gecebilir veya atlayabilirsin).
4. Son olarak, "su noktalara dikkat edilirse..." ya da "bir dahaki sefere ... konusuna
   dikkat edilmesi onerilir" tarzinda SOMUT VE UYGULANABILIR bir tavsiye paragrafi
   yaz - bu tavsiye, yavas olan adimlarla dogrudan ilgili olsun (genel gecer bir
   tavsiye degil).

Kurallar:
- KESINLIKLE madde isareti (-, *, numarali liste) veya kalin/markdown (**metin**)
  kullanma. Sadece duz metin paragraflari.
- Her aksiyon tipini SADECE BIR KEZ anlat (zaten kac kez tekrarlandigi bilgisi
  verildi, tek tek zaman araliklarini sayma).
- Bir aksiyonu SADECE ait oldugu listedeki (yavas/hizli) yonde anlat, ters yorumlama.
- Verilen listelerin OTESINDE genel bir yargi kurma (orn. "hizli adim yoktur, bu yuzden
  genel verimlilik dusuktur" gibi bir cikarim YAPMA - bu veriyle desteklenmeyen bir
  genellemedir).
- Rapor 150-250 kelime, sade Turkce, teknik terim (segment, model, pencere, JSON,
  etiket vb.) kullanma.
- SADECE rapor metnini yaz. Ingilizce kelime, not, aciklama, parantez ici yorum
  EKLEME. Rapor metninin disinda hicbir sey yazma."""
    return prompt


def call_openai(prompt: str, api_key: str) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=800,
    )
    return response.choices[0].message.content


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", default="video_analysis_report.json")
    parser.add_argument("--out", default="verimlilik_raporu.txt")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("HATA: OPENAI_API_KEY ortam degiskeni bulunamadi.")
        print('PowerShell icin: $env:OPENAI_API_KEY = "sk-..."')
        sys.exit(1)

    with open(args.json_path, "r", encoding="utf-8") as f:
        report_data = json.load(f)

    prompt = build_prompt(report_data)
    print("LLM'e istek gonderiliyor...")
    report_text = call_openai(prompt, api_key)

    print("\n" + "=" * 60)
    print(report_text)
    print("=" * 60)

    with open(args.out, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"\nKaydedildi: {args.out}")


if __name__ == "__main__":
    main()
