# LLM-Vision Assembly Time Analyzer

Bu proje; bilgisayarlı görü (Computer Vision) ve büyük dil modellerini (LLM) entegre
ederek, montaj videolarındaki operasyon sürelerini otomatik olarak tespit eden ve bunu
doğal dilde bir verimlilik raporuna dönüştüren uçtan uca bir video anlama sistemidir.

Proje kapsamında, **IKEA ASM Dataset** üzerindeki çoklu kamera açılı montaj videoları
işlenerek her montaj adımının süresi otomatik tespit edilmiş ve veri setinden
hesaplanan standart sürelerle karşılaştırılmıştır.

Bu proje İstanbul Medeniyet Üniversitesi Endüstri Mühendisliği bitirme tezi kapsamında geliştirilmiştir.

## Sistem Mimarisi ve İş Akışı

```
┌──────────────────────────────────────────────────────────┐
│   Ham Video Akışı (.avi, 2 Kamera Açısı: dev1, dev2)      │
└───────────────────────────┬────────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────┐
│   Ön İşleme & Paralel Tensör Çıkarımı (fps=8, yerel CPU)  │
└───────────────────────────┬────────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────┐
│   VideoMAE-Base ile Sıralı Öznitelik Çıkarımı (donmuş)    │
└───────────────────────────┬────────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────┐
│   BiLSTM Tabanlı Zaman-Adımı Sınıflandırma (10 Aksiyon)   │
└───────────────────────────┬────────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────┐
│   Çoklu Kamera Füzyonu (Çoğunluk Oyu & Güven Skoru)       │
└───────────────────────────┬────────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────┐
│   Mobilya Tipine Özel Standart Süre Karşılaştırması       │
└───────────────────────────┬────────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────┐
│   GPT-4o-mini ile Otomatik Verimlilik Raporlaması         │
└──────────────────────────────────────────────────────────┘
```

**Veri Seti:** [IKEA ASM Dataset](https://ikeaasm.github.io/) (Ben-Shabat et al.,
WACV 2021) — 371 montaj videosu, nominal tasarımda 3 senkronize kamera açısı
(`dev1`/`dev2`/`dev3`), karesel aksiyon etiketleri. **Bu projede yalnızca `dev1` ve
`dev2` kamera açıları kullanılmıştır** — `dev3` veri setinde mevcut olmakla
birlikte bu çalışmanın kapsamı dışında tutulmuştur.

## Örnek Çıktı Raporu

Sistemin ürettiği yapılandırılmış JSON verisi, LLM katmanı tarafından işlenerek
aşağıdaki gibi akıcı bir metne dönüştürülür (gerçek bir sistem çalıştırmasından
alınmıştır):

> Toplam süresi 129 saniye olan ve 23 adım içeren bu montaj videosunda, bazı aksiyon
> tipleri belirgin şekilde yavaş gerçekleştirilmiştir. Örneğin, "align part" işlemi 5
> kez yapılarak toplam 15 saniye sürmüştür ve bu, ortalama standarttan yüzde 74 daha
> uzun bir süre almıştır; muhtemelen parçanın hizalanmasında zorluk yaşandığı için
> böyle bir durum ortaya çıkmıştır. "Tighten leg" işlemi ise ortalama süreden yüzde 36
> daha uzun sürmüş, muhtemelen vida veya bağlantı noktasıyla ilgili bir zorluktan
> kaynaklanmaktadır. "Move table" eylemi yüzde 53 daha uzun sürmüş; ancak bu tahminde
> kameralar arasında tam bir görüş birliği olmadığından bu bulguyu ihtiyatla
> değerlendirmek gerekir. Bir dahaki sefere, özellikle "align part" ve "tighten leg"
> adımlarına daha fazla dikkat edilmesi önerilir.

Not: Rapor, model tahminlerinin ne kadar güvenilir olduğunu da (kameralar arası görüş
birliği düşükse) metne yansıtacak şekilde tasarlanmıştır.

## Model Geliştirme ve Optimizasyon Süreci

Geliştirme sürecinde, zamansal bağlamın korunması ve sınıf dengesizliğinin
giderilmesi amacıyla iteratif bir optimizasyon izlendi:

| Aşama | Yaklaşım | Çözülen Problem | Accuracy | Macro F1* |
|---|---|---|---|---|
| 1 | 8sn bağımsız pencere, 33 ince sınıf | — | %50 | 0.06 |
| 2 | Aynı pencere, 12 kaba sınıf | Nadir sınıflar birleştirilerek dengesizlik azaltıldı | %42 | 0.17 |
| 3 | fps=8 + **BiLSTM** (bağlamlı sıralı sınıflandırma), 12 sınıf | Kök neden (pencere/aksiyon süre uyumsuzluğu) çözüldü | %64 | 0.38 |
| 4 (final) | Aşırı nadir 2 sınıf daha birleştirme, 10 sınıf | Son ince ayar | **%65** | **0.44** |

*Macro F1'i neden ayrıca takip ettik: test verisinin ~%45'i tek bir sınıfa
(`spin leg`) ait olduğu için accuracy tek başına iyimser bir tablo çizebilir. Macro
F1, her sınıfı eşit ağırlıklandırdığı için modelin nadir sınıflarda da gerçekten
ilerleme kaydedip kaydetmediğini gösteren tamamlayıcı bir gösterge.

### Tasarım Kararları

- **Zaman senkronizasyon hatası ve düzeltmesi:** Geliştirme sürecinde, segment
  sürelerinin hesaplanmasında yanlışlıkla orijinal video fps'i (25) kullanıldığı
  tespit edildi; doğrusu tensörün örnekleme fps'iydi (8). Bu hata, segmentler
  arasında açıklanamayan zaman boşlukları olarak fark edildi; düzeltmeden sonra
  segmentler kusursuz kesintisiz hale geldi ve bu tutarlılık, düzeltmenin
  doğruluğunu da doğruladı.
- **LLM mimarisi seçimi:** Doğal dil raporlama için önce ücretsiz/yerel Llama 3
  (Ollama üzerinden) denendi. Model, önceden doğru şekilde etiketlenmiş ("yavaş"/
  "hızlı" diye açıkça işaretlenmiş) verileri bile sistematik olarak ters
  yorumladı (örn. "%80 daha kısa sürdü" ifadesini "yavaş" olarak sundu) — bu,
  prompt mühendisliğiyle düzelmedi. Bunun üzerine, maliyeti pratikte önemsiz olan
  (rapor başına ~$0.0005) GPT-4o-mini'ye geçildi; bu model aynı veriyle güvenilir
  ve tutarlı sonuç verdi.
- **Çoklu kamera füzyonu:** `dev1` ve `dev2` kamera açılarının tahminleri zaman
  bazında hizalanıp çoğunluk oyuyla birleştiriliyor; kameralar arası uyum oranı
  da her segment için ayrıca hesaplanıp düşükse raporda belirtiliyor.
- **Mobilya tipine özel standart süre:** Standart süre tüm veri setinin değil, o
  videonun mobilya kategorisinin medyanından hesaplanıyor (yeterli veri yoksa
  genel medyana düşüyor) — farklı mobilya tiplerinin doğal olarak farklı montaj
  sürelerine sahip olması nedeniyle bu karşılaştırmayı daha adil hale getiriyor.

## Proje Yapısı

```
01_preprocessing/       Ham video -> düşük-fps tensör (.npy) + metadata
02_feature_extraction/  Donmuş VideoMAE ile video başına sıralı öznitelik çıkarımı
03_model_training/      BiLSTM zaman-adımı sınıflandırıcı eğitimi (temporal_model.pt)
04_inference/           Çoklu kamera füzyonu + standart süre analiziyle çıkarım
05_llm_report/          Analiz çıktısının GPT-4o-mini ile doğal dil raporuna çevrilmesi
```

## Kurulum

```bash
pip install -r requirements.txt
```

- **Veri hazırlığı (01):** Sadece `opencv-python` ve `numpy` gerektirir.
- **Model ve çıkarım (02, 03, 04):** GPU destekli `torch` ve `transformers` gerektirir.
- **Raporlama (05):** Bir OpenAI API key gerektirir (`export OPENAI_API_KEY="sk-..."`).

> **Not:** Bu repo, tensör dosyalarını (`.npy`), eğitilmiş model ağırlıklarını
> (`.pt`) veya ham videoları içermez (`.gitignore` ile hariç tutulmuştur) — bunlar
> onlarca-yüzlerce GB boyutunda, Git/GitHub buna uygun değil. Projeyi sıfırdan
> reprodüklemek için önce [IKEA ASM Dataset](https://ikeaasm.github.io/)'ten ham
> videoları indirip aşağıdaki adımları sırayla çalıştırman gerekir.

## Kullanım

1. **Tensör çıkarımı ve metadata** (yerel, CPU, paralel):
   ```bash
   python 01_preprocessing/video_to_tensor.py --root /path/to/videos \
       --gt_json gt_segments.json --out ./tensors --fps 8 --size 224

   python 01_preprocessing/build_metadata.py --root /path/to/videos \
       --gt_json gt_segments.json --tensors_dir ./tensors --fps 8 \
       --out ./tensors/metadata.json
   ```
2. **Öznitelik çıkarımı** (Colab/GPU önerilir): `02_feature_extraction/extract_sequences.py`
   içindeki yol değişkenlerini kendi ortamına göre düzenleyip çalıştır.
3. **Model eğitimi**:
   ```bash
   python 03_model_training/train_temporal_model.py
   ```
4. **Çıkarım**: `04_inference/inference_report_v2.py` içindeki `VIDEO_KEY`'i
   istediğin videoyla değiştirip çalıştır. Çıktı: `video_analysis_report_multicam.json`.
5. **LLM raporu**:
   ```bash
   export OPENAI_API_KEY=sk-...
   python 05_llm_report/generate_llm_report.py --json_path video_analysis_report_multicam.json
   ```

## Bilinen Sınırlamalar ve Gelecek Çalışma

- Macro F1 (0.44) mükemmel değil; özellikle 1 saniyenin altındaki mikro aksiyonlar
  hâlâ zor.
- Standart süreler veri setindeki medyan üzerinden hesaplanıyor; düşük örneklemli
  sınıflarda (`push table`, n=18 gibi) istatistiksel güvenilirlik sınırlı.
- Sadece IKEA ASM veri setinin mobilyaları için kalibre edilmiş; farklı bir montaj
  türüne genellenmesi test edilmedi.
- Daha yüksek örnekleme hızı (12+ fps) veya CRF/2. katman BiLSTM gibi mimari
  iyileştirmeler zaman/kaynak kısıtı nedeniyle denenmedi.

## Atıf

Bu proje [IKEA ASM Dataset](https://ikeaasm.github.io/) kullanır:

```
Ben-Shabat, Y., Yu, X., Saleh, F., Campbell, D., Rodriguez-Opazo, C., Li, H., & Gould, S. (2021).
The IKEA ASM Dataset: Understanding People Assembling Furniture through Actions, Objects and Pose.
WACV 2021.
```
