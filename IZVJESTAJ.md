# Motivacija, analiza rezultata i konačni sažetak  
**Few-Shot OOD Detection na ImageNet-Val s CLIP ViT-B-16**  

---

## Motivacija

AI sustavi često trebaju puno podataka da dobro funkcioniraju za dani zadatak. Što više, to bolje. Međutim, za neke stvari/slučajeve jednostavno nema dovoljno podataka da bi se sustav dobro istrenirao i evaluirao. 

**Real-world scenariji:**
- **Medicinska dijagnostika**: Što ako istreniramo model da prepozna 1000 bolesti, ali pacijent dođe sa 1001. koju model nikad nije vidio? U takvom slučaju želimo da model kaže "ne znam" umjesto da s visokom sigurnošću daje pogrešnu dijagnozu.
- **Autonomna vožnja**: Ako istreniramo model za prepoznavanje objekata na cesti i dogodi se situacija koje nema u datasetu (npr. slon pobjegne iz zoološkog vrta i stoji na cesti), vrlo je važno zbog sigurnosti putnika i drugih u blizini da model donese ispravnu odluku - da prepozna da ne zna što vidi i reagira oprezno.
- **Industrijska kontrola kvalitete**: Sustav treniran na normalnim proizvodima mora detektirati defekte koje nikad prije nije vidio, umjesto da ih klasificira kao poznate kategorije.

**Dva ključna izazova:**

1. **Few-Shot Learning**: U praksi često imamo vrlo malo primjera po klasi (K=1-16 uzoraka). Kako naučiti robusne modele s tako malo podataka?

2. **Out-of-Distribution (OOD) Detection**: Kako prepoznati uzorke koji ne pripadaju ni jednoj treniranoj klasi? Model mora znati "kada ne zna".

**Fundamentalni trade-off**: Želimo model koji je istovremeno:
- Točan na poznatim klasama (In-Distribution accuracy)
- Oprezan na nepoznatim uzorcima (OOD detection)

Ova dva cilja često su u suprotnosti - modeli koji su vrlo sigurni na ID podacima često su i previše sigurni na OOD podacima.

---

## Definicija Problema

**Zadatak**: Few-shot klasifikacija s detekcijom out-of-distribution uzoraka

**Setup:**
- **In-Distribution (ID)**: ImageNet-Val dataset
  - 50,000 slika
  - 1000 klasa (od "tench" do "toilet tissue")
  - Split: 80% za trening, 20% za testiranje
  
- **Out-of-Distribution (OOD)**: ImageNet-O dataset
  - 2,000 slika
  - Objekti izvan ImageNet taksonomije
  - Koristi se samo za evaluaciju OOD detekcije

**Few-Shot Regime:**
- K ∈ {0, 1, 2, 4, 8, 16} uzoraka po klasi za trening
- K=0: Zero-shot (samo text embeddings, bez image treninga)
- K=16: Maksimalno 16,000 training uzoraka (16 × 1000 klasa)

**Arhitektura:**
- **Feature Extraction**: CLIP ViT-B-16 (pre-trained)
  - Input: RGB slike 224×224
  - Output: 512-dimenzionalni embedding vektor
  - Normalizirani (||v|| = 1)
  
- **Classification Heads**: Četiri pristupa
  1. **Zero-Shot**: Cosine similarity s text embeddingima
  2. **Prototype**: Distance do class centroids
  3. **Linear Probe**: Linearna transformacija (512 → 1000)
  4. **Gaussian**: Gaussian Discriminant Analysis s Mahalanobis distance

**Evaluacijske Metrike:**

*In-Distribution Performance:*
- **Accuracy**: Postotak točno klasificiranih ID uzoraka
- **ECE** (Expected Calibration Error): Poklapanje confidence s accuracy

*Out-of-Distribution Detection:*
- **AUROC**: Area Under ROC Curve (1.0 = savršeno, 0.5 = random)
- **FPR@95**: False Positive Rate kada je True Positive Rate = 95%
  - Niže je bolje (manje OOD uzoraka prihvaćeno kao ID)

**Cilj**: Identificirati metodu koja postiže najbolji balans između:
1. Visoke ID točnosti (accuracy)
2. Niske OOD false positive rate (FPR@95)
3. Dobre kalibracije (ECE)

**Istraživačka pitanja:**
1. Koja metoda najbolje balansira ID accuracy i OOD detection?
2. Koliko je shotova potrebno da nadmašimo zero-shot CLIP?
3. Zašto neke metode propadaju u few-shot režimu?
4. Kako se confidence distribucije razlikuju između metoda?

---

## 1. Detaljna Analiza Rezultata

### 1.1 Neuspjeh Gaussian Heada

**Problem**  
Gaussian head potpuno ne uspijeva pri K=1, s točnošću od samo 0.11% (praktički slučajno za 1000 klasa). Čak i pri K=16, dok doseže 68.9% točnosti, performanse za OOD detekciju (AUROC=0.571) su tek blago bolje od slučajnih (0.5).  

**Uzroci**

- **Prokletstvo Dimenzionalnosti**  
  - Dimenzija feature-a: 512 (CLIP ViT-B-16)  
  - Parametri kovarijance: 512×512 = 262,144  
  - Dostupni podaci pri K=1: 1 uzorak po klasi  
  - Rezultat: Kovarijacijska matrica je singularna i ne može se pouzdano invertirati → Mahalanobis udaljenost besmislena.  

- **Kršenje pretpostavke vezane kovarijance**  
  Gaussian head pretpostavlja da sve klase dijele istu strukturu kovarijance. U CLIP-ovom semantičkom prostoru to nije istina:  
  - "goldfish" se grupe usko (niska varijanca)  
  - "space shuttle" se znatno razlikuje po izgledu (visoka varijanca)  

- **Nedovoljna Shrinkage Regularizacija**  
  Čak i s λ=0.5, regularizacija ne nadoknađuje manjak podataka. Pri K=1 matrica postaje skoro čisto dijagonalna, što ne hvata stvarnu strukturu klasa.  

**Zašto se performanse poboljšavaju pri većim K**  

- Pri K=16, 16 uzoraka po klasi daje bolje procjene kovarijance  
- Točnost doseže 68.9% (najviša od svih metoda)  
- Ipak, OOD detekcija ostaje loša (AUROC=0.571)  
- Mahalanobis udaljenost je stabilnija, ali povjerenja za OOD i dalje nisu dobro kalibrirana

**Overconfidence Problem:**  
- **K=1**: Confidence ~0.01 za obje distribucije (potpuna nesigurnost, near-uniform)
- **K≥2**: Dramatičan skok na confidence ~0.99 za obje distribucije (ekstremna sigurnost)
- Model prelazi direktno iz "ne znam ništa" u "znam sve" - nema middle ground
- **Rezultat**: ID i OOD imaju identične confidence distribucije → nemoguće razlikovati
- Mahalanobis distance + softmax → oštre odluke → overconfident predictions
- Objašnjava AUROC~0.57 (jedva bolje od random) usprkos visokoj accuracy
- **Paradoks**: Najbolja kalibracija (ECE=0.299) jer je konzistentno overconfident  

---

### 1.2 Točnost vs OOD Detekcije

| Metoda        | K=16 Točnost | K=16 AUROC | K=16 FPR@95 |
|---------------|--------------|------------|-------------|
| Linear Probe  | 52.5%        | 0.752      | 86.6%       |
| Prototype     | 62.9%        | 0.782      | 79.3%       |
| Gaussian      | 68.9%        | 0.571      | 91.5%       |

**Ključna spoznaja:**  
- Pri K=1, Linear Probe ima malu prednost u točnosti (30.2% vs 29.6%).  
- Pri K≥8, Prototype dominira u **točnosti i OOD detekciji**.  

**Zašto Linear Probe uspijeva pri malom K**  
- Discriminativno treniranje optimizira cross-entropy  
- 100 epoha s Adam optimizerom  
- Weight decay 1e-2  
- Rizik od overfittinga: granice se mogu precizno prilagoditi malim ID uzorcima  

**Zašto Prototype dominira pri višem K**  
- Čuva CLIP-ovu semantičku strukturu (prosjek embeddinga)  
- Bolja separacija confidence distribucija: Iako su apsolutne vrijednosti niske (~0.001 zbog 1000 klasa), ID i OOD distribucije su bolje odvojene nego kod drugih metoda
- Konzistentnija OOD detekcija: Uža varijanca OOD scorova → manje preklapanje s ID  
- Nema overfittinga (nema learnable parametara)  

**Zašto Linear Probe ne uspijeva u OOD**  
- Granice naučene za ID → previše samouvjereni na OOD  
- Primjer: ImageNet-O "tube" može biti klasificiran kao "trombone"  

---

### 1.3 Kalibracija (ECE)

**Temperature Scaling:**  
- Prosječni ECE: 0.413 → loša kalibracija  
- K=16 kalibracija po metodi:  
  - Gaussian: 0.299 (najbolje)  
  - Linear Probe: 0.524  
  - Prototype: 0.628 (najgore)  

**Zaključak:**  
- Iako Prototype ima najbolji AUROC, visoki ECE znači da sirovi confidence score možda ne odražava stvarnu vjerojatnost.  
- Temperature tuning može dodatno poboljšati kalibraciju.  

---

### 1.4 Zero-Shot Baseline

- Accuracy: 58.3%  
- AUROC: 0.750  
- FPR@95: 84.3%  

**Značaj:**  
- Zero-shot CLIP nadmašuje sve metode s malim K (1-4)  
- Few-shot metode nadmašuju zero-shot tek pri K≥8

**Confidence karakteristike:**
- Niske apsolutne vrijednosti (~0.0015 za ID, ~0.0012 za OOD)
- Razlog: 1000 klasa → uniformna distribucija = 0.1% po klasi
- Model je "blago iznad uniformne" - točan je (58.3%), ali nije siguran
- Gaussove distribucije s značajnim preklapanjem
- Ovo je tipično za zero-shot na velikom broju klasa  

---

### 1.5 Vizualizacije

- 67 plotova ukupno:  
  - 16 Reliability Diagrams  
  - 16 PR Curves  
  - 16 Retained Accuracy vs Rejection  
  - 3 t-SNE embeddings  

---

### 1.6 Preporuke

- **Najbolji model:** Prototype head, K=16  
  - Accuracy: 62.9%  
  - AUROC: 0.782  
  - FPR@95: 79.3%  

**Preporuka po broju shotova:**

| Shots | Preporučena metoda | Razlog |
|-------|------------------|--------|
| K=0   | Zero-Shot CLIP   | 58.3% točnosti, bez treninga |
| K=1-4 | Zero-Shot / Linear Probe | Few-shot još nema prednost |
| K=8-16| Prototype        | Najbolja kombinacija ID točnosti + OOD detekcija |
| K≥16  | Prototype / Gaussian | Gaussian sustiže u točnosti |

**Ključna poruka:**  
> "Prototype head s K=16 shots postiže najbolju ravnotežu između ID točnosti (62.9%) i OOD detekcije (AUROC=0.782), nadmašujući zero-shot CLIP za 4.6% u točnosti i 3.2 poena u AUROC. Poboljšanja dolaze od konzistentnije i uže OOD distribucije, ne od drastično različitih confidence vrijednosti - sve metode (osim Gaussian) pokazuju niske apsolutne confidence zbog velikog broja klasa (1000)."

---

## 2. Konačni Sažetak Rezultata

**Performanse po metodama (prosjek 3 sjemena):**

| K-shot | Metoda       | Accuracy | AUROC | FPR@95 | ECE  |
|--------|-------------|---------|-------|--------|------|
| 0      | ZeroShot    | 58.3%   | 0.750 | 84.3%  | 0.582 |
| 1      | LinearProbe | 30.2% ±0.5 | 0.693 ±0.005 | 89.5% ±0.2 | 0.301 ±0.005 |
| 1      | Prototype   | 29.6% ±0.5 | 0.692 ±0.004 | 89.9% ±0.5 | 0.294 ±0.005 |
| 1      | Gaussian    | 0.1% ±0.0  | 0.500 ±0.000 | 100% ±0.0   | 0.000 ±0.0 |
| 2      | LinearProbe | 37.0% ±0.5 | 0.718 ±0.007 | 89.2% ±0.4 | 0.369 ±0.005 |
| 2      | Prototype   | 41.0% ±0.4 | 0.731 ±0.003 | 86.4% ±0.7 | 0.409 ±0.004 |
| 2      | Gaussian    | 46.2% ±0.7 | 0.528 ±0.009 | 92.9% ±0.6 | 0.532 ±0.007 |
| 4      | LinearProbe | 43.8% ±0.5 | 0.740 ±0.007 | 87.5% ±1.6 | 0.437 ±0.005 |
| 4      | Prototype   | 52.0% ±0.3 | 0.760 ±0.003 | 82.2% ±1.1 | 0.519 ±0.003 |
| 4      | Gaussian    | 58.0% ±0.5 | 0.548 ±0.004 | 91.8% ±0.4 | 0.411 ±0.006 |
| 8      | LinearProbe | 49.3% ±0.3 | 0.747 ±0.002 | 87.2% ±1.2 | 0.492 ±0.003 |
| 8      | Prototype   | 58.6% ±0.4 | 0.773 ±0.002 | 80.8% ±0.8 | 0.585 ±0.004 |
| 8      | Gaussian    | 65.0% ±0.2 | 0.564 ±0.004 | 92.5% ±0.7 | 0.339 ±0.002 |
| 16     | LinearProbe | 52.5% ±0.4 | 0.752 ±0.003 | 86.6% ±0.7 | 0.524 ±0.004 |
| 16     | Prototype   | 62.9% ±0.4 | 0.782 ±0.001 | 79.3% ±0.2 | 0.628 ±0.004 |
| 16     | Gaussian    | 68.9% ±0.4 | 0.571 ±0.004 | 91.5% ±0.6 | 0.299 ±0.005 |

**Zaključci**

1. **Gaussian Head katastrofalan pri K=1**  
   - 0.11% točnosti  
   - Uzrok: visoka dimenzionalnost + premalo uzoraka  

2. **Zero-shot CLIP dominira pri malom K**  
   - K<8: Zero-shot bolji od few-shot metoda  

3. **Prototype vs Linear Probe**  
   - Linear Probe: prednost pri K=1  
   - Prototype: bolji za K≥4 u ID točnosti i OOD detekciji  

4. **Kalibracija (ECE)**  
   - Prosjek: 0.413 → loša  
   - Gaussian: najbolje kalibriran (ECE=0.299)  
   - Prototype: najlošije kalibriran (ECE=0.628)  


**Zaključak:**  
- **Prototype head s K=16** postiže najbolju ravnotežu ID točnosti i OOD detekcije  
- Distance-based metode u CLIP-ovom semantičkom prostoru nadmašuju discriminative metode u few-shot OOD zadacima  
- Gaussian head pri malom K ilustrira izazov **curse of dimensionality**  
- Zero-shot CLIP je praktičan za K<8  

---
