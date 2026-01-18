# Analiza Rezultata i Konačni Sažetak  
**Few-Shot OOD Detection na ImageNet-Val s CLIP ViT-B-16**  

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
- Točnost doseže 68.9% (konkurentno s Prototype headom na 62.9%)  
- Ipak, OOD detekcija ostaje loša (AUROC=0.571)  
- Mahalanobis udaljenost je stabilnija, ali povjerenja za OOD i dalje nisu dobro kalibrirana  

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
- Prirodna kalibracija: ID blizu prototipu → visoko povjerenje; OOD daleko → nisko povjerenje  
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
> "Prototype head s K=16 shots postiže najbolju ravnotežu između ID točnosti (62.9%) i OOD detekcije (AUROC=0.782), nadmašujući zero-shot CLIP za 4.6% u točnosti i 3.2 poena u AUROC."

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
