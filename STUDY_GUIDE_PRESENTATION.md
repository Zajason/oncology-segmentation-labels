# Study Guide — Παρουσίαση Πτυχιακής Εργασίας

**Θέμα:** Αυτοματοποιημένη παραγωγή συνθετικών συνόλων δεδομένων κατάτμησης στιγμιοτύπων σε ογκολογικές ιατρικές εικόνες
**Διάρκεια:** ~10 λεπτά · **Slides:** 11 · **Παρουσιαστές:** 4

---

## Κατανομή Παρουσιαστών

| # | Slides | Διάρκεια | Περιεχόμενο |
|---|---|---|---|
| **Άτομο 1** | 1, 2, 3 | ~2 λεπτά | Τίτλος · Στόχος & Pipeline · Datasets |
| **Άτομο 2** | 4, 5 | ~2 λεπτά | Μέθοδοι · Engineering & Υλοποίηση |
| **Άτομο 3** | 6, 7, 8 | ~3 λεπτά | BUSI Results · BraTS Results · U-Net Training |
| **Άτομο 4** | 9, 10, 11 | ~3 λεπτά | YOLO Phase 2 · Brain Tumor · Συμπεράσματα |

---

## Βασικά νούμερα που πρέπει να ξέρουν ΟΛΟΙ (cheat sheet)

| Dataset | Cases | Best Method | Best IoU |
|---|---|---|---|
| BUSI | 647 | U-Net | **0.909** |
| BraTS 2020 | 24.422 | Guided U-Net | **0.856** |
| Brain Tumor | 7.200 | — (no GT) | N/A |

| Phase 2 YOLO M_mAP50 | Best Result |
|---|---|
| BUSI | **0.888** (Guided U-Net + YOLOv26x) |
| BraTS | **0.617** (Guided U-Net + YOLOv11x) |
| Brain Tumor | **0.757** (U-Net DT + YOLOv11x) |

**12 μέθοδοι σε 3 ομάδες:**
- Classical CV: Otsu, Multi-Otsu, Adaptive, Watershed, Otsu+Watershed, Connected Components, Random Walker, Chan-Vese, Morph GAC (9)
- Foundation Model: SAM (1)
- Learned: U-Net, Guided U-Net (2)

---

# 👤 ΑΤΟΜΟ 1 — Slides 1, 2, 3 (~2 λεπτά)

## Slide 1: Τίτλος (~20 δευτ.)

**Τι λέω:**
> Καλησπέρα. Η εργασία μας αφορά την αυτοματοποιημένη παραγωγή συνθετικών segmentation masks σε ογκολογικές ιατρικές εικόνες. Το κεντρικό ερώτημα: μπορούμε να αντικαταστήσουμε τη χειροκίνητη επισημείωση ειδικών με αλγοριθμικά παραγόμενα labels; Σε 10 λεπτά θα δούμε πώς το κάναμε και τι μετρήσαμε.

**Key points:**
- 3 datasets · 12 μέθοδοι · YOLO training
- Δουλειά πάνω σε IEEE Access resubmission

## Slide 2: Στόχος & Pipeline (~50 δευτ.)

**Τι λέω:**
> Στόχος μας: ένα επαναλήψιμο pipeline που παίρνει δημόσια oncology datasets, παράγει synthetic masks με 12 μεθόδους, τις μετατρέπει σε YOLO polygon labels, και τέλος εκπαιδεύει YOLO segmentation μοντέλα.
>
> Δεξιά βλέπετε τα 7 βήματα: ξεκινάμε από raw dataset, φτιάχνουμε manifest, παίρνουμε ή παράγουμε boxes, τρέχουμε τις 12 μεθόδους, μετατρέπουμε masks σε polygons, εκπαιδεύουμε YOLO, και τέλος μετράμε. Η αξιολόγηση γίνεται σε 2 επίπεδα: IoU vs expert masks στο Phase 1, και mAP από YOLO στο Phase 2.

**Key points:**
- 7 βήματα pipeline (να τα ξέρω στη σειρά)
- Polygon conversion: `cv2.findContours`
- 2 επίπεδα αξιολόγησης

**Πιθανή ερώτηση:** _Γιατί χρειάζονται και τα 2 επίπεδα;_
→ IoU μετράει την ποιότητα της μάσκας απευθείας. Το mAP δείχνει αν τα synthetic labels είναι ωφέλιμα για downstream training (πραγματικό use case).

## Slide 3: Datasets (~50 δευτ.)

**Τι λέω:**
> Δουλέψαμε με 3 δημόσια oncology datasets, 3 διαφορετικές τροπικότητες.
>
> **BUSI**: ultrasound μαστού, 647 εικόνες, έχουμε expert GT masks για όλες — άρα μπορούμε να μετρήσουμε IoU απευθείας.
>
> **BraTS 2020**: MRI γλοίωμα, 24.422 2D axial slices που εξάγαμε από τα 3D volumes — χρησιμοποιήσαμε το FLAIR channel. Έχουμε expert voxel masks.
>
> **Brain Tumor MRI**: 7.200 εικόνες, αλλά **δεν έχει expert segmentation masks** — μόνο class labels (glioma, meningioma, pituitary, notumor). Άρα δεν μπορούμε να αναφέρουμε IoU για αυτό — θα ήταν λάθος. Η στρατηγική για το Brain Tumor θα εξηγηθεί στο slide 10.

**Key points:**
- BUSI = ultrasound, BraTS = MRI, Brain Tumor = MRI (αλλά διαφορετικό dataset)
- **Μόνο BUSI & BraTS έχουν GT → IoU**
- Brain Tumor → όχι IoU, διαφορετική στρατηγική

**Πιθανές ερωτήσεις:**
- _Γιατί δεν χρησιμοποιήσατε όλα τα MRI channels του BraTS;_ → FLAIR είναι το πιο ενημερωτικό για όγκους. Επικέντρωση σε 2D για να ταιριάζει με YOLO input.
- _Πόσα slices ανά volume στο BraTS;_ → 155 axial slices ανά volume, αλλά κρατήσαμε μόνο όσα έχουν GT tumor pixels.

---

# 👤 ΑΤΟΜΟ 2 — Slides 4, 5 (~2 λεπτά)

## Slide 4: Μέθοδοι Παραγωγής Μασκών (~60 δευτ.)

**Τι λέω:**
> Υλοποιήσαμε 12 μεθόδους με **ακριβώς κοινό interface**: `generate_masks(image, boxes, config) → list of masks`. Αυτό επιτρέπει δίκαια σύγκριση και plug-and-play.
>
> Τις χωρίζουμε σε 3 ομάδες:
>
> **Classical CV (9):** Otsu, Multi-Otsu, Adaptive thresholding, Watershed, Otsu+Watershed, Connected Components, Random Walker, Chan-Vese, Morphological GAC. Όλα CPU, χωρίς training, άμεσα explainable.
>
> **Foundation Model (1):** SAM από τη Meta — pre-trained, bbox-prompted, χωρίς fine-tuning. ViT-B checkpoint.
>
> **Learned (2):** U-Net vanilla (πλήρης εποπτεία — oracle baseline) και Guided U-Net (box-supervised, βάσει Bilic & Egger 2023). Αυτές οι 2 χρειάζονται training.
>
> Η επιλογή των μεθόδων δεν ήταν τυχαία — απαντά απευθείας σε αιτήματα MICCAI reviewers: SAM (Reviewer #2), ConnectedComponents και Guided U-Net (Reviewer #1).

**Key points:**
- Common interface: ίδιο signature για όλες
- 9 + 1 + 2 = 12 μέθοδοι
- Bilic & Egger 2023 = reference για Guided U-Net (IEEE SMC, DOI 10.1109/SMC53992.2023.10324276)

**Πιθανές ερωτήσεις:**
- _Γιατί SAM ViT-B και όχι ViT-L;_ → ViT-B χωράει σε T4 GPU, αρκετά καλό για bbox prompts.
- _Τι διαφορά έχει το Guided U-Net από το vanilla;_ → Παίρνει image + bbox encoding ως input, εκπαιδεύεται σε triplets (image, bbox, GT mask).

## Slide 5: Engineering & Υλοποίηση (~50 δευτ.)

**Τι λέω:**
> Η υλοποίηση είναι μηχανικά σωστή, όχι μόνο αλγοριθμικά. Πάνω βλέπετε το pipeline: `build_manifests.py` ενοποιεί τα 3 διαφορετικά dataset formats σε κοινό CSV — έτσι ο `phase1_runner.py` μπορεί να τρέξει κάθε μέθοδο χωρίς αλλαγές. Μετά γίνεται evaluation και τέλος YOLO training.
>
> 4 βασικές μηχανικές συνεισφορές:
>
> 1. **Manifest builders** — ενιαία CSV δομή για όλα τα datasets, επαναλήψιμο σε νέα.
> 2. **Model caching** — SAM, U-Net, Guided U-Net φορτώνονται μία φορά, όχι ανά εικόνα. Δραματική επιτάχυνση.
> 3. **Early stopping στο YOLO** — patience αντί fixed 50 epochs. Αποφυγή overfitting και εξοικονόμηση compute.
> 4. **Auto polygon export** — `cv2.findContours` → YOLO `.txt` labels αυτόματα. Απαντά απευθείας σε MICCAI Reviewer #2.

**Key points:**
- Common interface = δίκαια σύγκριση
- Model caching → ταχύτητα
- Early stopping → αποφυγή spurious epochs
- `cv2.findContours` αντί custom polygon code

**Πιθανές ερωτήσεις:**
- _Πόση επιτάχυνση από το model caching;_ → Δεν έχουμε μετρήσει επακριβώς, αλλά τάξη μεγέθους — το U-Net χρειάζεται δευτερόλεπτα για φόρτωση.
- _Τι patience χρησιμοποιήσατε;_ → Default ultralytics (patience=50 στο YOLO, αλλά συνολικά epochs = 50 max).

---

# 👤 ΑΤΟΜΟ 3 — Slides 6, 7, 8 (~3 λεπτά)

## Slide 6: BUSI Results (~60 δευτ.)

**Τι λέω:**
> Αποτελέσματα Phase 1 στο BUSI, 647 εικόνες, σύγκριση όλων των 12 μεθόδων vs expert masks.
>
> Πρώτο το **U-Net** με IoU **0.909** — σχεδόν τέλεια επικάλυψη. Από κοντά το **Guided U-Net** στο **0.896**.
>
> Τρίτο το **SAM** στο **0.777** — αξιόλογο χωρίς καθόλου training. Από classical μεθόδους η καλύτερη είναι το **Morph GAC** με 0.747, αλλά κοστίζει 329 δευτερόλεπτα runtime.
>
> Στο κάτω άκρο: Multi-Otsu μόλις 0.052 — εντελώς ακατάλληλο για ultrasound.
>
> Το κύριο συμπέρασμα: τα learned methods ξεχωρίζουν καθαρά.

**Key numbers (BUSI IoU):**
| Method | IoU | Dice |
|---|---|---|
| U-Net | **0.909** | 0.952 |
| Guided U-Net | 0.896 | 0.944 |
| SAM | 0.777 | 0.868 |
| Morph GAC | 0.747 | 0.850 |
| Chan-Vese | 0.702 | 0.814 |
| Watershed | 0.625 | 0.754 |
| Multi-Otsu | 0.052 | 0.094 |

**Πιθανή ερώτηση:** _Γιατί το vanilla U-Net νικά το Guided στο BUSI;_
→ Το vanilla έχει πλήρη εποπτεία (oracle), ενώ το Guided παίρνει το bbox ως extra signal. Η διαφορά είναι μικρή (0.013) — εντός θορύβου.

## Slide 7: BraTS Results (~60 δευτ.)

**Τι λέω:**
> Στο BraTS, 24.422 slices, το μοτίβο επαναλαμβάνεται αλλά με μια ενδιαφέρουσα αλλαγή.
>
> Πρώτο τώρα είναι το **Guided U-Net** με IoU **0.856**, και δεύτερο το **U-Net** στο **0.851**. Πρακτικά ισόπαλα — μέσα στο θόρυβο.
>
> Τρίτο πάλι το SAM στο 0.658, αλλά εδώ το gap από τα learned είναι πιο μεγάλο — MRI είναι πιο δύσκολο πρόβλημα.
>
> Οι classical μέθοδοι εδώ είναι πιο ομοιόμορφες, 0.48–0.64. Η ίδια η δυσκολία του MRI μοιράζεται σε όλους.
>
> **Το κρίσιμο:** και στα 2 GT datasets, τα learned methods κερδίζουν καθαρά. Αυτή η consistency είναι η βάση για το domain transfer στο Brain Tumor — αν δεν είχαμε αυτή τη σύγκλιση, η μεταφορά θα ήταν αμφίβολη.

**Key numbers (BraTS IoU):**
| Method | IoU | Dice |
|---|---|---|
| Guided U-Net | **0.856** | 0.919 |
| U-Net | 0.851 | 0.917 |
| SAM | 0.658 | 0.784 |
| Adaptive | 0.638 | 0.757 |
| Otsu+Watershed | 0.632 | 0.750 |

**Πιθανή ερώτηση:** _Γιατί άλλαξε η σειρά (Guided > Vanilla);_
→ Στο MRI με πιο σύνθετο shape, το bbox prompt του Guided δίνει meaningful spatial guidance. Στο ultrasound με πιο localized βλάβες, το vanilla τα καταφέρνει ακριβώς το ίδιο καλά.

## Slide 8: U-Net Training (~50 δευτ.)

**Τι λέω:**
> Πριν χρησιμοποιήσουμε τα learned models σε production, τα εκπαιδεύσαμε και αξιολογήσαμε.
>
> 4 training runs, 20 epochs το καθένα. Το γράφημα δείχνει το validation IoU ανά epoch.
>
> **BUSI U-Net**: best val IoU **0.8726** στο epoch 19.
> **BUSI Guided U-Net**: **0.8730** στο epoch 17 — οριακά καλύτερα.
> **BraTS U-Net**: 0.8178 στο epoch 15.
> **BraTS Guided U-Net**: **0.8206** στο epoch 15 — επίσης οριακά καλύτερα.
>
> Το κρίσιμο μήνυμα: η επιλογή των U-Net / Guided U-Net δεν ήταν τυφλή. Είδαμε ότι συγκλίνουν σταθερά και ξεπερνούν τις άλλες μεθόδους ποσοτικά.

**Key numbers (Best Val IoU):**
- U040 BUSI U-Net: 0.8726 @ epoch 19
- U042 BUSI Guided: 0.8730 @ epoch 17
- U041 BraTS U-Net: 0.8178 @ epoch 15
- U043 BraTS Guided: 0.8206 @ epoch 15

**Πιθανές ερωτήσεις:**
- _Γιατί 20 epochs;_ → Είδαμε convergence πριν το 20 σε όλα τα runs.
- _Loss function;_ → Standard για segmentation: Dice loss + BCE.
- _Validation split;_ → Στατικό 80/20, ίδιο seed για όλα τα runs.

---

# 👤 ΑΤΟΜΟ 4 — Slides 9, 10, 11 (~3 λεπτά)

## Slide 9: YOLO Phase 2 (~60 δευτ.)

**Τι λέω:**
> Phase 2: παίρνουμε τις synthetic masks των top-2 μεθόδων (U-Net, Guided U-Net) και τις χρησιμοποιούμε ως labels για να εκπαιδεύσουμε YOLO segmentation models.
>
> **BUSI**: Όλα τα runs πάνω από 0.87 M_mAP50. Καλύτερο: Guided U-Net + **YOLOv26x-seg** με **M_mAP50 = 0.888**. Είναι εξαιρετικό για synthetic labels.
>
> **BraTS**: Χαμηλότερα νούμερα, 0.58–0.62. Καλύτερο: Guided U-Net + YOLOv11x με **0.617**. Αυτό είναι αναμενόμενο — MRI segmentation είναι πιο δύσκολο πρόβλημα.
>
> **Η κύρια μετρική είναι το Mask mAP (M_mAP), όχι Box mAP.** Αυτό δείχνει την ποιότητα των πραγματικών segmentation masks του μοντέλου.
>
> Το σημαντικό: τα synthetic labels του pipeline είναι αρκετά συνεπή ώστε να εκπαιδεύσουν λειτουργικά YOLO models.

**Key numbers (Phase 2 best M_mAP50):**
| Dataset | Best | Method + Model |
|---|---|---|
| BUSI | **0.888** | Guided U-Net + YOLOv26x |
| BraTS | **0.617** | Guided U-Net + YOLOv11x |

**Πιθανή ερώτηση:** _Γιατί δεν δοκιμάσατε άλλες μεθόδους και στο Phase 2;_
→ Όπως ορίζει το brief, παίρνουμε **top-2 ανά dataset** για να μη σπαταλήσουμε GPU σε μεθόδους που ξέρουμε ότι θα αποτύχουν.

## Slide 10: Brain Tumor (~70 δευτ.)

**Τι λέω:**
> Το Brain Tumor είναι η ειδική περίπτωση. **Δεν έχει expert masks** — οπότε δεν υπολογίζουμε IoU. Θα ήταν επιστημονικά λάθος.
>
> Η στρατηγική μας σε 3 σημεία:
>
> 1. **Domain transfer**: το BraTS-trained U-Net checkpoint κάνει inference στο Brain Tumor. Κοινή pathology (brain MRI), οπότε η μεταφορά έχει νόημα.
> 2. **Weak box prompts**: full-image boxes για tumor classes, empty για notumor.
> 3. **Indirect validation**: αν οι masks ήταν κακές, το downstream YOLO θα αποτύχαινε.
>
> **Και πράγματι, τα νέα αποτελέσματα Phase 2 επιβεβαιώνουν τη στρατηγική:**
>
> - **U-Net + YOLOv11x → M_mAP50 = 0.757**
> - U-Net + YOLOv26x → 0.756
> - Guided U-Net + YOLOv11x → 0.434
> - Guided U-Net + YOLOv26x → 0.428
>
> Δύο σημαντικά findings:
> 1. Το U-Net (domain transfer) λειτούργησε — 0.757 είναι εξαιρετικό για dataset χωρίς GT.
> 2. Το Guided U-Net υποαποδίδει εδώ — γιατί χρειάζεται meaningful boxes, όχι full-image weak prompts. **Αυτό επικυρώνει γιατί επιλέξαμε U-Net για το domain transfer.**

**Key numbers (Brain Tumor YOLO M_mAP50):**
| Method | Model | M_mAP50 |
|---|---|---|
| **U-Net (DT)** | **YOLOv11x** | **0.757** |
| U-Net (DT) | YOLOv26x | 0.756 |
| Guided U-Net | YOLOv11x | 0.434 |
| Guided U-Net | YOLOv26x | 0.428 |

**Πιθανές ερωτήσεις:**
- _Γιατί δεν εκπαιδεύσατε U-Net κατευθείαν στο Brain Tumor;_ → Δεν υπάρχουν GT masks για supervision. Domain transfer από BraTS είναι το σωστό path.
- _Γιατί διαφέρουν τόσο πολύ U-Net vs Guided στο Brain Tumor;_ → Το Guided περιμένει meaningful bbox prompts. Με full-image boxes χάνει νόημα και το model κάνει random predictions.
- _Πώς ξέρετε ότι η stratregy δούλεψε χωρίς GT;_ → Το YOLO mAP είναι έμμεση μέτρηση. 0.757 mAP δεν θα μπορούσε να βγει αν οι synthetic masks ήταν τυχαίες.

## Slide 11: Συμπεράσματα (~30 δευτ.)

**Τι λέω:**
> Συνοψίζω σε 6 σημεία:
>
> 1. **Επαναλήψιμο pipeline** από raw dataset μέχρι YOLO training.
> 2. **Ποσοτική επιβεβαίωση** σε 25.000+ cases — U-Net κερδίζει σε IoU.
> 3. **U-Net & Guided U-Net** consistent νικητές και στα 2 GT datasets.
> 4. **YOLO validation σε 3 datasets**: 0.888 / 0.617 / 0.757 mAP.
> 5. **Brain Tumor** — επιστημονική ειλικρίνεια χωρίς ψευδές IoU.
> 6. **Επόμενα**: SAM-Med2D, qualitative figures, IEEE Access submission.
>
> Ευχαριστούμε για την προσοχή σας. Είμαστε στη διάθεσή σας για ερωτήσεις.

---

## Q&A — Πιθανές δύσκολες ερωτήσεις

**Q: Γιατί χρειαζόμαστε synthetic masks αν έχουμε expert masks σε BUSI/BraTS;**
→ Το pipeline δοκιμάζεται εκεί που έχουμε GT για να μετρηθεί. Ο σκοπός είναι να εφαρμοστεί σε *νέα* datasets που δεν έχουν GT — π.χ. το Brain Tumor.

**Q: Πόσο γενικεύσιμο είναι το pipeline;**
→ Το common interface επιτρέπει εύκολη προσθήκη νέων datasets και νέων μεθόδων. Πρακτικά: χρειάζεται νέο manifest builder ανά νέο dataset format.

**Q: Γιατί cv2.findContours και όχι κάτι πιο sophisticated;**
→ Απάντηση σε MICCAI Reviewer #2. Το `findContours` είναι standard, fast, και έχει validated implementation. Δεν χρειάζεται custom solution.

**Q: Είναι ηθικό/ασφαλές να χρησιμοποιούμε synthetic labels σε clinical setting;**
→ Όχι ως άμεσο clinical decision. Το pipeline είναι research tool για να μειωθεί το κόστος annotation και να επιταχυνθεί η έρευνα. Τελική κλινική απόφαση πάντα από ιατρό.

**Q: Πώς ξέρετε ότι δεν υπερπροσαρμόζεται (overfitting);**
→ Validation IoU είναι held-out. Phase 2 YOLO χρησιμοποιεί early stopping. Cross-dataset consistency (BUSI ↔ BraTS) είναι extra validation.

**Q: Ποιο είναι το computational cost;**
→ Classical: CPU, λίγα δευτερόλεπτα ανά image. SAM: GPU inference. U-Net training: ~2-4 ώρες ανά run στο T4. YOLO training: ~15-45 min ανά run BUSI, ~6-10 ώρες ανά BraTS run.

---

## Χρονισμός σε Δοκιμή

Δοκιμάστε εκ των προτέρων **2 φορές** με ρολόι:
- 1η φορά: ελεύθερη ανάγνωση → βλέπεις αν χωράει
- 2η φορά: live, με χρονόμετρο ανά slide

**Στόχος ανά άτομο:**
- Άτομο 1: 2:00
- Άτομο 2: 2:00
- Άτομο 3: 3:00
- Άτομο 4: 3:00
- **Σύνολο: 10:00**

Αν ξεπεράσεις, κόψε από τις παρομοιώσεις και τις λεπτομέρειες — όχι από τα νούμερα.
