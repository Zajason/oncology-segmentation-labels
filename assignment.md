# Αυτοματοποιημένη παραγωγή συνθετικών συνόλων δεδομένων κατάτμησης στιγμιοτύπων σε ογκολογικές ιατρικές εικόνες από δημόσια διαθέσιμα σύνολα δεδομένων

## Περίληψη

Η εργασία εστιάζει στην αυτοματοποιημένη παραγωγή συνθετικών μασκών και polygon-based annotations για προβλήματα instance segmentation στην ογκολογική απεικόνιση, με στόχο τη μείωση του χρόνου και του κόστους χειροκίνητης επισημείωσης.

Ως πεδίο εφαρμογής θα χρησιμοποιηθούν ανοικτά δημόσια σύνολα δεδομένων από τον ογκολογικό τομέα, όπως τα Brain Tumor, BraTS 2020 και BUSI, τα οποία καλύπτουν διαφορετικές απεικονιστικές τροπικότητες και διαφορετικούς τύπους βλαβών/όγκων.

## Στόχος της εργασίας

Βασικός στόχος είναι η ανάπτυξη και αξιολόγηση ενός αυτοματοποιημένου pipeline που, ξεκινώντας από διαθέσιμες επισημειώσεις εντοπισμού ή από υπάρχουσες μάσκες αναφοράς, θα παράγει συνθετικές μάσκες στιγμιοτύπων υψηλής ποιότητας.

Η κατασκευή των μασκών θα βασιστεί σε κλασικές τεχνικές υπολογιστικής όρασης, με ενδεικτικές μεθόδους τις:

- Otsu thresholding
- Multi-Otsu thresholding
- Adaptive/local thresholding
- Watershed
- Random Walker segmentation
- Chan-Vese segmentation
- Morphological Geodesic Active Contours / Morphological Snakes

Στη συνέχεια, οι μάσκες θα μετατρέπονται αυτοματοποιημένα σε polygon labels κατάλληλα για εκπαίδευση μοντέλων τύπου YOLO segmentation.

## Αξιολόγηση

Η ποιότητα των συνθετικών μασκών θα αξιολογηθεί με βάση δείκτες επικάλυψης, κυρίως Intersection over Union (IoU), μέσω σύγκρισης με τις μάσκες που έχουν δοθεί από ειδικούς.

Έπειτα θα πραγματοποιηθεί εκπαίδευση και πειραματική σύγκριση μοντέλων YOLOv11 και YOLOv26 πάνω στα παραγόμενα σύνολα δεδομένων, ώστε να διερευνηθεί ποια μέθοδος ή ποιος συνδυασμός μεθόδων οδηγεί σε πιο αξιόπιστη και πρακτικά αξιοποιήσιμη συνθετική παραγωγή labels για ογκολογικές εφαρμογές.

Η συνολική στόχευση της εργασίας είναι να αναδείξει έναν επαναλήψιμο τρόπο κατασκευής συνθετικών MIS datasets, που να μπορεί να υποστηρίξει μελλοντική έρευνα και ανάπτυξη κλινικών εφαρμογών.

## Λέξεις κλειδιά

Ιατρική απεικόνιση, ογκολογία, brain tumor, BraTS, BUSI, instance segmentation, synthetic masks, polygon annotations, classical computer vision, scikit-image, Watershed, Random Walker, Chan-Vese, YOLOv11, YOLOv26, IoU.

## Βιβλιογραφία

1. Menze, B.H., Jakab, A., Bauer, S., et al. *The Multimodal Brain Tumor Image Segmentation Benchmark (BRATS).* IEEE Transactions on Medical Imaging, 34, 1993-2024, 2015.
2. Bakas, S., Akbari, H., Sotiras, A., et al. *Advancing The Cancer Genome Atlas glioma MRI collections with expert segmentation labels and radiomic features.* Scientific Data, 4, 170117, 2017.
3. Bakas, S., Reyes, M., Jakab, A., et al. *Identifying the Best Machine Learning Algorithms for Brain Tumor Segmentation, Progression Assessment, and Overall Survival Prediction in the BRATS Challenge.* arXiv, 2019.
4. Al-Dhabyani, W., Gomaa, M., Khaled, H., et al. *Dataset of breast ultrasound images.* Data in Brief, 28, 104863, 2020.
5. Jocher, G., Qiu, J., Chaurasia, A. *Ultralytics YOLO.* GitHub repository, 2023.
6. Nguyen, M.T.P., Le, T.N., Tran, M.K.P. *Leukemia Detection Based on YOLOv11 with Global and Local Contexts Interaction.* In Computational Collective Intelligence, Springer, 2026.
7. Beucher, S. *Watershed, Hierarchical Segmentation and Waterfall Algorithm.* In Mathematical Morphology and Its Applications to Image Processing, Springer, 1994.
