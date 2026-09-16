# Video-Pipeline: Preprocessing, Training, Evaluation

Komplett mit synthetischen 2-Sekunden-Clips durchgetestet. Vor dem ersten echten
Lauf `config.json` an deine Klassennamen und Trim-Fenster anpassen.

## Erwartete Ordnerstruktur

```
data/raw/<klassenname>/<video_id>.mp4
```

`<klassenname>` muss einem Key im `"classes"`-Block von `config.json` entsprechen.

## Ausführungsreihenfolge

```bash
# 1. Erst klein testen
python trim_and_convert.py --config config.json --limit_per_class 2
python preview_clip.py data/processed/<klassenname>/<video_id>.npy

# 2. Dann Preprocessing komplett
python trim_and_convert.py --config config.json
python split_dataset.py --manifest data/manifest.csv --out_dir data

# 3. Step 0 aus dem Trainingsdiagramm — genau EINMAL
python extract_features.py

# 4. Trainingsschleife — beliebig oft, während du Hyperparameter tunst
python train_classifier.py
python plot_training_history.py            # NUR nach diesem konkreten Trainingslauf gültig

# 5. Evaluation — genau EINMAL, ganz am Ende
python evaluate.py
python training_report.py                  # train vs. val vs. test Vergleich (Overfitting-Check)
```

## Erst mit 1-3 Videos testen

`--limit_per_class 2` verarbeitet nur die ersten 2 Videos pro Klasse (alphabetisch
nach Dateiname). `preview_clip.py` speichert ein paar gleichmäßig verteilte Frames
eines verarbeiteten Clips nebeneinander als PNG — damit prüfst du per Auge, ob
Farben (RGB, nicht BGR) und Trimm-Fenster stimmen, bevor du den ganzen Datensatz
durchlaufen lässt.

## Die Dateien

**Preprocessing**
- `config.json` — Trim-Fenster (`start_sec`/`end_sec`) pro Klasse und `target_num_frames`.
- `trim_and_convert.py` — Stage 1. Ein Skript, config-gesteuert (nicht eins pro Klasse).
  Dekodiert jeden Clip, sampelt `target_num_frames` Frames gleichmäßig aus dem
  Trim-Fenster der Klasse, konvertiert BGR→RGB, schreibt ein `.npy` pro Clip und
  `data/manifest.csv`. Hat `--limit_per_class` für den Testlauf.
- `split_dataset.py` — Stage 2. Stratifizierter train/val/test-Split, gruppiert nach
  `group_id` (Standard: Dateiname). Falls mehrere Clips aus derselben Aufnahme/Person
  stammen, `group_id` upstream auf diese gemeinsame ID setzen, sonst leaken sie
  über die Splits.
- `preview_clip.py` — visuelle Kontrolle eines verarbeiteten Clips (optional).

**Training (Frozen Backbone)**
- `video_dataset.py` — `VideoClipDataset` + Transforms. Wird von `extract_features.py`
  importiert, nicht direkt ausgeführt.
- `extract_features.py` — Step 0, läuft einmal. Friert R3D-18 ein (`fc` → `nn.Identity()`),
  schickt jeden Test-/Val-Clip einmal (deterministischer Center-Crop) und jeden Train-Clip
  `1 + N_AUGMENTED_COPIES` mal (Center-Crop + N zufällig gecroppte/geflippte Varianten)
  durch, speichert je einen 512-d Vektor nach `data/features/<split>/...` und schreibt
  `data/{split}_features.csv` + `data/class_to_idx.json`. `--n_augmented` steuert die Anzahl
  (Standard 4). val/test bekommen absichtlich NIE augmentierte Kopien — sonst sind die
  Auswertungszahlen nicht mehr ehrlich. Neu ausführen bei Änderungen an Trim-Fenstern,
  `CROP_SIZE`, `N_AUGMENTED_COPIES` oder Backbone.
- `metrics_utils.py` — gemeinsame Metrik-Funktionen (Accuracy, F1 macro/weighted,
  ROC-AUC und PR-AUC one-vs-rest). Wird von Training, Evaluation und Report importiert,
  damit alle drei exakt gleich rechnen. Nicht direkt ausgeführt.
- `feature_dataset.py` — `FeatureDataset`, lädt die gecachten Feature-Vektoren.
  Wird von `train_classifier.py` importiert, nicht direkt ausgeführt.
- `train_classifier.py` — Trainingsschleife. `Dropout -> Linear` (ohne explizites Softmax,
  weil `nn.CrossEntropyLoss` das intern macht), Adam, Early Stopping. Speichert
  `data/best_classifier.pt` nur bei verbessertem val_loss. Schreibt außerdem JEDE Epoche
  `data/training_history.csv` (train/val loss, accuracy, macro-F1 sowie val ROC-AUC
  und PR-AUC) und am Ende
  `data/training_summary.json` neu — das MUSS während des Laufs passieren, das lässt
  sich danach nicht rekonstruieren. Billig wiederholt auszuführen, weil das Backbone
  nie neu läuft.
- `plot_training_history.py` — liest `training_history.csv` und zeichnet Loss- und
  Accuracy-Kurven (train + val) sowie ein drittes Panel mit F1 und val ROC-AUC als PNG. Kann JEDERZEIT danach laufen, solange die CSV
  vom aktuellen Trainingslauf existiert.
- `training_report.py` — Loss, Accuracy, macro-F1 und ROC-AUC auf train, val UND test mit dem trainierten
  Klassifikator, in einer Tabelle. Kann jederzeit nach dem Training laufen, braucht nur
  die schon gecachten Feature-CSVs + den Checkpoint. Großer Abstand train_acc vs.
  test_acc = Overfitting-Signal, wird auch automatisch angemerkt.

**Evaluation**
- `evaluate.py` — lädt die gecachten Test-Features und `best_classifier.pt`, setzt
  `.eval()` (Dropout aus) und `torch.no_grad()`, gibt Accuracy, F1 (macro/weighted), ROC-AUC
  und PR-AUC (gesamt + pro Klasse) sowie den per-Klasse-Report aus und schreibt
  `data/eval/confusion_matrix.png`, `data/eval/roc_curves.png`,
  `data/eval/metrics.json` (alle Zahlen zum Zitieren) und `data/eval/predictions.csv`
  (eine Zeile pro Clip mit Confidence, correct-Flag und Wahrscheinlichkeit pro Klasse). Mit
  `--features data/val_features.csv` kannst du zwischendurch auf dem Val-Split messen.

**Nicht Teil der Frozen-Backbone-Pipeline**
- `train_example.py` — gehört zum alternativen End-to-End-Fine-Tuning-Ansatz. Nur als
  Shape-Sanity-Check nützlich, für die Pipeline aus deinen Diagrammen nicht nötig.

## Hinweise

- `target_num_frames` (Standard 16) ist unabhängig von der Auflösung — in `config.json` änderbar.
- Mean/std in `video_dataset.py` entsprechen dem pretrained-r3d_18-Rezept. Bei deutlich
  höherer Auflösung lohnt es sich irgendwann, eigene Statistiken zu berechnen.
- `CROP_SIZE` in `extract_features.py` steht auf 112. Wenn du auf 176/224 gehst, muss
  `extract_features.py` neu laufen — `FEATURE_DIM` bleibt aber 512, weil das
  Adaptive-Pooling die Größe wegmittelt.
