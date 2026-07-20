# Lokale Modellartefakte

Dieser Ordner ist für lokale Detector- und Action-Recognition-Gewichte gedacht.
Große Gewichte werden nicht in Git eingecheckt. Herkunft, Lizenz, SHA-256 und der
vollständige Ein-/Ausgabevertrag jedes tatsächlich verwendeten Modells müssen in
`docs/vision-architecture.md` oder einer späteren Manifestdatei dokumentiert
werden.

Vorgeschlagene lokale Namen:

```text
models/
  detector.onnx
  action-recognizer.onnx
  labels-detector.txt
  labels-actions.txt
```

Erst wenn der Rust-Loader implementiert ist, sollen diese Dateien über
`tauri.conf.json > bundle > resources` in Installer aufgenommen werden. So kann
ein fehlendes oder inkompatibles Modell während der Entwicklung nicht
versehentlich als funktionsfähiger Produktionspfad erscheinen.

