# Lokale Vision-Pipeline

## Entscheidung: Rust in der App, Python für Modellentwicklung

Für die ausgelieferte Anwendung ist **Rust als Inferenzschicht** die bessere
Zielarchitektur. Python bleibt das Werkzeug für Training, Evaluation, Export und
kleine Modelltests.

| Kriterium | Rust-Inferenz | Python-Sidecar |
| --- | --- | --- |
| Installation | Eine native Anwendung, keine Python-Umgebung | Zusätzliche, plattformspezifische Sidecar-Binaries |
| Startzeit und Speicher | Niedriger, wenn Runtime und Sessions einmal geladen werden | Höher durch Interpreter und ML-Pakete |
| Tauri-Integration | Direkte Commands und kontrollierter Zustand | Prozessmanagement und IPC nötig |
| Modellentwicklung | Weniger komfortabel | PyTorch/TensorFlow und Datentools sind klar stärker |
| Prototyping | Mehr Integrationsarbeit | Schnell für Experimente |

Empfohlener Ablauf:

1. Modelle in Python trainieren oder feinabstimmen.
2. Vor- und Nachverarbeitung mit festen Testvektoren einfrieren.
3. Modelle nach ONNX exportieren und mit ONNX Runtime in Python validieren.
4. Dieselben Modelle und Testvektoren in Rust laden.
5. Python nicht mit der fertigen Desktop-Anwendung ausliefern.

Ein Python-Sidecar ist sinnvoll, wenn ein benötigter Operator nicht sauber nach
ONNX exportiert werden kann oder ein Forschungsmodell sehr häufig gewechselt
wird. Tauri kann solche Binaries bündeln, sie müssen aber für jede Zielarchitektur
separat gebaut und über IPC abgesichert werden.

## Aktueller Datenfluss

```text
Webcam (getUserMedia, nur Video)
  -> Canvas-Sampling mit 320 x 240 und maximal 8 FPS
  -> RGBA nach RGB
  -> Tauri-Command vision_process_frame
  -> LowLevelDetector
       -> 0..n ObjectDetection { Klasse, Konfidenz, Bounding-Box }
  -> TemporalActionRecognizer
       -> ActionPrediction { Klasse, Konfidenz, Warm-up }
  -> TypeScript-Callback
```

Das reguläre Pet-Fenster startet die Kamera **nicht automatisch**.
`WebcamVisionStream.start(...)` soll dort durch eine sichtbare Benutzeraktion
ausgelöst werden. `stop()` beendet alle Media-Tracks. Bilder werden weder
gespeichert noch über das Netzwerk übertragen.

Für die lokale Entwicklung gibt es zusätzlich das Fenster `vision-debug`. Es
wird über die globale Konstante `ENABLE_VISION_DEBUG_VIEW` in `src/config.ts`
ein- oder ausgeschaltet. Bei `true` wird es neben dem Pet positioniert und zeigt
Kamerabild, Bounding-Boxes und die Action-Ausgabe. Bei `false` bleibt es verborgen
und startet keine Kamera. Das Debug-Fenster besitzt zusätzlich sichtbare
Start-/Stop-Schaltflächen. Das automatische Starten bei aktiviertem Debug-Schalter
ist eine bewusst auf den Entwicklungsmodus begrenzte Ausnahme.

## Implementierte Verträge

Frontend: `src/vision.ts`

- `WebcamVisionStream`: Kamera-Lebenszyklus, FPS-Begrenzung und Backpressure
- `getVisionStatus()`: aktive Implementierung und Eingabeparameter
- `processVisionFrame()`: einzelnes RGB-Frame verarbeiten
- `resetVision()`: Modellzustand bei Kamerawechsel oder Neustart löschen

Backend: `src-tauri/src/vision.rs`

- `LowLevelDetector`: austauschbare Erkennungsstufe
- `TemporalActionRecognizer`: austauschbare zeitliche Klassifikationsstufe
- `VisionBackend`: validiert Dimensionen und Zeitstempel und hält Streamzustand
- serialisierbare Bounding-Box-, Detection-, Action- und Status-Typen

Der vorhandene `movinet_*`-Endpunkt bleibt vorerst als kompatibler,
eigenständiger Prototyp bestehen. Neue Integrationen sollen die allgemeinere
`vision_*`-Schnittstelle verwenden.

Minimaler Aufruf aus einer späteren Einstellungs- oder Kameraansicht:

```ts
import { WebcamVisionStream } from "./vision";

const vision = new WebcamVisionStream();

cameraButton.addEventListener("click", async () => {
  await vision.start({
    onPrediction: ({ detections, action }) => {
      console.debug({ detections, action });
    },
    onError: (error) => console.error(error),
  });
});

window.addEventListener("beforeunload", () => vision.stop());
```

Der Start gehört absichtlich nicht in den allgemeinen App-Bootstrap: Eine
Kamera-Freigabe beim Start der Desktop-Pet-Anwendung wäre überraschend und gäbe
dem Nutzer keine verständliche Ein-/Aus-Steuerung.

## Was der Prototyp leistet – und was nicht

Der aktuelle Detector markiert zusammenhängend umschlossene Bildänderungen als
`foreground-region`. Die Action-Stufe klassifiziert die Bewegung des
Box-Schwerpunkts als stillstehend, links, rechts, oben oder unten. Damit lassen
sich Kamera, IPC, Zustand, Reset und zeitliches Warm-up ohne große Modelldateien
testen.

Das ist **keine semantische Objekterkennung** und **kein trainiertes Action-
Recognition-Modell**. Die Bezeichner `runtime: prototype` und der Statustext
machen das auch zur Laufzeit sichtbar.

## Zielarchitektur mit ONNX

### Stufe 1: Objekterkennung

Die erste ML-Stufe sollte ein kleines Object-Detection-Modell sein, nicht nur ein
Image-Classifier. Nur ein Detector liefert die Position des Vordergrundobjekts,
die anschließend verfolgt und ausgeschnitten werden kann.

Benötigte Schritte in einer Rust-Implementierung:

1. ONNX-Session einmal beim App-Start laden.
2. RGB nach dem Modellvertrag skalieren, letterboxen und normalisieren.
3. Inferenz ausführen.
4. Scores filtern und Non-Maximum Suppression anwenden, falls nicht im Graphen.
5. Boxen in Koordinaten des Kamera-Frames zurückrechnen.
6. Optional eine stabile `trackId` ergänzen, wenn mehrere Objekte auftreten.

Die konkrete Modellwahl hängt von den gesuchten Klassen ab. Für Personenaktionen
sollte zuerst `person` erkannt und verfolgt werden. Für eigene Haustiere oder
spezielle Gegenstände ist meist Fine-Tuning auf den tatsächlich benötigten
Klassen erforderlich. Vor der Übernahme eines Modells müssen Gewichte- und
Code-Lizenzen getrennt geprüft werden.

### Stufe 2: Action Recognition

Die Action-Stufe benötigt zeitlichen Kontext. Zwei sinnvolle Varianten sind:

- Ein Streaming-Modell wie MoViNet, das seinen Zustand Frame für Frame
  fortschreibt. Das passt gut zu geringer Latenz.
- Ein Clip-Modell, das beispielsweise 8, 16 oder 32 normalisierte Frames aus
  einem Ringpuffer verarbeitet. Es ist einfacher zu exportieren, benötigt aber
  mehr Speicher und verursacht Inferenzspitzen.

Für MoViNet darf der Modellzustand nicht zwischen Frames verloren gehen. Er muss
bei Kamerawechsel, Auflösungswechsel, längerer Unterbrechung und explizitem
`vision_reset` geleert werden. Ein Action-Modell sollte den Person-Crop oder eine
Kombination aus Crop und Detection-Metadaten erhalten; reine Klassenlabels der
ersten Stufe enthalten zu wenig Bewegungsinformation.

### Runtime und Beschleunigung

ONNX Runtime bietet native Inferenz und verschiedene Execution Provider. Der
Rust-Zugriff ist laut ONNX-Runtime-Dokumentation eine Community-API. Deshalb soll
die konkrete Rust-Crate vor dem Einbau auf Zielplattformen fest gepinnt und mit
einem minimalen Lade-/Inferenztest abgesichert werden.

Empfohlener Start:

- zuerst CPU für reproduzierbare Korrektheit;
- anschließend Windows-Hardwarebeschleunigung separat benchmarken;
- auf macOS einen passenden CoreML-fähigen Build testen;
- immer einen CPU-Fallback und verständliche Statusfehler behalten.

## Modellvertrag vor dem Einbau festlegen

Für jedes Modell gehören die folgenden Angaben versioniert in eine kleine
Manifestdatei oder direkt in eine Rust-Konfiguration:

- Modellname, Version, Herkunft und Lizenz
- SHA-256 der Gewichtedatei
- Input- und Output-Tensornamen
- Layout (`NCHW` oder `NHWC`) und Datentyp
- feste oder dynamische Dimensionen
- Farbkanalreihenfolge, Skalierung, Mittelwert und Standardabweichung
- Resize-/Crop-/Letterbox-Regeln
- Klassenliste und Score-Schwelle
- zeitliche Fenstergröße und erwartete FPS
- bekannte Einschränkungen und Trainingsdomäne

Ohne diesen Vertrag können Python und Rust trotz identischer Gewichte
unterschiedliche Ergebnisse produzieren.

## Performance-Messpunkte

Für eine belastbare lokale Pipeline sollten mindestens erfasst werden:

- Capture- und Konvertierungszeit
- IPC-Zeit und übertragene Bytes
- Detector-Latenz
- Action-Latenz
- verworfene Frames durch Backpressure
- Prozessspeicher nach Warm-up

Die aktuelle JSON-/Array-Übertragung ist für einen Prototyp bei 320 × 240 und
8 FPS gedacht. Für Produktion sollte entweder die Kamera nativ in Rust erfasst
oder ein binärer Tauri-IPC-Pfad genutzt werden. Wichtig ist, immer nur ein Frame
gleichzeitig zu verarbeiten; `WebcamVisionStream` verwirft bereits neue Ticks,
solange eine Inferenz läuft.

## Datenschutz und Produktverhalten

- Kamera nur nach expliziter Benutzeraktion starten.
- Aufnahmezustand sichtbar anzeigen.
- Keine Audiofreigabe anfordern.
- Frames standardmäßig nicht persistieren.
- Keine Telemetrie mit Bilddaten.
- Kamera beim Schließen, Pausieren und bei Fehlern sicher stoppen.
- Auf macOS ist `NSCameraUsageDescription` in `src-tauri/Info.plist` hinterlegt.

## Referenzen

- [Tauri: externe Binaries/Sidecars](https://v2.tauri.app/develop/sidecar/)
- [Tauri: macOS-Anwendungsbundle und Kamera-Hinweis](https://v2.tauri.app/distribute/macos-application-bundle/)
- [ONNX Runtime: Einstieg und verfügbare APIs](https://onnxruntime.ai/docs/get-started/)
- [ONNX Runtime: Web-App – native Runtime für beste Performance](https://onnxruntime.ai/docs/tutorials/web/build-web-app.html)
- [TensorFlow: MoViNet für Streaming Action Recognition](https://www.tensorflow.org/hub/tutorials/movinet)
