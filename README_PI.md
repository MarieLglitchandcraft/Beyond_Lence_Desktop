# Beyond Lence auf dem Raspberry Pi 5

## Pinbelegung

Die Software verwendet BCM-Nummern. Die Zahlen in Klammern sind die physischen
Pins des 40-Pin-Anschlusses.

- Status-LED: GPIO 4 (Pin 7)
- Modus 1: GPIO 17 (Pin 11)
- Modus 2: GPIO 27 (Pin 13)
- Modus 3: GPIO 22 (Pin 15)
- Modus 4: GPIO 23 (Pin 16)
- Fototaster: GPIO 24 (Pin 18)
- Abschalttaster: GPIO 25 (Pin 22) gegen GND (Pin 20)
- Encoder CLK: GPIO 5 (Pin 29)
- Encoder DT: GPIO 6 (Pin 31)
- Encoder-Taster: GPIO 13 (Pin 33)
- KY-021 Signal: GPIO 18 (Pin 12); Plus an 3,3 V, Minus an GND
- Display-Not-Aus OUT: GPIO 19 (Pin 35)
- Passiver Piezo: GPIO 14 (Pin 8) über 100 bis 220 Ohm; Minus an GND
- Detektor-LED 1: GPIO 16 (Pin 36)
- Detektor-LED 2: GPIO 20 (Pin 38)
- Detektor-LED 3: GPIO 21 (Pin 40)
- Detektor-LED 4: GPIO 12 (Pin 32)
- Detektor-LED 5: GPIO 26 (Pin 37)

Die Modusschalter und Taster schalten gegen GND und verwenden die internen
Pull-up-Widerstände. Jede LED benötigt einen eigenen Vorwiderstand, zum Beispiel
220 bis 330 Ohm. Der Encoder darf die GPIO-Eingänge nur mit 3,3 Volt versorgen.

Der Encoder-Taster setzt die Frequenz auf 0 zurück. Die eingestellte Frequenz
wird ausschließlich im Bildschirm-HUD angezeigt. Die fünf LEDs gehören zum
Anomalie-Detektor: Sie blitzen gleichzeitig mit jedem Piezo-Klick. Bei einem
schwachen Signal leuchtet eine LED, bei einem starken Signal leuchten bis zu
fünf LEDs. Ohne erkannte Anomalie bleiben sie dunkel.

## Piezo-Geigerzähler

Die Software steuert einen kleinen passiven Zweipol-Piezo auf GPIO 14
(physischer Pin 8). Die andere Seite kommt an GND, zum Beispiel Pin 6. In die
GPIO-Leitung gehört ein Widerstand von 100 bis 220 Ohm. Bei Frequenz 0 und ohne
aktiven Anomalie-Modus bleibt der Piezo still. Die unregelmäßige Klickrate steigt
mit der Frequenz sowie mit sichtbaren Wesen und Portalen.

Diese Verdrahtung ist nur für einen kleinen passiven Piezo gedacht. Einen
dynamischen 8-Ohm-Lautsprecher oder ein unbekanntes Buzzer-Modul nicht direkt an
einen GPIO anschließen; dafür wird eine passende Verstärker- beziehungsweise
Treiberstufe benötigt.

GPIO 14 ist gleichzeitig der UART-TX-Pin des Raspberry Pi. Während Beyond Lence
läuft, darf deshalb kein serieller Adapter oder anderes UART-Gerät parallel an
Pin 8 angeschlossen sein.

## Klappdisplay und Not-Aus

Der KY-021 bleibt wie folgt angeschlossen:

- S an GPIO 18, physischer Pin 12
- Plus an 3,3 Volt, physischer Pin 1
- Minus an GND

Der Magnet muss bei eingeklapptem Display direkt am Reed-Sensor liegen. Wird
das Display aufgeklappt und der Magnet entfernt, spiegelt die Software das
Kameramotiv und die visuellen Phänomene horizontal wie bei einer Selfie-Kamera.
HUD, Statusmeldungen und Debug-Schrift werden erst danach lesbar und ungespiegelt
gezeichnet. Zum Schluss dreht die Software ausschließlich das fertige
Beyond-Lence-Programmbild um 180 Grad für das mechanisch umgeklappte Panel.
Der HDMI-Systemausgang, der Desktop, der Mauszeiger und eine VNC-Sitzung werden
niemals als Systemausgabe gedreht. Es werden dafür keine
System-Rotationsbefehle ausgeführt.
Der Sensor erkennt nur die beiden Zustände
eingeklappt/aufgeklappt, keinen beliebigen Drehwinkel.

Auch gespeicherte sichtbare Fotos erhalten im Selfie-Modus die korrigierte
Orientierung. Diagnose-Rohbilder bleiben unverändert, damit sie weiterhin das
originale Kamerasignal zeigen.

WayVNC überträgt denselben Wayland-Ausgang wie das HDMI-Display. Deshalb sieht
eine VNC-Verbindung das gedrehte Beyond-Lence-Programm ebenfalls gedreht, obwohl
Desktop, Mauskoordinaten und Systemausrichtung unverändert bleiben. Für zwei
gleichzeitige Orientierungen wäre ein zweiter virtueller Wayland-Ausgang oder
ein eigener ungedrehter Kontrollstream erforderlich.

Der dreipolige Flugzeugschalter wird mit OUT an GPIO 19, physischer Pin 35,
angeschlossen. Plus/VCC darf ausschließlich an 3,3 Volt und Minus/GND an GND.
Die Software wertet ein 3,3-Volt-HIGH-Signal an OUT als aktiven Not-Aus. Niemals
5 oder 12 Volt an OUT oder einen GPIO anschließen.

Ist der Not-Aus aktiv, wird das Kamerafenster sofort vollständig schwarz, der
Piezo wird stumm und der Fototaster gesperrt. Der Raspberry Pi und das Display
bleiben elektrisch eingeschaltet. Für eine echte Trennung der
Display-Stromversorgung wird eine zum konkreten Display passende Treiber- oder
Relaisstufe benötigt; sie darf nicht direkt von einem GPIO gespeist werden.

GPIO 19 verwendet einen internen Pulldown und eine zusätzliche stabile
Umschaltzeit von 250 ms. Dadurch führen sehr kurze Kontaktunterbrechungen nicht
mehr zu einem flackernden schwarzen Display. Ein echter Wackelkontakt kann bei
gestopptem Beyond-Lence-Programm zwölf Sekunden lang geprüft werden:

```sh
cd ~/beyond_lence_neu
./.pi_venv/bin/python pi_setup/test_aircraft_switch.py
```

Während des Tests den Schalter einmal langsam ein- und wieder ausschalten. Zwei
saubere Zustandswechsel sind normal; schnelle Mehrfachwechsel weisen auf einen
Wackelkontakt, eine schlechte Lötstelle oder eine instabile Masseverbindung
hin.

## Foto-Rückmeldung

Nach erfolgreichem Speichern simuliert das Display kurz einen hellen
Kamerablitz. Danach bleibt exakt die gespeicherte Aufnahme insgesamt zwei
Sekunden stehen. In ihrer Mitte erscheint `SPECTRAL IMAGE CAPTURED`. Die
Aufnahmedatei selbst enthält diese Bestätigung nicht zusätzlich; sie speichert
weiterhin das sichtbare Kamerabild mit den Beyond-Lence-Effekten.

## Ein- und Ausschalten

Der untere Taster wird als normal offener, zurückspringender Taster zwischen
GPIO 25 (physischer Pin 22) und GND (physischer Pin 20) angeschlossen. Das
Installationsskript aktiviert dafür den Kernel-Abschalttaster mit 300 ms
Entprellung. Ein Druck löst ein kontrolliertes Herunterfahren aus.

Diese Variante kann den heruntergefahrenen Raspberry Pi 5 nicht wieder starten.
Zum Einschalten wird der kleine eingebaute Power-Taster auf dem Pi verwendet.
Der Akku beziehungsweise das Netzteil bleibt dabei angeschlossen.

## Projekt vom Mac auf den Pi kopieren

Mac und Raspberry Pi müssen im selben Netzwerk sein und SSH muss auf dem Pi
aktiviert sein. Im Terminal auf dem Mac im Beyond-Lence-Projektordner ausführen:

```sh
./pi_setup/copy_to_pi.sh PI-BENUTZERNAME
```

Beispiel, falls der Benutzer auf dem Pi `marie` heißt:

```sh
./pi_setup/copy_to_pi.sh marie
```

Das Skript kopiert Programm und Assets nach `~/beyond_lence_neu` auf dem Pi.
Die Mac-Umgebung, alte Fotos und Protokolle werden nicht übertragen.

## Installation und Autostart

Am sichersten wird die Installation in einem Terminal direkt auf dem bereits
sichtbaren Pi-Desktop ausgeführt. Alternativ kann man sich vom Mac verbinden,
wenn der Desktop auf dem Pi bereits angemeldet und sichtbar ist:

```sh
ssh PI-BENUTZERNAME@raspberrypi.local
```

Auf dem Raspberry Pi dann ausführen:

```sh
cd ~/beyond_lence_neu
chmod +x pi_setup/install_on_pi.sh
./pi_setup/install_on_pi.sh
sudo reboot
```

Danach den Raspberry Pi neu starten. Der Desktop muss automatisch angemeldet
werden, damit das OpenCV-Kamerafenster ohne Benutzereingriff erscheinen kann.
Das Installationsskript installiert Picamera2 und GPIO Zero über Raspberry Pi
OS, richtet MediaPipe in einer eigenen Pi-Umgebung ein, prüft Gesicht, Pose und
Personenmaske mit den mitgelieferten Modellen und legt den Desktop-Autostart an.
Die Installation gilt als fehlgeschlagen, wenn eines der drei MediaPipe-Modelle
nicht gestartet werden kann.

Voraussetzung ist Raspberry Pi OS 64 Bit mit Python 3.9 bis 3.12. Empfohlen ist
Raspberry Pi OS Bookworm 64 Bit mit Python 3.11. Das Skript verwendet auf ARM64
MediaPipe 0.10.18, weil diese Version noch ein fertiges Linux-ARM64-Paket für
diese Python-Versionen bereitstellt. Neuere MediaPipe-Versionen besitzen kein
fertiges Linux-ARM64-Paket mehr.

Die sichtbaren Aufnahmen werden im Ordner `photos` gespeichert. Das laufende
Programm schreibt Fehler und Startmeldungen nach `logs/beyond-lence.log`.

## Displaydiagnose und Fenster-Test

Die Kamera und die Effekte arbeiten intern weiterhin mit 1024 × 600 Pixeln.
Diese Werte sind keine fest eingetragene Displaygröße mehr. Beim Start erkennt
Beyond Lence die tatsächlich nutzbare Bildschirmauflösung über Wayland, X11,
die Desktopumgebung oder als Rückfallebene über DRM/EDID und den Framebuffer.
Das fertige Bild einschließlich HUD wird proportional auf diese Fläche
skaliert. Abweichende Seitenverhältnisse erhalten schwarze Ränder; das Bild
wird weder gestreckt noch abgeschnitten.

`wlr-randr` und `xrandr --current` können dabei ausschließlich lesend zur
Ermittlung der Auflösung verwendet werden. Das Projekt führt keinerlei
Rotations-, Transformations- oder Konfigurationsbefehl für einen Systemausgang
aus.

Auf dem Raspberry Pi erfolgt die sichtbare Ausgabe über Pygame/SDL. OpenCV
bleibt für Kamera, Erkennung und Effekte zuständig, erzeugt dort aber kein
Fenster mehr. Damit kann die Qt-HighGUI-Oberfläche weder eine Werkzeugleiste
einblenden noch die Bildfläche in helle Rechtecke zerlegen. Auf dem Mac bleibt
OpenCV als Anzeige-Rückfallebene verfügbar.

Wenn das System das HDMI-Display im Hochformat meldet, schreibt das Programm
eine deutliche Warnung in das Protokoll. Es ändert die Systemrotation nicht
selbstständig.

Vor dem Vollbildtest sollte Beyond Lence in einem normalen, verschiebbaren
Fenster gestartet werden:

```sh
cd ~/beyond_lence_neu
./.pi_venv/bin/python main.py --windowed
```

Danach folgt der Vollbildtest:

```sh
./.pi_venv/bin/python main.py --fullscreen
```

Ohne Zusatz ist Vollbild weiterhin der Standard. Beim Start protokolliert das
Programm die erkannte Bildschirm- und Kameraauflösung, das erste Kameraframe,
die Rendergröße, die Skalierungsfaktoren, den Fenstermodus sowie die erkannte
Wayland-/X11- und OpenCV-Umgebung. Falls die automatische Displayerkennung
nicht verfügbar ist, kann ausschließlich zur Diagnose eine Größe vorgegeben
werden:

```sh
BEYOND_LENCE_DISPLAY_SIZE=1280x800 ./.pi_venv/bin/python main.py --windowed
```

Wenn kein gültiges Kamerabild ankommt, erscheint statt eines defekten
Vollbildfensters eine lesbare Fehlermeldung. Das Programm beendet sich danach
sauber und bleibt nicht in einer leeren Schleife hängen.

Der durch das Installationsskript erzeugte Autostart liegt auf dem Raspberry Pi
unter:

```text
/home/mariel/.config/autostart/beyond-lence.desktop
```

Diese Datei startet `pi_setup/start_beyond_lence.sh`, das wiederum `main.py`
mit der Pi-Python-Umgebung aufruft. Für die Displaykorrektur wird dieser
Autostart nicht verändert.
