// Mini traffic-light model for the demo (ESP32 or Arduino Uno).
// Receives one line per update over USB serial at 115200 baud: four characters, one per approach
// in the order N E S W, each G (green), Y (yellow) or R (red). Example: "GRGR\n".
// Fail-safe: if nothing arrives for 10 s, all approaches flash yellow.
// Research prototype only: never connect to a real junction.

const int PINS[4][3] = {   // {red, yellow, green} per approach; change to match your wiring
  {13, 12, 14},            // N
  {27, 26, 25},            // E
  {33, 32, 23},            // S
  {22, 21, 19},            // W
};
const unsigned long TIMEOUT_MS = 10000;
unsigned long lastMsg = 0;
String line;

void setApproach(int a, char c) {
  digitalWrite(PINS[a][0], c == 'R' ? HIGH : LOW);
  digitalWrite(PINS[a][1], c == 'Y' ? HIGH : LOW);
  digitalWrite(PINS[a][2], c == 'G' ? HIGH : LOW);
}

void setup() {
  Serial.begin(115200);
  for (int a = 0; a < 4; a++)
    for (int k = 0; k < 3; k++) pinMode(PINS[a][k], OUTPUT);
  for (int a = 0; a < 4; a++) setApproach(a, 'R');
  Serial.println("READY");
}

void loop() {
  while (Serial.available()) {
    char ch = Serial.read();
    if (ch == '\n') {
      line.trim();
      if (line.length() == 4) {
        for (int a = 0; a < 4; a++) setApproach(a, line[a]);
        lastMsg = millis();
        Serial.print("OK ");
        Serial.println(line);
      }
      line = "";
    } else {
      line += ch;
    }
  }
  if (millis() - lastMsg > TIMEOUT_MS) {   // lost the controller: flashing yellow
    bool on = (millis() / 500) % 2;
    for (int a = 0; a < 4; a++) setApproach(a, on ? 'Y' : 'X');
  }
}
