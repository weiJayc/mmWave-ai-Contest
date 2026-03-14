#include <IRremote.hpp>

#define IR_RECEIVE_PIN 2
#define IR_SEND_PIN 3

#define MAX_RAW_LEN 700
#define GESTURE_COUNT 4

// ===== 資料庫 =====
uint16_t rawData[GESTURE_COUNT][MAX_RAW_LEN];
uint8_t rawLen[GESTURE_COUNT];

IRData gestureInfo[GESTURE_COUNT];

// ===== 學習模式 =====
bool learningMode = true;
int learnIndex = 0;

// ===== 防連續發送 =====
unsigned long lastSendTime = 0;
const unsigned long SEND_COOLDOWN = 800;
bool lastCommendIsPull = false;

enum GESTURE {DoubleTap, PUSH, WAVELEFT, WAVERIGHT};

void setup() {

  Serial.begin(9600);

  IrReceiver.begin(IR_RECEIVE_PIN, ENABLE_LED_FEEDBACK);
  IrSender.begin(IR_SEND_PIN, ENABLE_LED_FEEDBACK);

  /*
  Serial.println("=== IR Gesture Controller ===");
  Serial.println("Learning Mode");
  Serial.println("Press remote buttons to record 4 gestures");
  */
  pinMode(7, OUTPUT);
}

void loop() {
  // ===== Learning Mode =====
  if (learningMode) {
    
    // clear buffer
    while (Serial.available()) {
      Serial.readStringUntil('\n');
    }

    if (IrReceiver.decode()) {
      // 忽略 repeat frame
      if (IrReceiver.decodedIRData.flags & IRDATA_FLAGS_IS_REPEAT) {
        //Serial.println("Repeat ignored");
        IrReceiver.resume();
        return;
      }

      // ===== 存 IRData =====
      gestureInfo[learnIndex] = IrReceiver.decodedIRData;

      // ===== 存 RAW =====
      rawLen[learnIndex] = IrReceiver.decodedIRData.rawlen - 1;

      for (int i = 1; i <= rawLen[learnIndex]; i++) {
        rawData[learnIndex][i-1] =
          IrReceiver.irparams.rawbuf[i] * MICROS_PER_TICK;
      }

      //Serial.print("Learned Gesture ");
      for (int i = 0; i <= learnIndex; ++i) {
        digitalWrite(7, HIGH);
        delay(300);
        digitalWrite(7, LOW);
        delay(300);
      }
      
      /*
      Serial.println(learnIndex + 1);

      Serial.print("Protocol: ");
      Serial.println(getProtocolString(gestureInfo[learnIndex].protocol));

      Serial.print("Bits: ");
      Serial.println(gestureInfo[learnIndex].numberOfBits);

      Serial.print("Raw length: ");
      Serial.println(rawLen[learnIndex]);
      */

      learnIndex++;

      IrReceiver.resume();

      if (learnIndex >= GESTURE_COUNT) {

        learningMode = false;

        //Serial.println("Learning Complete!");
        //Serial.println("Input 1~4 to send IR signal");
      }
    }

  }

  // ===== Control Mode =====
  else {

    // clear IrReciiver buffer
    while (IrReceiver.decode()) {
      IrReceiver.resume();
    }

    if (Serial.available()) {

      String s = Serial.readStringUntil('\n');
      s.trim();
      if (s == "Background") { return; }

      if(s == "Pull"){ //重製指令
        
        if (lastCommendIsPull) {
          learningMode = true;
          learnIndex = 0;
        }
        lastCommendIsPull = true;
        //Serial.println("reset button");
        return;
      }

      lastCommendIsPull = false;
      int gesture = -1;
      if(s == "DoubleTap"){
        gesture = DoubleTap;
      }
      else if(s == "Push"){
        gesture = PUSH;
      }
      else if(s == "WaveLeft"){
        gesture = WAVELEFT;
      }
      else if(s == "WaveRight"){
        gesture = WAVERIGHT;
      }

      if (gesture < 0 || gesture >= GESTURE_COUNT) {
        //Serial.println("Invalid input");
        return;
      }

      //int idx = gesture - 1;

      unsigned long now = millis();

      if (now - lastSendTime < SEND_COOLDOWN) {
        //Serial.println("Cooldown active");
        return;
      }

      // ===== 使用 RAW 發送 =====
      //IrSender.sendRaw(rawData[idx], rawLen[idx], 38);

      
      if(gesture != -1) {
        IrSender.sendRaw(rawData[gesture], rawLen[gesture], 38);
        digitalWrite(7, HIGH);
        delay(100);
        digitalWrite(7, LOW);
      }
      
      lastSendTime = now;
    }
  }
}