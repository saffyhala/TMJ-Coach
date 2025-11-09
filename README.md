# TMJ Coach  
**Tagline:** *Your personal jaw-wellness AI companion*

---

## 🎯 Overview  

Jaw tension, clenching, and grinding are common but often ignored — especially among students, remote workers, and people under stress.  
These habits can lead to **headaches**, **dizziness**, **neck pain**, **ear tension**, and **temporomandibular joint (TMJ) disorders**.

According to clinical research, **5–12% of the population** experience TMJ-related symptoms, making it one of the most common orofacial pain conditions.  
([American Academy of Family Physicians, 2023](https://www.aafp.org/pubs/afp/issues/2023/0100/temporomandibular-disorders.html))

**TMJ Coach** is a lightweight AI agent that:
- Watches your jaw via webcam  
- Detects clenching, lateral shifting, or grinding in real-time  
- Gently nudges you to relax  
- Learns from your feedback to adapt over time

---

## 💡 Problem & Motivation  

- Many users clench or grind unconsciously during stressful or focused tasks.  
- This can lead to:
  - Jaw pain and stiffness  
  - Headaches or dizziness  
  - Ear issues or tinnitus  
  - Sleep disruption  
- Existing solutions (mouth guards, physiotherapy) are **reactive**, not **preventive**.

**TMJ Coach** addresses this by being proactive and personalized as it learns your relaxed state and provides live feedback before symptoms worsen.

---

## 🚀 What TMJ Coach Does  

1. **Calibration** – Learns your relaxed baseline for lip gap and mouth width.  
2. **Detection** – Uses your webcam and MediaPipe FaceMesh to track:
   - Lip gap (vertical jaw position)  
   - Mouth width (horizontal tension)  
   - Jaw offset (left/right deviation)  
   - Lateral motion (side-to-side activity)  
3. **Classification** – Uses heuristics + an online `SGDClassifier` trained from your feedback.  
4. **Nudging** – Gives you a subtle red overlay and a spoken message when clenching is detected.  
5. **Learning** – Asks, “Were you clenching?” — your yes/no feedback improves its model.  
6. **Adaptivity** – Reinforcement learning (RL) adjusts how often and how quickly it reminds you based on your responses.  

---

## 🧱 Architecture  

```text
Webcam Input
     │
     ▼
MediaPipe FaceMesh
     │
     ▼
Feature Extraction
(lip gap, mouth width, jaw offset, lateral activity)
     │
     ├──► Heuristic Clench Detection
     │
     ▼
Online Classifier (SGD)
     │
     ▼
RL Agent (Adaptive Sensitivity)
     │
     ▼
UI + Voice Feedback
(PyQt6 app + pyttsx3 TTS)
```

---

## 🧠 How the Agent Works  

### 🧩 Feature Extraction  

Per frame, the model computes the following features:  

```python
# Core per-frame feature extraction
lip_gap_ratio = lip_gap / baseline_lip_gap
mouth_width_ratio = mouth_width / baseline_mouth_width
jaw_offset = (chin_x - nose_x) / mouth_width   # normalized left/right shift
lateral_score = compute_lateral_activity(jaw_history, window_seconds=2.0)
```

These features describe **vertical closure** and **horizontal motion** of the jaw, forming a compact 4D feature vector:

```python
features = [lip_gap_ratio, mouth_width_ratio, jaw_offset, lateral_score]
```

---

### ⚙️ Detection Logic  

Before the ML model has enough data, the app uses simple heuristics to detect clenching or grinding:

```python
# Heuristic rules
vertical_clench = (lip_gap_ratio < 0.7) and (mouth_width_ratio > 1.05)
lateral_clench  = (lip_gap_ratio < 0.95) and (abs(jaw_offset) > 0.2)
grinding        = (lateral_score > 0.4)

heuristic_clench = vertical_clench or lateral_clench or grinding
```

Once you start providing YES/NO feedback, the live classifier replaces these heuristics:

```python
# Online model prediction
is_clenched, p_clench = model_predict_is_clenched(features, fallback_is_clenched=heuristic_clench)
```

---

### 🔁 Feedback & Continual Learning  

When clenching is detected for longer than a threshold (e.g., 5 seconds), the system issues a **nudge**:

```python
if is_clenched and clenched_duration > threshold:
    speak("Try relaxing your jaw.")
    show_red_overlay()
```

Then, TMJ Coach asks for feedback:

```python
# Prompt
print("Were you clenching?")
# User clicks:
#   ✓ YES → label as clenching (1)
#   ✗ NO  → label as relaxed (0)

add_training_example(features, label)
```

The labeled data updates the classifier immediately through continual learning:

```python
clf.partial_fit(X, y)
```

---

### 🧭 Reinforcement Learning Agent  

To avoid over-notifying, the app uses a lightweight **Q-learning** agent that dynamically adjusts its sensitivity and cooldown:

```python
# Three action modes
param_sets = [
    {"thresh": 3.0, "cooldown": 60.0},   # aggressive
    {"thresh": 5.0, "cooldown": 90.0},   # neutral
    {"thresh": 8.0, "cooldown": 150.0},  # gentle
]
```

It learns from your behavior and feedback:

```python
# User feedback buttons
👍 Helpful → reward = +1.0
⚠ Too Much → reward = -1.0
🚫 Can’t    → reward = -1.5
```

Q-learning update rule:

```python
td_target = reward + gamma * max(Q[next_state])
Q[state, action] += alpha * (td_target - Q[state, action])
```

Over time, the agent converges toward the best balance between helpful and unobtrusive reminders adapting to the user's unique clenching habits.
