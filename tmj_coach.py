import cv2
import mediapipe as mp
import numpy as np
import time
import pyttsx3
import pyautogui
from collections import deque
from sklearn.linear_model import SGDClassifier
import threading
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                              QHBoxLayout, QLabel, QPushButton, QProgressBar, QFrame)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject, QSize
from PyQt6.QtGui import QImage, QPixmap, QFont, QPalette, QColor
import sys

mp_face = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# ---------- TTS ----------
tts_engine = pyttsx3.init()
tts_engine.setProperty("rate", 170)

def speak(msg: str):
    try:
        tts_engine.say(msg)
        tts_engine.runAndWait()
    except Exception as e:
        print(f"[TTS] Error: {e}")

# ---------- Helpers ----------
def dist(p1, p2):
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2) ** 0.5

def get_points(face_lm, w, h):
    lm = face_lm.landmark

    # central lips
    upper_idx, lower_idx = 13, 14
    # mouth corners
    left_corner_idx, right_corner_idx = 61, 291
    # nose tip & chin
    nose_idx, chin_idx = 1, 152

    upper = (int(lm[upper_idx].x * w), int(lm[upper_idx].y * h))
    lower = (int(lm[lower_idx].x * w), int(lm[lower_idx].y * h))
    left_corner = (int(lm[left_corner_idx].x * w), int(lm[left_corner_idx].y * h))
    right_corner = (int(lm[right_corner_idx].x * w), int(lm[right_corner_idx].y * h))
    nose = (int(lm[nose_idx].x * w), int(lm[nose_idx].y * h))
    chin = (int(lm[chin_idx].x * w), int(lm[chin_idx].y * h))

    lip_gap = dist(upper, lower)
    mouth_width = dist(left_corner, right_corner)

    return upper, lower, left_corner, right_corner, nose, chin, lip_gap, mouth_width

def compute_lateral_activity(jaw_history, window_seconds=2.0):
    if not jaw_history:
        return 0.0

    now = time.time()
    offsets = []
    for t, offset, gap_ratio in jaw_history:
        if now - t <= window_seconds and gap_ratio < 1.1:  # mouth fairly closed
            offsets.append(offset)

    if len(offsets) < 5:
        return 0.0

    amp = max(offsets) - min(offsets)
    sign_flips = 0
    prev_sign = np.sign(offsets[0])
    for o in offsets[1:]:
        s = np.sign(o)
        if s != 0 and prev_sign != 0 and s != prev_sign:
            sign_flips += 1
        if s != 0:
            prev_sign = s

    lateral_score = amp * sign_flips
    return lateral_score

# ---------- State / Agent ----------
agent_state = {
    "calibrating": True,
    "calib_start_time": None,
    "calib_duration": 5.0,

    "baseline_lip_gap": None,
    "baseline_mouth_width": None,

    "clench_seconds_threshold": 5.0,
    "nudge_cooldown": 90.0,
    "last_nudge_time": 0.0,
    "nudge_active_until": 0.0,

    "helpful_count": 0,
    "too_frequent_count": 0,
    "cant_do_count": 0,

    "clench_start_time": None,

    "session_clench_count": 0,
    
    "waiting_for_feedback": False,
    "feedback_prompt_until": 0.0,
    "last_detection_features": None,
}

# ---------- ML model ----------
model_state = {
    "clf": None,
    "X": [],
    "y": [],
    "min_samples_per_class": 10,
}

# ---------- Simple RL Agent (Q-learning over 3 modes) ----------
class SimpleRLAgent:
    def __init__(self):
        self.param_sets = [
            {"thresh": 3.0, "cooldown": 60.0},   # aggressive
            {"thresh": 5.0, "cooldown": 90.0},   # neutral
            {"thresh": 8.0, "cooldown": 150.0},  # gentle
        ]
        self.num_states = 3  # low clench, medium, high
        self.num_actions = len(self.param_sets)
        self.Q = np.zeros((self.num_states, self.num_actions), dtype=np.float32)
        self.epsilon = 0.2
        self.alpha = 0.3
        self.gamma = 0.9

        self.last_state = 1 
        self.last_action = 1

        self.recent_clenches = deque(maxlen=30)
        self.recent_negative_feedback = deque(maxlen=30)

    def update_state_stats(self, clench_event=False, negative_feedback=False):
        self.recent_clenches.append(1 if clench_event else 0)
        self.recent_negative_feedback.append(1 if negative_feedback else 0)

    def get_state(self):
        if len(self.recent_clenches) == 0:
            rate = 0.0
        else:
            rate = sum(self.recent_clenches) / len(self.recent_clenches)

        if rate < 0.2:
            return 0 
        elif rate < 0.5:
            return 1 
        else:
            return 2 

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            action = np.random.randint(self.num_actions)
        else:
            action = int(np.argmax(self.Q[state]))
        self.last_state = state
        self.last_action = action
        return action

    def apply_action(self, action_index):
        params = self.param_sets[action_index]
        agent_state["clench_seconds_threshold"] = params["thresh"]
        agent_state["nudge_cooldown"] = params["cooldown"]
        print(f"[RL] Mode {action_index} -> thresh={params['thresh']}s, cooldown={params['cooldown']}s")

    def update(self, reward):
        s = self.last_state
        a = self.last_action
        s_next = self.get_state()
        best_next = np.max(self.Q[s_next])
        td_target = reward + self.gamma * best_next
        td_error = td_target - self.Q[s, a]
        self.Q[s, a] += self.alpha * td_error
        print(f"[RL] Update: state={s}, action={a}, reward={reward:.2f}, Q={self.Q[s, a]:.2f}")

rl_agent = SimpleRLAgent()

# ---------- Feedback & training ----------
def handle_feedback_ui(feedback_type):
    if feedback_type == "yes_clenching" and agent_state["last_detection_features"] is not None:
        add_training_example(agent_state["last_detection_features"], label=1)
        agent_state["waiting_for_feedback"] = False
        print("[FEEDBACK] Labeled as CLENCHING")
        return
    elif feedback_type == "no_not_clenching" and agent_state["last_detection_features"] is not None:
        add_training_example(agent_state["last_detection_features"], label=0)
        agent_state["waiting_for_feedback"] = False
        print("[FEEDBACK] Labeled as RELAXED")
        return
    
    if feedback_type == "helpful":
        agent_state["helpful_count"] += 1
        rl_agent.update_state_stats(clench_event=False, negative_feedback=False)
        rl_agent.update(reward=+1.0)
    elif feedback_type == "too_much":
        agent_state["too_frequent_count"] += 1
        rl_agent.update_state_stats(clench_event=False, negative_feedback=True)
        rl_agent.update(reward=-1.0)
    elif feedback_type == "cant":
        agent_state["cant_do_count"] += 1
        rl_agent.update_state_stats(clench_event=False, negative_feedback=True)
        rl_agent.update(reward=-1.5)

    state = rl_agent.get_state()
    action = rl_agent.select_action(state)
    rl_agent.apply_action(action)

def add_training_example(features, label):
    model_state["X"].append(features)
    model_state["y"].append(label)
    X = np.array(model_state["X"], dtype=np.float32)
    y = np.array(model_state["y"], dtype=np.int32)

    unique, counts = np.unique(y, return_counts=True)
    counts_dict = dict(zip(unique, counts))
    if (0 not in counts_dict) or (1 not in counts_dict):
        return

    if (counts_dict[0] < model_state["min_samples_per_class"] or
        counts_dict[1] < model_state["min_samples_per_class"]):
        return

    if model_state["clf"] is None:
        print("[MODEL] Training initial classifier...")
        clf = SGDClassifier(loss="log_loss", max_iter=1000, tol=1e-3)
        clf.fit(X, y)
        model_state["clf"] = clf
    else:
        print("[MODEL] Updating classifier with new samples...")
        model_state["clf"].partial_fit(X, y)

    print(f"[MODEL] Trained with {len(y)} samples. "
          f"Class counts: relaxed={counts_dict.get(0,0)}, clench={counts_dict.get(1,0)}")

def model_predict_is_clenched(features, fallback_is_clenched):
    clf = model_state["clf"]
    if clf is None:
        return fallback_is_clenched, None
    probs = clf.predict_proba(features.reshape(1, -1))[0]
    p_clench = probs[1]
    return p_clench > 0.6, p_clench

def maybe_nudge(is_clenched, clenched_duration, vertical=False, lateral=False, grinding=False):
    now = time.time()
    if not is_clenched:
        return

    if clenched_duration < agent_state["clench_seconds_threshold"]:
        return

    if now - agent_state["last_nudge_time"] < agent_state["nudge_cooldown"]:
        return

    agent_state["last_nudge_time"] = now
    agent_state["nudge_active_until"] = now + 3.0
    agent_state["session_clench_count"] += 1
    rl_agent.update_state_stats(clench_event=True, negative_feedback=False)
    
    agent_state["waiting_for_feedback"] = True
    agent_state["feedback_prompt_until"] = now + 10.0

    print("[AGENT] Nudge: possible jaw loading/clenching.")
    if grinding:
        speak("I see a lot of side to side jaw movement. Try resting your jaw and stopping the grinding.")
    elif lateral:
        speak("Your jaw looks shifted to one side. Try bringing it back to centre and relaxing.")
    else:
        speak("You might be clenching your jaw. Let your teeth separate and soften your cheeks.")

class UpdateSignals(QObject):
    frame_update = pyqtSignal(np.ndarray)
    status_update = pyqtSignal(str, str)
    progress_update = pyqtSignal(float)
    stats_update = pyqtSignal()
    feedback_prompt = pyqtSignal(bool)

class DetectionThread(threading.Thread):
    def __init__(self, signals):
        super().__init__(daemon=True)
        self.signals = signals
        self.running = True
        self.calib_lip_gaps = []
        self.calib_widths = []

    def run(self):
        cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            print("[ERROR] Could not open webcam.")
            return

        jaw_history = deque(maxlen=120)

        with mp_face.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as face_mesh:

            agent_state["calibrating"] = False
            last_frame_time = time.time()

            while self.running:
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.flip(frame, 1)
                h, w, _ = frame.shape
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = face_mesh.process(rgb)
                now = time.time()

                display_frame = rgb.copy()
                status_text = ""
                sub_text = ""

                is_clenched = False
                clenched_duration = 0.0
                vertical_clench = False
                lateral_clench = False
                grinding = False

                if result.multi_face_landmarks:
                    face_lm = result.multi_face_landmarks[0]
                    (upper, lower, left_corner, right_corner,
                     nose, chin, lip_gap, mouth_width) = get_points(face_lm, w, h)

                    # Draw face mesh for visualization
                    # Draw face mesh on a separate overlay for opacity control
                    mesh_overlay = display_frame.copy()
                    mp_drawing.draw_landmarks(
                        mesh_overlay,
                        face_lm,
                        mp_face.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing.DrawingSpec(
                            color=(255, 255, 255),
                            thickness=1,
                            circle_radius=1
                        )
                    )
                    # Blend the mesh overlay with opacity (0.0 = invisible, 1.0 = fully visible)
                    opacity = 0.2  # Adjust this value (0.0 to 1.0)
                    display_frame = cv2.addWeighted(mesh_overlay, opacity, display_frame, 1 - opacity, 0)

                    if agent_state["calibrating"]:
                        elapsed = now - agent_state["calib_start_time"]
                        progress = min(elapsed / agent_state['calib_duration'], 1.0)
                        self.signals.status_update.emit("Calibrating...", f"{elapsed:.1f}/{agent_state['calib_duration']}s")
                        self.signals.progress_update.emit(progress)

                        self.calib_lip_gaps.append(lip_gap)
                        self.calib_widths.append(mouth_width)
                        
                        if elapsed >= agent_state["calib_duration"]:
                            if self.calib_lip_gaps and self.calib_widths:
                                agent_state["baseline_lip_gap"] = float(np.mean(self.calib_lip_gaps))
                                agent_state["baseline_mouth_width"] = float(np.mean(self.calib_widths))
                                print(f"[CALIB] Baseline lip gap={agent_state['baseline_lip_gap']:.2f}, "
                                      f"mouth width={agent_state['baseline_mouth_width']:.2f}")
                            agent_state["calibrating"] = False
                            self.signals.status_update.emit("Monitoring", "Jaw relaxed")
                            self.calib_lip_gaps.clear()
                            self.calib_widths.clear()
                    else:
                        base_gap = agent_state["baseline_lip_gap"] or lip_gap
                        base_width = agent_state["baseline_mouth_width"] or mouth_width

                        gap_ratio = lip_gap / (base_gap + 1e-6)
                        width_ratio = mouth_width / (base_width + 1e-6)

                        jaw_offset_px = chin[0] - nose[0]
                        jaw_offset_norm = jaw_offset_px / (mouth_width + 1e-6)

                        jaw_history.append((now, jaw_offset_norm, gap_ratio))
                        lateral_score = compute_lateral_activity(jaw_history, window_seconds=2.0)

                        vertical_clench = (gap_ratio < 0.7) and (width_ratio > 1.05)
                        lateral_threshold = 0.20
                        is_lateral_shift = abs(jaw_offset_norm) > lateral_threshold
                        lateral_clench = (gap_ratio < 0.95) and is_lateral_shift
                        grinding = lateral_score > 0.4

                        heuristic_clench = vertical_clench or lateral_clench or grinding

                        current_features = np.array(
                            [gap_ratio, width_ratio, jaw_offset_norm, lateral_score],
                            dtype=np.float32
                        )
                        
                        agent_state["last_detection_features"] = current_features

                        is_clenched, p_clench = model_predict_is_clenched(
                            current_features,
                            fallback_is_clenched=heuristic_clench
                        )

                        if is_clenched:
                            if agent_state["clench_start_time"] is None:
                                agent_state["clench_start_time"] = now
                            clenched_duration = now - agent_state["clench_start_time"]
                        else:
                            agent_state["clench_start_time"] = None
                            clenched_duration = 0.0

                        maybe_nudge(
                            is_clenched,
                            clenched_duration,
                            vertical=vertical_clench,
                            lateral=lateral_clench,
                            grinding=grinding,
                        )

                        status_text = "CLENCHING" if is_clenched else "relaxed"
                        sub_text = f"lat_score={lateral_score:.2f}"
                        if p_clench is not None:
                            sub_text += f"  p={p_clench:.2f}"

                        self.signals.stats_update.emit()
                        
                        if agent_state["waiting_for_feedback"] and now < agent_state["feedback_prompt_until"]:
                            self.signals.feedback_prompt.emit(True)
                        else:
                            if agent_state["waiting_for_feedback"] and now >= agent_state["feedback_prompt_until"]:
                                agent_state["waiting_for_feedback"] = False
                            self.signals.feedback_prompt.emit(False)

                else:
                    status_text = "no face"
                    sub_text = ""

                if now < agent_state["nudge_active_until"]:
                    overlay = display_frame.copy()
                    overlay[:] = (0, 0, 255)
                    display_frame = cv2.addWeighted(overlay, 0.35, display_frame, 0.65, 0)

                self.signals.status_update.emit(status_text, sub_text)

                if now - last_frame_time > 0.03:
                    last_frame_time = now
                    self.signals.frame_update.emit(display_frame)

        cap.release()

# ---------- PyQt6 UI ----------
class TMJCoachWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("TMJ Coach")
        self.setWindowFlags(Qt.WindowType.WindowStaysOnTopHint)
        
        # Colors
        self.bg_dark = QColor("#0d0d0d")
        self.bg_card = QColor("#1a1a1a")
        self.accent = QColor("#1e90ff")
        self.warning = QColor("#ff3b3b")
        self.success = QColor("#0bda51")
        self.text_light = QColor("#ffffff")
        self.text_dim = QColor("#b0b0b0")
        
        # Set dark palette
        palette = QPalette()
        palette.setColor(QPalette.ColorRole.Window, self.bg_dark)
        palette.setColor(QPalette.ColorRole.WindowText, self.text_light)
        palette.setColor(QPalette.ColorRole.Base, self.bg_card)
        palette.setColor(QPalette.ColorRole.AlternateBase, self.bg_dark)
        palette.setColor(QPalette.ColorRole.Text, self.text_light)
        palette.setColor(QPalette.ColorRole.Button, self.bg_card)
        palette.setColor(QPalette.ColorRole.ButtonText, self.text_light)
        self.setPalette(palette)
        
        # Position on right
        screen = QApplication.primaryScreen().geometry()
        window_w, window_h = 320, 520
        x = screen.width() - window_w - 20
        y = int(screen.height() * 0.1)
        self.setGeometry(x, y, window_w, window_h)
        self.setFixedSize(window_w, window_h)
        
        # Main widget
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Video preview
        self.video_label = QLabel()
        self.video_label.setFixedSize(320, 180)
        self.video_label.setStyleSheet(f"background-color: {self.bg_card.name()};")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.video_label)
        
        # Status section
        status_container = QWidget()
        status_container.setStyleSheet(f"background-color: {self.bg_dark.name()};")
        status_layout = QVBoxLayout(status_container)
        status_layout.setContentsMargins(15, 12, 15, 8)
        
        self.status_label = QLabel("TMJ COACH")
        self.status_label.setFont(QFont("Helvetica", 16, QFont.Weight.Bold))
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet(f"color: {self.text_light.name()};")
        status_layout.addWidget(self.status_label)
        
        self.substatus_label = QLabel("Press Start to begin")
        self.substatus_label.setFont(QFont("Helvetica", 11))
        self.substatus_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.substatus_label.setStyleSheet(f"color: {self.text_dim.name()};")
        status_layout.addWidget(self.substatus_label)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximum(100)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(4)
        self.progress_bar.setStyleSheet(f"""
            QProgressBar {{
                background-color: {self.bg_dark.name()};
                border: none;
            }}
            QProgressBar::chunk {{
                background-color: {self.accent.name()};
            }}
        """)
        status_layout.addWidget(self.progress_bar)
        status_layout.addSpacing(8)
        
        layout.addWidget(status_container)
        
        # Stats
        stats_container = QWidget()
        stats_container.setStyleSheet(f"background-color: {self.bg_card.name()}; padding: 10px;")
        stats_layout = QVBoxLayout(stats_container)
        stats_layout.setContentsMargins(15, 10, 15, 10)
        
        self.stats_label = QLabel("⚡ Alerts: 0  •  📊 Training: 0")
        self.stats_label.setFont(QFont("Helvetica", 10))
        self.stats_label.setStyleSheet(f"color: {self.text_dim.name()};")
        stats_layout.addWidget(self.stats_label)
        
        layout.addWidget(stats_container)
        layout.addSpacing(12)
        
        # Calibration button
        self.calib_button = QPushButton("⚙  START CALIBRATION")
        self.calib_button.setFont(QFont("Helvetica", 12, QFont.Weight.Bold))
        self.calib_button.setFixedHeight(50)
        self.calib_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.calib_button.setStyleSheet(f"""
            QPushButton {{
                background-color: {self.accent.name()};
                color: white;
                border: none;
                border-radius: 4px;
                padding: 10px;
            }}
            QPushButton:hover {{
                background-color: #4169e1;
            }}
            QPushButton:pressed {{
                background-color: #1c7ed6;
            }}
            QPushButton:disabled {{
                background-color: #555555;
            }}
        """)
        self.calib_button.clicked.connect(self.start_calibration)
        
        calib_container = QWidget()
        calib_container.setStyleSheet("background-color: transparent;")
        calib_layout = QVBoxLayout(calib_container)
        calib_layout.setContentsMargins(15, 0, 15, 15)
        calib_layout.addWidget(self.calib_button)
        layout.addWidget(calib_container)
        
        # Feedback prompt (hidden initially)
        self.feedback_frame = QWidget()
        self.feedback_frame.setStyleSheet(f"""
            background-color: #2d1f1f;
            border: 1px solid {self.warning.name()};
            border-radius: 6px;
        """)
        feedback_layout = QVBoxLayout(self.feedback_frame)
        feedback_layout.setContentsMargins(10, 10, 10, 10)
        
        feedback_title = QLabel("❓ WERE YOU CLENCHING?")
        feedback_title.setFont(QFont("Helvetica", 11, QFont.Weight.Bold))
        feedback_title.setStyleSheet(f"color: {self.warning.name()}; border: none;")
        feedback_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        feedback_layout.addWidget(feedback_title)
        
        feedback_subtitle = QLabel("Help train the AI model")
        feedback_subtitle.setFont(QFont("Helvetica", 9))
        feedback_subtitle.setStyleSheet("color: #cccccc; border: none;")
        feedback_subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        feedback_layout.addWidget(feedback_subtitle)
        
        feedback_btn_layout = QHBoxLayout()
        
        self.btn_yes = QPushButton("✓ YES")
        self.btn_yes.setFont(QFont("Helvetica", 11, QFont.Weight.Bold))
        self.btn_yes.setFixedHeight(45)
        self.btn_yes.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_yes.setStyleSheet(f"""
            QPushButton {{
                background-color: {self.warning.name()};
                color: white;
                border: none;
                border-radius: 4px;
            }}
            QPushButton:hover {{
                background-color: #ff5555;
            }}
        """)
        self.btn_yes.clicked.connect(lambda: handle_feedback_ui("yes_clenching"))
        
        self.btn_no = QPushButton("✗ NO")
        self.btn_no.setFont(QFont("Helvetica", 11, QFont.Weight.Bold))
        self.btn_no.setFixedHeight(45)
        self.btn_no.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_no.setStyleSheet("""
            QPushButton {
                background-color: #333333;
                color: white;
                border: none;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #444444;
            }
        """)
        self.btn_no.clicked.connect(lambda: handle_feedback_ui("no_not_clenching"))
        
        feedback_btn_layout.addWidget(self.btn_yes)
        feedback_btn_layout.addWidget(self.btn_no)
        feedback_layout.addLayout(feedback_btn_layout)
        
        feedback_container = QWidget()
        feedback_container_layout = QVBoxLayout(feedback_container)
        feedback_container_layout.setContentsMargins(15, 0, 15, 12)
        feedback_container_layout.addWidget(self.feedback_frame)
        layout.addWidget(feedback_container)
        self.feedback_frame.hide()
        
        # Behavior feedback
        behavior_label = QLabel("NOTIFICATION BEHAVIOR")
        behavior_label.setFont(QFont("Helvetica", 9, QFont.Weight.Bold))
        behavior_label.setStyleSheet(f"color: {self.text_dim.name()}; padding-left: 15px;")
        layout.addWidget(behavior_label)
        layout.addSpacing(5)
        
        btn_container = QWidget()
        btn_layout = QHBoxLayout(btn_container)
        btn_layout.setContentsMargins(15, 0, 15, 15)
        btn_layout.setSpacing(4)
        
        self.btn_helpful = self.create_feedback_button("👍\nHelpful", "#1e3a1e", self.success.name(), "helpful")
        self.btn_toomuch = self.create_feedback_button("⚠\nToo Much", "#3a2d1e", "#ffa726", "too_much")
        self.btn_cant = self.create_feedback_button("🚫\nCan't", "#3a1e1e", self.warning.name(), "cant")
        
        btn_layout.addWidget(self.btn_helpful)
        btn_layout.addWidget(self.btn_toomuch)
        btn_layout.addWidget(self.btn_cant)
        layout.addWidget(btn_container)
        
        # Footer
        footer = QLabel("Always on top • Real-time ML detection")
        footer.setFont(QFont("Helvetica", 8))
        footer.setStyleSheet("color: #555555;")
        footer.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(footer)
        layout.addSpacing(10)
        
        layout.addStretch()
        
        # Setup signals
        self.signals = UpdateSignals()
        self.signals.frame_update.connect(self.update_frame)
        self.signals.status_update.connect(self.update_status)
        self.signals.progress_update.connect(self.update_progress)
        self.signals.stats_update.connect(self.update_stats)
        self.signals.feedback_prompt.connect(self.show_feedback_prompt)
        
        # Start detection thread
        self.detection_thread = DetectionThread(self.signals)
        self.detection_thread.start()
    
    def create_feedback_button(self, text, bg, fg, feedback_type):
        btn = QPushButton(text)
        btn.setFont(QFont("Helvetica", 10, QFont.Weight.Bold))
        btn.setFixedHeight(60)
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {bg};
                color: {fg};
                border: none;
                border-radius: 4px;
            }}
            QPushButton:hover {{
                background-color: {bg}dd;
            }}
        """)
        btn.clicked.connect(lambda: handle_feedback_ui(feedback_type))
        return btn
    
    def update_frame(self, frame):
        # Convert to QImage
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        scaled = pixmap.scaled(320, 180, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.video_label.setPixmap(scaled)
    
    def update_status(self, main, sub):
        if "CLENCHING" in main.upper() or "GRINDING" in main.upper():
            color = self.warning.name()
        elif "relaxed" in main.lower():
            color = self.success.name()
        elif "Calibrating" in main:
            color = self.accent.name()
        else:
            color = self.text_light.name()
        
        self.status_label.setText(main.upper())
        self.status_label.setStyleSheet(f"color: {color};")
        self.substatus_label.setText(sub)
    
    def update_progress(self, progress):
        self.progress_bar.setValue(int(progress * 100))
    
    def update_stats(self):
        count = agent_state['session_clench_count']
        samples = len(model_state["X"])
        self.stats_label.setText(f"⚡ Alerts: {count}  •  📊 Training: {samples}")
    
    def show_feedback_prompt(self, show):
        if show:
            self.feedback_frame.show()
        else:
            self.feedback_frame.hide()
    
    def start_calibration(self):
        agent_state["calibrating"] = True
        agent_state["calib_start_time"] = time.time()
        self.calib_button.setEnabled(False)
        self.calib_button.setText("⏳ CALIBRATING...")
        
        # Re-enable after calibration
        QTimer.singleShot(5500, self.finish_calibration)
    
    def finish_calibration(self):
        self.calib_button.setEnabled(True)
        self.calib_button.setText("✓ RECALIBRATE")
    
    def closeEvent(self, event):
        self.detection_thread.running = False
        event.accept()

def main():
    app = QApplication(sys.argv)
    window = TMJCoachWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
