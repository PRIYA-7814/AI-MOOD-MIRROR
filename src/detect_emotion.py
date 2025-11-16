# src/detect_emotion.py
import os
import cv2
import numpy as np

# default module-level handles (ensure defined regardless of import branches)
DeepFace = None
fer_module = None
FerClass = None
fer_detector = None
mp = None
mp_face_mesh = None
mp_drawing = None

# Try DeepFace first (best). If not available, try fer. If neither, use mediapipe heuristic.
USE_BACKEND = None

# Try DeepFace unless disabled via environment (useful for cloud deploys)
DeepFace = None
if os.getenv('DISABLE_DEEPFACE', '0') != '1':
    try:
        from deepface import DeepFace
        USE_BACKEND = "deepface"
    except Exception:
        DeepFace = None

# Try fer
if USE_BACKEND is None:
    try:
        # in some versions the API is different; import safely and instantiate lazily
        import importlib
        fer_module = importlib.import_module('fer')
        # try common class names exported by different fer versions
        FerClass = getattr(fer_module, 'FER', None) or getattr(fer_module, 'FERDetector', None) or getattr(fer_module, 'Detector', None)
        if FerClass is not None:
            fer_detector = None  # instantiate later when needed
            USE_BACKEND = "fer"
        else:
            # if the class isn't found, don't set backend
            fer_module = None
    except Exception:
        fer_module = None
        FerClass = None
        fer_detector = None

# Prepare mediapipe fallback for a simple smile detector
if USE_BACKEND is None:
    try:
        import mediapipe as mp
        mp_face_mesh = mp.solutions.face_mesh
        mp_drawing = mp.solutions.drawing_utils
        USE_BACKEND = "mediapipe"
    except Exception:
        mp = None
        mp_face_mesh = None
        mp_drawing = None
        USE_BACKEND = "none"

# Utility: convert BGR frame to RGB for analyzers that expect RGB
def to_rgb(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

def get_emotion(frame):
    """
    Returns (emotion_label, confidence)
    emotion_label: one of ('happy','sad','angry','surprise','neutral','fear','disgust') or 'unknown'
    confidence: float 0..1
    """
    if USE_BACKEND == "deepface" and DeepFace is not None:
        try:
            # DeepFace.analyze expects RGB or BGR depending on version; pass RGB to be safe
            res = DeepFace.analyze(to_rgb(frame), actions=['emotion'], enforce_detection=False)
            # normalize response: some versions return a list
            if isinstance(res, list) and len(res) > 0:
                res = res[0]
            # try multiple keys for dominant emotion
            emotion = None
            if isinstance(res, dict):
                emotion = res.get('dominant_emotion') or res.get('dominant_emotions') or res.get('dominant')
            # deepface provides emotion dict under different keys in some versions
            emotion_scores = {}
            if isinstance(res, dict):
                emotion_scores = res.get('emotion') or res.get('emotions') or {}
            confidence = 0.0
            try:
                if emotion and isinstance(emotion_scores, dict) and emotion in emotion_scores:
                    val = emotion_scores[emotion]
                    # some versions give percentages (0-100)
                    confidence = float(val) / 100.0 if val > 1 else float(val)
            except Exception:
                confidence = 0.0
            # if we couldn't determine, print the raw result for debugging
            if (not emotion) or (confidence == 0.0):
                try:
                    print("[deepface debug] raw analyze result:", res)
                except Exception:
                    pass
            return (emotion if emotion else "unknown", min(confidence, 1.0))
        except Exception:
            return ("unknown", 0.0)

    if USE_BACKEND == "fer" and FerClass is not None:
        try:
            # instantiate detector lazily (avoids heavy work at import time)
            global fer_detector
            if fer_detector is None:
                try:
                    fer_detector = FerClass()
                except Exception:
                    # some fer versions may need no-arg init or different factory — try default
                    fer_detector = FerClass()
            # fer detector expects RGB frames too
            rgb = to_rgb(frame)
            results = fer_detector.detect_emotions(rgb)
            if results and isinstance(results, list) and len(results) > 0:
                emotions = results[0].get("emotions", {})
                if emotions:
                    top = max(emotions, key=emotions.get)
                    conf = emotions[top]
                    return (top, float(conf))
            return ("unknown", 0.0)
        except Exception:
            return ("unknown", 0.0)

    if USE_BACKEND == "mediapipe" and mp_face_mesh is not None:
        # Very simple heuristic smile detector using landmarks around lips.
        # This is not a full emotion detector but works as a demo fallback: "happy" vs "neutral"
        try:
            rgb = to_rgb(frame)
            h, w, _ = frame.shape
            with mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True) as face_mesh:
                results = face_mesh.process(rgb)
                if not results or not results.multi_face_landmarks:
                    return ("unknown", 0.0)
                lm = results.multi_face_landmarks[0].landmark
                # choose a few lip landmarks: upper lip and lower lip
                # indices from Mediapipe FaceMesh: top/lower lip region (approx)
                # Using normalized coordinates -> convert to pixel
                def xy(i):
                    return np.array([lm[i].x * w, lm[i].y * h])

                # approximate: upper inner lip 13, lower inner lip 14 (these indices may vary)
                # Use a small set to calculate vertical mouth opening vs width
                try:
                    top_lip = xy(13)
                    bottom_lip = xy(14)
                    left_mouth = xy(61)
                    right_mouth = xy(291)
                except Exception:
                    # fallback indices if not available
                    top_lip = xy(13)
                    bottom_lip = xy(14)
                    left_mouth = xy(61)
                    # If landmark 291 is not available, approximate the right mouth corner
                    # as an offset from the left mouth corner to avoid indexing errors.
                    if len(lm) > 291:
                        right_mouth = xy(291)
                    else:
                        # approximate a reasonable horizontal offset (pixels)
                        right_mouth = left_mouth + np.array([30.0, 0.0])
                vert = np.linalg.norm(bottom_lip - top_lip)
                hor = np.linalg.norm(right_mouth - left_mouth) + 1e-6
                ratio = vert / hor
                # When smiling, mouth width increases (horizontal) and opening may be moderate.
                # We'll invert ratio to set confidence for "happy".
                # Tune thresholds if needed.
                if ratio > 0.25:
                    return ("surprise", float(min((ratio - 0.25) * 4, 1.0)))
                # Try simple smile detection by curve: compute corner-up movement using other landmarks
                # This is basic — mark happy if width is large compared to face width
                face_width = np.linalg.norm(xy(234) - xy(454)) if len(lm) > 454 else hor * 3
                smile_score = hor / (face_width + 1e-6)
                if smile_score > 0.26:
                    # happy
                    conf = float(min((smile_score - 0.26) * 5, 0.95))
                    return ("happy", conf)
                return ("neutral", 0.4)
        except Exception:
            return ("unknown", 0.0)

    # final fallback
    return ("unknown", 0.0)

