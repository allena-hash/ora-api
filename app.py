from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import requests
import numpy as np
import similarity

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

@app.after_request
def add_cors(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response

@app.route("/", defaults={"path": ""}, methods=["OPTIONS"])
@app.route("/<path:path>", methods=["OPTIONS"])
def options_handler(path=""):
    return jsonify({}), 200

# ── Config ───────────────────────────────────────────────────
# ▼▼▼ PASTE YOUR VALUES HERE ▼▼▼
SB_URL = os.environ.get("SUPABASE_URL", "YOUR_SUPABASE_URL")
SB_KEY = os.environ.get("SUPABASE_KEY", "YOUR_SUPABASE_ANON_KEY")
HF_SPACE = os.environ.get("HF_SPACE", "YOUR_HF_SPACE_URL")  # e.g. https://username-ora-emotion.hf.space
# ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

SB_HEADERS = {
    "apikey": SB_KEY,
    "Authorization": f"Bearer {SB_KEY}",
    "Content-Type": "application/json",
    "Prefer": "return=representation"
}

def sb_get(path):
    r = requests.get(f"{SB_URL}/rest/v1/{path}", headers=SB_HEADERS, timeout=10)
    if not r.ok:
        print(f"Supabase error {r.status_code}: {r.text}")
        r.raise_for_status()
    return r.json()

def sb_post(path, body):
    r = requests.post(
        f"{SB_URL}/rest/v1/{path}",
        headers=SB_HEADERS,
        json=body,
        timeout=10
    )
    if not r.ok:
        print(f"Supabase error {r.status_code}: {r.text}")
        r.raise_for_status()
    return r.json()

# ── Health ───────────────────────────────────────────────────
@app.route("/health")
def health():
    return jsonify({"status": "ok", "version": "ora-v2"})

# ── Process audio via HuggingFace Space ──────────────────────
@app.route("/process-audio", methods=["POST"])
def process_audio():
    """
    Receives audio blob from frontend
    Sends to HuggingFace Space for: transcription + emotion + voiceprint
    Returns all results
    """
    if "audio" not in request.files:
        return jsonify({"error": "No audio file"}), 400

    audio_file = request.files["audio"]

    try:
        # Forward to HuggingFace Space
        hf_response = requests.post(
            f"{HF_SPACE}/process",
            files={"audio": ("recording.webm", audio_file.stream, "audio/webm")},
            timeout=120  # Models need time on first run
        )

        if not hf_response.ok:
            raise Exception(f"HF Space error: {hf_response.text}")

        data = hf_response.json()
        return jsonify(data)

    except Exception as e:
        print(f"Process audio error: {e}")
        # Fallback — return neutral if HF Space fails
        return jsonify({
            "transcript": "",
            "language": "en",
            "emotion": "neutral",
            "detail": "quietly-present",
            "voiceprint": [],
            "is_clean": True,
            "error": str(e)
        })

# ── Find similar responses ────────────────────────────────────
@app.route("/similar", methods=["POST"])
def find_similar():
    """
    POST { "text": "...", "emotion": "sad", "question_id": "uuid", "exclude_id": "uuid", "n": 2 }
    Returns { "similar": [...responses], "scores": [...] }
    """
    body = request.get_json()
    text = body.get("text", "").strip()
    emotion = body.get("emotion", "neutral")
    question_id = body.get("question_id")
    exclude_id = body.get("exclude_id")
    n = int(body.get("n", 2))

    if not question_id:
        return jsonify({"error": "question_id required"}), 400

    # Fetch responses for this question
    path = f"responses?question_id=eq.{question_id}&select=*"
    if exclude_id:
        path += f"&id=neq.{exclude_id}"

    try:
        responses = sb_get(path)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    if not responses:
        return jsonify({"similar": [], "scores": []})

    # Filter by same or related emotion first
    RELATED = {
        "happy":       ["hopeful","excited","relieved","calm"],
        "sad":         ["lonely","numb","overwhelmed","nostalgic"],
        "lonely":      ["sad","numb","nostalgic"],
        "anxious":     ["overwhelmed","frustrated","confused"],
        "overwhelmed": ["anxious","frustrated","sad"],
        "angry":       ["frustrated","overwhelmed"],
        "frustrated":  ["angry","overwhelmed","anxious"],
        "nostalgic":   ["sad","lonely","hopeful"],
        "hopeful":     ["happy","calm","relieved"],
        "calm":        ["hopeful","relieved","neutral"],
        "numb":        ["lonely","sad","confused"],
        "excited":     ["happy","hopeful"],
        "confused":    ["anxious","overwhelmed"],
        "neutral":     ["calm","hopeful"],
    }
    related = RELATED.get(emotion, [])

    # Priority pool — same emotion first
    exact = [r for r in responses if r.get("emotion") == emotion]
    close = [r for r in responses if r.get("emotion") in related and r.get("emotion") != emotion]
    rest  = [r for r in responses if r.get("emotion") not in [emotion] + related]

    # Build ordered pool
    pool = exact + close + rest

    if not pool:
        return jsonify({"similar": [], "scores": []})

    # Use TF-IDF on transcripts within the pool
    if text and len(text.strip()) > 3:
        candidates_text = [r.get("transcript", "") for r in pool]
        top_indices, scores = similarity.top_n_similar(text, candidates_text, n=n)
        similar = [pool[i] for i in top_indices]
        return jsonify({
            "similar": similar,
            "scores": [round(float(s), 3) for s in scores]
        })
    else:
        # No text — return top n from emotion pool
        import random
        result = pool[:n] if len(pool) >= n else pool
        return jsonify({"similar": result, "scores": [1.0] * len(result)})

# ── Match speaker voiceprint ──────────────────────────────────
@app.route("/match-speaker", methods=["POST"])
def match_speaker():
    """
    POST { "voiceprint": [...] }
    Fetches all stored voiceprints from Supabase and finds match
    Returns { "match": true/false, "speaker_id": "uuid" or null, "score": 0.87 }
    """
    body = request.get_json()
    new_print = np.array(body.get("voiceprint", []))

    if len(new_print) == 0:
        return jsonify({"match": False, "speaker_id": None, "score": 0})

    try:
        speakers = sb_get("speakers?select=id,voiceprint")
    except Exception as e:
        return jsonify({"match": False, "speaker_id": None, "score": 0, "error": str(e)})

    if not speakers:
        return jsonify({"match": False, "speaker_id": None, "score": 0})

    best_score = 0
    best_id = None

    for s in speakers:
        try:
            vp = s.get("voiceprint")
            if not vp:
                continue
            if isinstance(vp, str):
                import json
                vp = json.loads(vp)
            sp = np.array(vp)
            score = float(np.dot(new_print, sp) / (np.linalg.norm(new_print) * np.linalg.norm(sp)))
            if score > best_score:
                best_score = score
                best_id = s["id"]
        except Exception as e:
            print(f"Speaker match error: {e}")
            continue

    match = best_score >= 0.75
    return jsonify({
        "match": match,
        "speaker_id": best_id if match else None,
        "score": round(best_score, 3)
    })

# ── Get speaker impact — for returning visitor ────────────────
@app.route("/speaker-impact/<speaker_id>", methods=["GET"])
def speaker_impact(speaker_id):
    """
    Returns all responses by this speaker and their reactions
    """
    try:
        responses = sb_get(f"responses?speaker_id=eq.{speaker_id}&select=*")
        if not responses:
            return jsonify({"responses": [], "total_heard": 0, "total_reactions": 0})

        response_ids = [r["id"] for r in responses]
        total_reactions = 0
        for rid in response_ids:
            try:
                reactions = sb_get(f"reactions?response_id=eq.{rid}&select=id")
                total_reactions += len(reactions)
            except:
                pass

        return jsonify({
            "responses": responses,
            "total_heard": len(responses),
            "total_reactions": total_reactions
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
