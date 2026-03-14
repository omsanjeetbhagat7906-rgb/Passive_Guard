from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
import math

app = Flask(__name__)
CORS(app)

def extract_features(data):
    mouse   = data.get("mouse_events", [])
    keys    = data.get("key_events", [])
    scrolls = data.get("scroll_events", [])
    clicks  = data.get("click_events", [])
    time_on_page = data.get("time_on_page", 0)
    focus_loss   = data.get("focus_loss_count", 0)

    speeds, prev_dir, dir_changes = [], None, 0
    for i in range(1, len(mouse)):
        dx = mouse[i]["x"] - mouse[i-1]["x"]
        dy = mouse[i]["y"] - mouse[i-1]["y"]
        dt = max(mouse[i]["t"] - mouse[i-1]["t"], 1)
        speeds.append(math.sqrt(dx**2 + dy**2) / dt)
        direction = math.atan2(dy, dx)
        if prev_dir is not None and abs(direction - prev_dir) > 0.3:
            dir_changes += 1
        prev_dir = direction

    avg_speed = float(np.mean(speeds)) if speeds else 0
    speed_var = float(np.var(speeds))  if speeds else 0

    linearity = 0.0
    if len(mouse) >= 2:
        total_path = sum(math.sqrt((mouse[i]["x"]-mouse[i-1]["x"])**2+(mouse[i]["y"]-mouse[i-1]["y"])**2) for i in range(1,len(mouse)))
        straight   = math.sqrt((mouse[-1]["x"]-mouse[0]["x"])**2+(mouse[-1]["y"]-mouse[0]["y"])**2)
        linearity  = straight / max(total_path, 1)

    intervals = [keys[i]["t"]-keys[i-1]["t"] for i in range(1,len(keys)) if 0 < keys[i]["t"]-keys[i-1]["t"] < 5000]
    avg_interval  = float(np.mean(intervals)) if intervals else 0
    interval_var  = float(np.var(intervals))  if intervals else 0
    typing_rhythm = (float(np.std(intervals)) / avg_interval) if avg_interval > 0 else 0

    return [len(mouse), avg_speed, speed_var, dir_changes, len(keys), avg_interval,
            interval_var, len(scrolls), float(np.mean([abs(s["delta"]) for s in scrolls])) if scrolls else 0,
            time_on_page, len(clicks), focus_loss, linearity, typing_rhythm]


class GaussianNaiveBayes:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.priors, self.means, self.stds = {}, {}, {}
        for c in self.classes:
            Xc = X[y == c]
            self.priors[c] = len(Xc) / len(X)
            self.means[c]  = Xc.mean(axis=0)
            self.stds[c]   = Xc.std(axis=0) + 1e-9

    def predict_proba(self, X):
        results = []
        for x in X:
            log_probs = {c: np.log(self.priors[c]) - 0.5*np.sum(np.log(2*np.pi*self.stds[c]**2)+((x-self.means[c])**2/self.stds[c]**2)) for c in self.classes}
            max_lp = max(log_probs.values())
            probs  = {c: np.exp(log_probs[c]-max_lp) for c in self.classes}
            total  = sum(probs.values())
            results.append([probs.get(0,0)/total, probs.get(1,0)/total])
        return np.array(results)


def generate_training_data():
    rng = np.random.default_rng(42)
    n = 2000
    humans = np.column_stack([rng.integers(50,500,n).astype(float), rng.uniform(0.5,5,n), rng.uniform(0.5,10,n),
        rng.integers(5,80,n).astype(float), rng.integers(10,200,n).astype(float), rng.uniform(80,400,n),
        rng.uniform(500,20000,n), rng.integers(0,30,n).astype(float), rng.uniform(10,200,n),
        rng.uniform(10,300,n), rng.integers(1,10,n).astype(float), rng.integers(0,5,n).astype(float),
        rng.uniform(0.1,0.7,n), rng.uniform(0.2,1.5,n)])
    bots = np.column_stack([rng.integers(0,10,n).astype(float), rng.uniform(10,100,n), rng.uniform(0,0.1,n),
        rng.integers(0,3,n).astype(float), rng.integers(0,5,n).astype(float), rng.uniform(1,10,n),
        rng.uniform(0,5,n), rng.integers(0,2,n).astype(float), rng.uniform(0,5,n),
        rng.uniform(0.1,2,n), rng.integers(0,2,n).astype(float), rng.integers(0,1,n).astype(float),
        rng.uniform(0.9,1.0,n), rng.uniform(0,0.05,n)])
    X = np.vstack([humans, bots])
    y = np.array([1]*n + [0]*n)
    return X, y

X_train, y_train = generate_training_data()
model = GaussianNaiveBayes()
model.fit(X_train, y_train)
print("✅ ML model ready — numpy only, no sklearn!")


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/verify", methods=["POST"])
def verify():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data received"}), 400
    try:
        features = extract_features(data)

        # ── Rule-based pre-filter (catches obvious bots instantly) ──
        mouse_count  = features[0]
        key_count    = features[4]
        time_on_page = features[9]
        scroll_count = features[7]

        rule_bot = (
            (mouse_count == 0 and key_count <= 2 and time_on_page < 2) or
            (mouse_count == 0 and time_on_page < 1) or
            (time_on_page < 0.5)
        )
        if rule_bot:
            human_prob, bot_prob = 0.02, 0.98
        else:
            prob = model.predict_proba(np.array(features).reshape(1,-1))[0]
            human_prob, bot_prob = float(prob[1]), float(prob[0])

        if human_prob >= 0.75:   verdict, action, message = "HUMAN",        "ACCESS_GRANTED",    "High confidence human behavior detected."
        elif human_prob >= 0.50: verdict, action, message = "LIKELY_HUMAN", "ACCESS_GRANTED",    "Behavior appears human. Access granted."
        elif human_prob >= 0.30: verdict, action, message = "SUSPICIOUS",   "CHALLENGE_REQUIRED","Ambiguous signals. Please complete a quick challenge."
        else:                    verdict, action, message = "BOT",          "ACCESS_DENIED",     "Automated behavior detected. Access denied."

        names = ["Mouse Events","Avg Speed","Speed Variance","Direction Changes","Keystrokes",
                 "Avg Keystroke Interval","Interval Variance","Scroll Events","Avg Scroll Delta",
                 "Time on Page","Clicks","Focus Loss","Mouse Linearity","Typing Rhythm"]
        return jsonify({"verdict":verdict,"action":action,"message":message,
                        "human_probability":round(human_prob*100,1),"bot_probability":round(bot_prob*100,1),
                        "features":dict(zip(names,[round(f,3) for f in features]))})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/health")
def health():
    return jsonify({"status":"ok","model":"GaussianNaiveBayes","features":14})

if __name__ == "__main__":
    app.run(debug=True, port=5000)
