#!/usr/bin/env python3
"""
SNOUT trainer: one script, two modes.

  python3 snout_trainer.py learn   # sniff something, say what it is, save it
  python3 snout_trainer.py test    # sniff something, SNOUT guesses, you say if it's right
  python3 snout_trainer.py         # asks which mode

LEARN
  Pick what you are smelling from a numbered list of labels you've used before
  (or add a new one). Typed names are matched against existing labels, so
  "pizzza" or "Pizza " gets caught instead of silently becoming a new class.
  Each sniff is appended to snout_sniff_log.csv (same format as
  snout_log_sniff.py, so your old data still works).

TEST
  Trains a Random Forest on the CSV, sniffs, and prints its guess with
  confidence ("I think this is pizza (87%)" or "Not sure"). You then say
  whether it was right. Results go to snout_test_log.csv so you can track
  real accuracy. Optionally add the sniff to the training data and retrain.

Try it without hardware:  python3 snout_trainer.py learn --sim --fast
Needs (test mode only): pandas, scikit-learn.
"""

import argparse
import csv
import difflib
import math
import os
import random
import select
import statistics
import sys
import time
from collections import Counter

# ---- config ---------------------------------------------------------------

# Sensor name -> MCP3008 channel, matching the 7 Sept notebook wiring (CH0 unused).
CHANNEL_MAP = {"MQ2": 7, "MQ3": 6, "MQ4": 5, "MQ5": 4, "MQ6": 3, "MQ8": 2, "MQ135": 1}
CHANNELS = list(CHANNEL_MAP)

SNIFF_SECONDS = 18          # total active sniff window
BASELINE_SECONDS = 2        # leading portion of the window treated as baseline
SAMPLE_HZ = 5               # samples per second per channel
PURGE_SECONDS = 45          # clean-air purge between sniffs

TRAIN_CSV = "snout_sniff_log.csv"
TEST_CSV = "snout_test_log.csv"

CONFIDENCE_THRESHOLD = 0.50  # below this SNOUT says "not sure"
MIN_PER_CLASS = 3            # classes with fewer sniffs are left out of the model
MIN_USEFUL_DELTA = 5         # raw counts (~16 mV); below this the sniff looks empty
FUZZY_CUTOFF = 0.75          # how similar a typed label must be to an existing one
CLEAN_AIR_LABELS = {"ambient", "air", "clean air", "background", "baseline", "empty", "nothing"}

# ---- ADC --------------------------------------------------------------------


class RealADC:
    def __init__(self):
        try:
            import spidev
        except ImportError:
            sys.exit("spidev not installed. On the Pi: sudo apt install python3-spidev\n"
                     "(or run with --sim to try the program without hardware)")
        self.spi = spidev.SpiDev()
        self.spi.open(0, 0)
        self.spi.max_speed_hz = 1350000

    def begin_sniff(self):
        pass

    def read(self, ch: int) -> int:
        """Raw 10-bit value (0-1023) from MCP3008 channel ch."""
        resp = self.spi.xfer2([1, (8 + ch) << 4, 0])
        return ((resp[1] & 3) << 8) + resp[2]

    def close(self):
        self.spi.close()


class SimADC:
    """Fake sensors for testing the program. Each label gets a repeatable 'smell fingerprint'."""

    def __init__(self):
        self.target = "ambient"
        self._calls = 0
        self._profile = {}
        self._base = {}
        self._rng = random.Random()

    def begin_sniff(self):
        self._calls = 0
        prof_rng = random.Random(self.target)  # same label -> same fingerprint
        scale = 0.0 if self.target == "ambient" else 1.0
        self._profile = {n: prof_rng.uniform(8, 140) * scale * self._rng.uniform(0.85, 1.15)
                         for n in CHANNELS}
        self._base = {n: 150 + 40 * i for i, n in enumerate(CHANNELS)}

    def read(self, ch: int) -> int:
        name = next(n for n, c in CHANNEL_MAP.items() if c == ch)
        t = (self._calls // len(CHANNELS)) / SAMPLE_HZ
        self._calls += 1
        rise = 0.0 if t < BASELINE_SECONDS else 1 - math.exp(-(t - BASELINE_SECONDS) / 5)
        return max(0, min(1023, round(self._base[name] + self._profile[name] * rise
                                      + self._rng.gauss(0, 1.5))))

    def close(self):
        pass


# ---- features (identical to snout_log_sniff.py) ------------------------------


def extract_features(samples: dict) -> dict:
    baseline_n = BASELINE_SECONDS * SAMPLE_HZ
    features = {}
    for name, vals in samples.items():
        baseline = statistics.mean(vals[:baseline_n]) if len(vals) >= baseline_n else vals[0]
        signal = vals[baseline_n:] if len(vals) > baseline_n else vals
        peak = max(signal) if signal else baseline
        peak_idx = signal.index(peak) if signal else 0
        time_to_peak = peak_idx / SAMPLE_HZ if signal else 0.0

        features[f"{name}_baseline"] = round(baseline, 2)
        features[f"{name}_peak"] = round(peak, 2)
        features[f"{name}_delta"] = round(peak - baseline, 2)
        features[f"{name}_rise_rate"] = round((peak - baseline) / time_to_peak, 3) if time_to_peak > 0 else 0.0
    return features


def run_sniff(adc, fast: bool) -> dict:
    samples = {name: [] for name in CHANNELS}
    n_reads = SNIFF_SECONDS * SAMPLE_HZ
    interval = 0 if fast else 1.0 / SAMPLE_HZ

    print(f"Sniffing for {SNIFF_SECONDS}s... hold target ~2-5 cm from the array.")
    adc.begin_sniff()
    for i in range(n_reads):
        for name, ch in CHANNEL_MAP.items():
            samples[name].append(adc.read(ch))
        if interval:
            time.sleep(interval)
        if i % SAMPLE_HZ == 0 and not fast:
            print(f"  {i // SAMPLE_HZ}s")
    print("Sniff complete.")
    return extract_features(samples)


def describe_sniff(features: dict, expect_response: bool = True) -> bool:
    """Print the strongest responses. Returns False if the sniff looks empty
    (only flagged when a response was expected, so clean-air labels don't trigger it)."""
    deltas = sorted(((n, features[f"{n}_delta"]) for n in CHANNELS), key=lambda t: -abs(t[1]))
    print("Strongest responses: " + ", ".join(f"{n} {d:+.0f}" for n, d in deltas[:3]))
    if expect_response and max(abs(d) for _, d in deltas) < MIN_USEFUL_DELTA:
        print("WARNING: almost no response on any sensor. Was the target in position? "
              "Are the sensors warmed up?")
        return False
    return True


def purge(seconds: int, fast: bool):
    if fast or seconds <= 0:
        return
    print(f"Purge {seconds}s in clean air. Press Enter to skip.")
    for remaining in range(seconds, 0, -1):
        sys.stdout.write(f"\r  {remaining:>3}s ")
        sys.stdout.flush()
        ready, _, _ = select.select([sys.stdin], [], [], 1)
        if ready:
            sys.stdin.readline()
            break
    print("\r  done.   ")


# ---- CSV storage ---------------------------------------------------------------


def append_row(path: str, features: dict, label: str):
    fieldnames = ["timestamp", "label"] + list(features.keys())
    write_header = not os.path.exists(path)
    if not write_header:
        with open(path, newline="") as f:
            existing = next(csv.reader(f), [])
        if existing != fieldnames:
            sys.exit(f"{path} has different columns than this script produces "
                     f"(sensor names or order changed?). Move it aside or use --csv.")
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "label": label, **features})


def label_counts(path: str) -> Counter:
    """Labels in order of first appearance (so menu numbers stay stable)."""
    counts = Counter()
    if os.path.exists(path):
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                if row.get("label"):
                    counts[row["label"]] += 1
    return counts


# ---- label picking ----------------------------------------------------------------


def normalize(text: str) -> str:
    return " ".join(text.lower().replace(",", " ").split())


def ask_yes(prompt: str, default: bool = True) -> bool:
    ans = input(prompt + (" [Y/n] " if default else " [y/N] ")).strip().lower()
    return default if not ans else ans.startswith("y")


def resolve_label(text: str, existing: list, allow_new: bool = True):
    """Turn typed text into an existing label (catching typos) or a confirmed new one."""
    norm = normalize(text)
    if not norm:
        return None
    if norm in existing:
        return norm
    close = difflib.get_close_matches(norm, existing, n=1, cutoff=FUZZY_CUTOFF)
    if close and ask_yes(f"'{norm}' looks like the existing label '{close[0]}'. Use '{close[0]}'?"):
        return close[0]
    if not allow_new:
        print(f"'{norm}' is not a known label.")
        return None
    if ask_yes(f"Create NEW label '{norm}'?"):
        return norm
    return None


def pick_label(counts: Counter, prompt: str = "What are you smelling?", allow_new: bool = True):
    """Numbered menu of previous labels. Returns a label, or None to quit."""
    while True:
        existing = list(counts)
        print(f"\n{prompt}")
        for i, label in enumerate(existing, 1):
            print(f"  {i}) {label:<16} ({counts[label]} sniffs)")
        if allow_new:
            print("  n) new label")
        print("  q) quit")
        if not existing:
            print("  (no labels yet. Tip: start with 'ambient' (clean air) plus your targets.)")

        raw = input("> ").strip()
        low = raw.lower()
        if low in ("q", "quit", "exit"):
            return None
        if raw.isdigit() and 1 <= int(raw) <= len(existing):
            return existing[int(raw) - 1]
        if low in ("n", "new") and allow_new:
            raw = input("New label name: ")
        label = resolve_label(raw, existing, allow_new)
        if label:
            return label


# ---- learn mode -----------------------------------------------------------------------


def learn(adc, args):
    print("\n=== LEARN MODE ===")
    print("Sensors need ~10 min of warm-up after power-on before data is trustworthy.")
    saved = 0
    while True:
        counts = label_counts(args.csv)
        label = pick_label(counts)
        if label is None:
            break
        if isinstance(adc, SimADC):
            adc.target = label
        input(f"Put '{label}' in position, then press Enter to sniff...")
        features = run_sniff(adc, args.fast)
        looks_ok = describe_sniff(features, expect_response=label not in CLEAN_AIR_LABELS)
        if ask_yes(f"Save as '{label}'?", default=looks_ok):
            append_row(args.csv, features, label)
            saved += 1
            print(f"Saved. '{label}' now has {counts.get(label, 0) + 1} sniffs.")
        else:
            print("Discarded.")
        purge(PURGE_SECONDS, args.fast)
    print(f"\nLearn session over: {saved} sniffs saved to {args.csv}.")


# ---- test mode --------------------------------------------------------------------------


def train_model(path: str):
    """Fit a Random Forest on the CSV. Returns (model, feature_columns, counts) or None."""
    try:
        import pandas as pd
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import StratifiedKFold, cross_val_score
    except ImportError:
        sys.exit("Test mode needs pandas and scikit-learn: sudo apt install python3-pandas python3-sklearn")

    if not os.path.exists(path):
        print(f"No training data at {path}. Run learn mode first.")
        return None
    df = pd.read_csv(path)
    counts = df["label"].value_counts()
    small = counts[counts < MIN_PER_CLASS]
    if len(small):
        print("Leaving out labels with too few sniffs (need " + f"{MIN_PER_CLASS}+): "
              + ", ".join(f"{l} ({n})" for l, n in small.items()))
        df = df[~df["label"].isin(small.index)]
    if df["label"].nunique() < 2:
        print("Need at least 2 labels with enough sniffs to test. Run learn mode first.")
        return None

    X = df.drop(columns=["timestamp", "label"])
    y = df["label"]
    clf = RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced")

    n_splits = min(5, int(y.value_counts().min()))
    if n_splits >= 2:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        scores = cross_val_score(clf, X, y, cv=cv)
        print(f"Cross-validated accuracy on saved data: {scores.mean():.0%} "
              f"({len(df)} sniffs, {y.nunique()} labels)")
    clf.fit(X, y)
    return clf, list(X.columns), y.value_counts()


def predict(model, columns, features):
    import pandas as pd
    x = pd.DataFrame([features]).reindex(columns=columns, fill_value=0)
    proba = model.predict_proba(x)[0]
    ranked = sorted(zip(model.classes_, proba), key=lambda t: -t[1])
    return ranked


def log_test(path: str, predicted: str, confidence: float, actual: str):
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["timestamp", "predicted", "confidence", "actual", "correct"])
        w.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), predicted, f"{confidence:.3f}",
                    actual, int(predicted == actual)])


def test(adc, args):
    print("\n=== TEST MODE ===")
    trained = train_model(args.csv)
    if trained is None:
        return
    model, columns, _ = trained

    results = []  # (actual, predicted)
    while True:
        try:
            raw = input("\nPress Enter to sniff (q to quit)... ").strip().lower()
        except EOFError:
            break
        if raw in ("q", "quit"):
            break

        if isinstance(adc, SimADC):
            adc.target = random.choice(list(label_counts(args.csv)))
        features = run_sniff(adc, args.fast)
        describe_sniff(features, expect_response=False)  # clean air is a valid answer

        ranked = predict(model, columns, features)
        print()
        for label, p in ranked[:3]:
            print(f"  {label:<16} {p:>4.0%}  {'#' * round(p * 30)}")
        guess, conf = ranked[0]
        if conf >= CONFIDENCE_THRESHOLD:
            print(f"\nI think this is: {guess.upper()} ({conf:.0%})")
        else:
            print(f"\nNot sure. Best guess is {guess} but only {conf:.0%} "
                  f"(below {CONFIDENCE_THRESHOLD:.0%}).")
        if isinstance(adc, SimADC):
            print(f"(sim: the fake target was '{adc.target}')")

        ans = input("Was that right? [y = yes, n = no, s = skip] ").strip().lower()
        if ans.startswith("s") or not ans:
            purge(PURGE_SECONDS, args.fast)
            continue
        if ans.startswith("y"):
            actual = guess
        else:
            actual = pick_label(label_counts(args.csv), "What was it actually?")
            if actual is None:
                purge(PURGE_SECONDS, args.fast)
                continue

        log_test(args.test_csv, guess, conf, actual)
        results.append((actual, guess))
        print("Correct!" if actual == guess else f"Noted: it was {actual}, SNOUT said {guess}.")

        if ask_yes("Add this sniff to the training data and retrain?", default=False):
            append_row(args.csv, features, actual)
            retrained = train_model(args.csv)
            if retrained:
                model, columns, _ = retrained
        purge(PURGE_SECONDS, args.fast)

    if results:
        right = sum(a == p for a, p in results)
        print(f"\nSession: {right}/{len(results)} correct ({right / len(results):.0%}). "
              f"Log: {args.test_csv}")
        misses = Counter((a, p) for a, p in results if a != p)
        for (a, p), n in misses.most_common():
            print(f"  mixed up: actually {a}, said {p} (x{n})")


# ---- main -------------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description="SNOUT trainer (learn / test)")
    ap.add_argument("mode", nargs="?", choices=["learn", "test"])
    ap.add_argument("--csv", default=TRAIN_CSV, help="training data file")
    ap.add_argument("--test-csv", default=TEST_CSV, help="test results file")
    ap.add_argument("--sim", action="store_true", help="fake sensors, no hardware needed")
    ap.add_argument("--fast", action="store_true", help="skip sleeps and purge (for --sim)")
    args = ap.parse_args()

    mode = args.mode
    if mode is None:
        choice = input("Mode: 1) learn  2) test  > ").strip().lower()
        mode = {"1": "learn", "learn": "learn", "l": "learn",
                "2": "test", "test": "test", "t": "test"}.get(choice)
        if mode is None:
            sys.exit("Pick learn or test.")

    adc = SimADC() if args.sim else RealADC()
    try:
        (learn if mode == "learn" else test)(adc, args)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        adc.close()


if __name__ == "__main__":
    main()
