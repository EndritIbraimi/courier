"""
make_music.py -- creates assets/music.wav, the game's background music.

The music is made from scratch with simple sound waves (no copyrighted
songs). Run it only if you want to change the tune:  python3 make_music.py
"""
import os
import wave
import numpy as np

SR = 22050                       # samples per second (sound quality)
BPM = 120
BEAT = 60 / BPM                  # seconds per beat
BAR = 4 * BEAT                   # 4 beats per bar
BARS = 16                        # total length: 16 bars = 32 seconds, loops forever
N = int(SR * BAR * BARS)
buf = np.zeros(N)                # the whole song, starts silent
rng = np.random.default_rng(1)


def hz(midi):
    """Note number -> frequency in Hz (A4 = 69 = 440 Hz)."""
    return 440 * 2 ** ((midi - 69) / 12)


def tone(midi, dur, kind, vol, decay):
    """One note. kind = wave shape, decay = how fast it fades out."""
    t = np.arange(int(SR * dur)) / SR
    ph = (t * hz(midi)) % 1
    wave_ = {"sq": np.where(ph < 0.5, 1.0, -1.0) * 0.7,        # buzzy, retro
             "tri": 2 * np.abs(2 * ph - 1) - 1,                 # soft
             "sine": np.sin(2 * np.pi * ph)}[kind]              # pure
    return wave_ * np.exp(-decay * t) * np.minimum(1, t / 0.005) * vol


def put(sound, start):
    """Mix a sound into the song at `start` seconds."""
    i = int(start * SR)
    j = min(N, i + len(sound))
    buf[i:j] += sound[:j - i]


# chords: (bass note, chord notes). Progression: Am - F - C - G
CHORDS = [(45, [57, 60, 64]), (41, [57, 60, 65]), (48, [55, 60, 64]), (43, [55, 59, 62])]
# melody for the second half, 4 notes per bar (A minor pentatonic = always sounds nice)
MELODY = [[76, 74, 72, 74], [72, 72, 69, 72], [76, 79, 76, 74], [74, 72, 74, 67]]

kick_t = np.arange(int(SR * 0.15)) / SR
kick = np.sin(2 * np.pi * np.cumsum(50 + 100 * np.exp(-kick_t * 40)) / SR) * np.exp(-kick_t * 22) * 0.55

for bar in range(BARS):
    t0 = bar * BAR
    root, notes = CHORDS[bar % 4]
    for m in notes:                                              # soft pad
        put(tone(m - 12, BAR, "sine", 0.05, 0.4), t0)
    for b in range(4):                                           # bass line
        put(tone(root + (12 if b == 3 else 0), BEAT * 0.9, "tri", 0.30, 3), t0 + b * BEAT)
    for i, idx in enumerate([0, 1, 2, 1, 0, 1, 2, 1]):           # arpeggio, 8 notes per bar
        put(tone(notes[idx] + 12, BEAT / 2 * 0.95, "sq", 0.06, 9), t0 + i * BEAT / 2)
    for b in (0, 2):                                             # kick drum
        put(kick, t0 + b * BEAT)
    for b in (0.5, 1.5, 2.5, 3.5):                               # hi-hat
        put(rng.uniform(-1, 1, int(SR * 0.05)) * np.exp(-np.arange(int(SR * 0.05)) / SR * 80) * 0.10, t0 + b * BEAT)
    if bar >= 8:                                                 # melody joins in bar 9
        for b, m in enumerate(MELODY[bar % 4]):
            put(tone(m, BEAT * 0.95, "tri", 0.20, 2.5), t0 + b * BEAT)

buf = buf / np.abs(buf).max() * 0.8                              # keep it from clipping
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "music.wav")
os.makedirs(os.path.dirname(out), exist_ok=True)
with wave.open(out, "wb") as f:
    f.setnchannels(1)
    f.setsampwidth(2)
    f.setframerate(SR)
    f.writeframes((buf * 32767).astype(np.int16).tobytes())
print("Saved", out)
