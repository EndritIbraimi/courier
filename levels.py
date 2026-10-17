"""levels.py -- the list of levels. Each week we flip one "ready" to True."""

LEVELS = [
    {"id": 1, "name": "Empty Map",       "desc": "One package, one goal.",      "ready": True},
    {"id": 2, "name": "Walls",           "desc": "Obstacles block the way.",    "ready": False},
    {"id": 3, "name": "Battery",         "desc": "Recharge to keep going.",     "ready": False},
    {"id": 4, "name": "Many Packages",   "desc": "Deadlines: pick the order.",  "ready": False},
    {"id": 5, "name": "Big City",        "desc": "Bigger map, smarter AI.",     "ready": False},
    {"id": 6, "name": "Patrol Guards",   "desc": "Guards walk fixed paths.",    "ready": False},
    {"id": 7, "name": "Fog",             "desc": "Robot sees only nearby.",     "ready": False},
    {"id": 8, "name": "Learning Guards", "desc": "The guards learn too.",       "ready": False},
]

# Average score of a completely random robot on level 1 (from the baseline run).
# Shown after you play, so you can compare yourself with the random robot.
RANDOM_BASELINE = -73
