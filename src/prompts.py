TRIGGER = "pxlnes style"

_BASE_PROMPTS = [
    "a knight character, pixel art",
    "a treasure chest, pixel art",
    "a skeleton enemy, pixel art",
    "a stone dungeon wall tile, pixel art",
    "a health potion, pixel art",
    "a wizard character, pixel art",
    "a torch on a wall, pixel art",
    "a sword weapon icon, pixel art",
]

PROMPTS = [f"{p}, {TRIGGER}" for p in _BASE_PROMPTS]
