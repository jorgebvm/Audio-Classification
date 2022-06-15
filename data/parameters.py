from pathlib import Path

PATH: str = Path(__file__).parent.parent
LOAD: bool = True
RETRAIN: bool = False

PRINT_INTERVAL: int = 1
NUM_EPOCHS: int = 3

BATCH_SIZE: int = 16
NUM_WORKERS: int = 2
SHUFFLE: bool = True

NUM_CLASSES: int = 35
CLASSES_MAPPING = {
    'backward': 0,
    'bed': 1,
    'bird': 2,
    'cat': 3,
    'dog': 4,
    'down': 5,
    'eight': 6,
    'five': 7,
    'follow': 8,
    'forward': 9,
    'four': 10,
    'go': 11,
    'happy': 12,
    'house': 13,
    'learn': 14,
    'left': 15,
    'marvin': 16,
    'nine': 17,
    'no': 18,
    'off': 19,
    'on': 20,
    'one': 21,
    'right': 22,
    'seven': 23,
    'sheila': 24,
    'six': 25,
    'stop': 26,
    'three': 27,
    'tree': 28,
    'two': 29,
    'up': 30,
    'visual': 31,
    'wow': 32,
    'yes': 33,
    'zero': 34
}