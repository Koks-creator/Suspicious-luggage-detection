from typing import Union
from pathlib import Path


class Config:
    PROJECT_MAIN_PATH: str = Path(__file__).resolve().parent
    VIDEOS_FOLDER: str = rf"{PROJECT_MAIN_PATH}\TestVideos"
    VIDEO_CAPTURE: Union[int, str] = rf"{VIDEOS_FOLDER}\3723452-hd_1366_720_24fps.mp4"
    MODEL_FOLDER: str = rf"{PROJECT_MAIN_PATH}\Model"
    MODEL_NAME: str = "best2.pt"
    AREAS_FOLDER: str = "Areas"
    AREAS_FILE: str = fr"{AREAS_FOLDER}/areas.pkl"
    SORT_MAX_AGE: int = 20
    FRAME_COUNT_THR: int = 10
    FRAME_COUNT_STEP: int = 1
    ERROR_MARGIN: int = 20  # in pixels
    DEBUG: bool = True
    ALERTS_INTERVAL: int = 30
    TELEGRAM_MESSAGES: bool = True
    STATES: dict = {
        5: {"Name": "Warning", "Color": (0, 200, 200)},
        10: {"Name": "Suspicious", "Color": (0, 140, 255)},
        15: {"Name": "Very Suspicious", "Color": (0, 0, 200)}
    }
