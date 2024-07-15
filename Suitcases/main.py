from typing import Tuple, List, Union
from dataclasses import dataclass
from time import time
import requests
import json
import cv2
import numpy as np
import pandas as pd
import pickle

from Suitcases.detector import Detector
from Suitcases.sortalg import Sort
from Suitcases.config import Config


@dataclass
class SusSuitcasesApp:
    model_path: str = rf""
    sort_max_ahe: int = 20
    areas_file: str = "areas.pkl"
    debug: bool = False
    telegram_messages: bool = True
    alert_interval: int = 30
    __tokens_file_path: str = "tokens.json"

    def __post_init__(self):
        self.areas = []

        self.detector = Detector(self.model_path)
        self.sort_tracker = Sort(max_age=self.sort_max_ahe)

        self.switch = 0
        if self.areas_file:
            with open(self.areas_file, "rb") as f:
                self.areas = pickle.load(f)

        if self.telegram_messages:
            with open(self.__tokens_file_path) as tf:
                data = json.load(tf)
                self.__bot_token, self.__chat_id = data["BotToken"], data["ChatId"]

    def send_tg_message(self, msg: str) -> None:
        url = f"https://api.telegram.org/bot{self.__bot_token}/sendMessage?" \
              f"chat_id={self.__chat_id}&parse_mode=Markdown&text={msg}"
        response = requests.get(url)
        if response.status_code != 200:
            print(f"Error when making sending tg message: {response.status_code}, {response.content}")

    @staticmethod
    def get_center(bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
        return bbox[0] + (abs(bbox[2]-bbox[0])//2), bbox[1] + (abs(bbox[3]-bbox[1])//2)

    def rectangles_intersect(self, frame: np.array, rect1: list, rect2: list, error_margin: int):
        """
        Checks if the bboxes cross or not
        :param rect1:
        :param rect2:
        :param error_margin: in pixels
        :return:
        """
        x1_min, y1_min, x1_max, y1_max = rect1
        x2_min, y2_min, x2_max, y2_max = rect2

        x1_min -= error_margin
        y1_min -= error_margin
        x1_max += error_margin
        y1_max += error_margin

        # for debug, add frame param
        if self.debug:
            cv2.rectangle(frame, (x1_min, y1_min), (x1_max, y1_max), (0, 255, 255), 2)
        if x1_max < x2_min or x2_max < x1_min:
            return False

        if y1_max < y2_min or y2_max < y1_min:
            return False

        return True

    def check_for_areas(self, suitcase_center: Tuple[int, int]) -> str:
        for area_id, area in enumerate(self.areas):
            if cv2.pointPolygonTest(area, suitcase_center, False) > -1:
                return str(area_id)
        return "Unknown"

    def check_for_owners(self, frame: np.array, suitcase_bb: List[int], detections: pd.DataFrame) -> bool:
        """
        Loops through all people to check if person's bbox crossed with suitcase's bbox, if so, then we found owner
        :param suitcase_bb:
        :param detections:
        :return:
        """
        for detection in detections.iterrows():
            detection = detection[1]
            x1_p, y1_p, x2_p, y2_p, conf, class_id, class_name = detection
            if class_id == 1:  # person
                if self.rectangles_intersect(frame, suitcase_bb, [x1_p, y1_p, x2_p, y2_p], Config.ERROR_MARGIN) and not self.switch:
                    return True
        return False

    def run(self, video_cap: Union[int, str] = Config.VIDEO_CAPTURE,
            frame_count_thr: int = Config.FRAME_COUNT_THR,
            frame_count_step: int = Config.FRAME_COUNT_STEP):
        cap = cv2.VideoCapture(video_cap)

        data = {}
        p_time = 0
        alpha = 0.4
        alert_sent_time = None
        while cap.isOpened():
            detections_ar = np.empty((0, 5))
            success, frame = cap.read()
            if not success:
                break
            overlay = frame.copy()

            if alert_sent_time:
                diff = int(time() - alert_sent_time)
                if diff >= self.alert_interval:
                    alert_sent_time = None

            converted = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_draw, detections = self.detector.detect(img=converted)
            frame_draw = cv2.cvtColor(frame_draw, cv2.COLOR_RGB2BGR)

            for area_id, area in enumerate(self.areas):
                cv2.polylines(overlay, [area], True, (255, 255, 255), 4)
                cv2.fillPoly(overlay, [area], (0, 200, 0))
                M = cv2.moments(area)
                if M['m00'] != 0:
                    area_cx = int(M['m10'] / M['m00'])
                    area_cy = int(M['m01'] / M['m00'])
                    cv2.putText(frame, f"AREA {area_id}", (area_cx, area_cy), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 2)
            final_img = cv2.addWeighted(frame, alpha, overlay, 1 - alpha, 0)


            detections[['xmin', 'ymin', 'xmax', 'ymax']] = detections[['xmin', 'ymin', 'xmax', 'ymax']].astype(int)
            for detection in detections.iterrows():
                detection = detection[1]
                x1, y1, x2, y2, conf, class_id, class_name = detection

                if class_id == 0:  # suitcase
                    curr_arr = np.array([x1, y1, x2, y2, conf])
                    detections_ar = np.vstack((detections_ar, curr_arr))
                else:
                    cv2.putText(final_img, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_PLAIN, 1.5, (200, 0, 100), 2)
                    cv2.rectangle(final_img, (x1, y1), (x2, y2), (200, 0, 100), 2)
            tracker_results = self.sort_tracker.update(detections_ar)

            for tr in tracker_results:
                tr = np.array(tr).astype(int).tolist()
                x1_s, y1_s, x2_s, y2_s, obj_id = tr
                if obj_id not in data.keys():
                    data[obj_id] = {"Status": False, "FramesCount": 0, "Time": 0.0, "StartTime": 0.0, "State": "",
                                    "Color": (0, 0, 0), "Area": "Unknown"}

                cen = self.get_center((x1_s, y1_s, x2_s, y2_s))
                data[obj_id]["Area"] = self.check_for_areas(cen)
                # cv2.circle(frame, cen, 5, (255, 0, 255), -1)

                check = self.check_for_owners(final_img, [x1_s, y1_s, x2_s, y2_s], detections)
                # If no owner then add to FramesCount (max count frame_count_thr + 1),
                # otherwise substract (max count -frame_count_thr - 1)
                if check:
                    data[obj_id]["FramesCount"] += frame_count_step
                else:
                    data[obj_id]["FramesCount"] -= frame_count_step

                if data[obj_id]["FramesCount"] > frame_count_thr:
                    data[obj_id]["FramesCount"] = frame_count_thr + 1
                    data[obj_id]["Status"] = True
                    data[obj_id]["Time"] = 0
                    data[obj_id]["StartTime"] = 0

                if data[obj_id]["FramesCount"] < -frame_count_thr:
                    data[obj_id]["FramesCount"] = -frame_count_thr - 1
                    if not data[obj_id]["StartTime"]:
                        data[obj_id]["StartTime"] = time()

                    data[obj_id]["Time"] = time()
                    data[obj_id]["Status"] = False

                label = f"Suitcase: {obj_id}"
                if data[obj_id]["Status"]:
                    data[obj_id]["Color"] = (0, 200, 0)
                else:
                    time_count = round(data[obj_id]["Time"] - data[obj_id]["StartTime"], 1)
                    if time_count:
                        label = f"{label} {time_count}s"
                        try:
                            state = Config.STATES[int(time_count)]
                            data[obj_id]["State"] = f' ({state["Name"]})'  # -1 if, my optimizations skills know no boundaries
                            data[obj_id]["Color"] = state["Color"]
                        except KeyError:
                            pass

                        label += f"{data[obj_id]['State']}"

                cv2.rectangle(final_img, (x1_s, y1_s), (x2_s, y2_s), data[obj_id]["Color"], 2)
                cv2.putText(final_img, label, (x1_s, y1_s - 10), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 255), 2)

            if not alert_sent_time and self.telegram_messages:
                msg = "Summary: \n"
                send = False
                for key, val in data.items():
                    if "Very Suspicious" in val["State"]:
                        msg += f"suitcase {key} in area {val['Area']} is sus \n"
                        send = True
                if send:
                    self.send_tg_message(msg=msg)
                    alert_sent_time = time()

            key = cv2.waitKey(1)
            if self.debug:
                # print(data)
                cv2.putText(final_img, "Debug Mode", (10, 80), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 2)
                cv2.putText(final_img, "Press 'f' to return False when owner checking", (10, 110), cv2.FONT_HERSHEY_PLAIN, 1.2, (255, 255, 255), 2)
                cv2.putText(final_img, "Press 'g' to return True when owner checking", (10, 140), cv2.FONT_HERSHEY_PLAIN, 1.2, (255, 255, 255), 2)
                cv2.imshow("res", frame_draw)

                if key == ord("f"):
                    self.switch = 1
                if key == ord("g"):
                    self.switch = 0

            c_time = time()
            fps = int(1 / (c_time - p_time))
            p_time = c_time

            cv2.putText(final_img, f"FPS: {fps}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imshow("frame", final_img)

            if key == 27:
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    sus = SusSuitcasesApp(
        model_path=rf"{Config.MODEL_FOLDER}\{Config.MODEL_NAME}",
        sort_max_ahe=Config.SORT_MAX_AGE,
        # areas_file="",
        areas_file=Config.AREAS_FILE,
        debug=Config.DEBUG,
        alert_interval=Config.ALERTS_INTERVAL,
        telegram_messages=Config.TELEGRAM_MESSAGES
    )
    sus.run()
