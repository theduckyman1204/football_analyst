import cv2
from slugify import annotations
import torch 
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np
import pickle
import os

class Tracker: 
    def __init__(self, yolo_model_path, max_age=100, n_init=1, ball_conf_threshold=0.2):
        self.model = YOLO(yolo_model_path)
        self.deepsort = DeepSort(max_age=max_age, n_init=n_init)
        self.ball_conf_threshold = ball_conf_threshold
        self.tracks = {
            'players': {},
            'referees': {},
            'balls': {}
        }
        self.missed_ball_positions = []
        self.detections_file = 'detections.pkl'
        self.detections = self.load_detections()

    def load_detections(self):
        if os.path.exists(self.detections_file):
            with open(self.detections_file, 'rb') as f:
                print("Loaded existing detections.")
                return pickle.load(f)
        return []

    def save_detections(self, detections):
        with open(self.detections_file, 'wb') as f:
            pickle.dump(detections, f)
        print(f"Detections saved to {self.detections_file}")

    def interpolate_ball_position(self):
        if len(self.missed_ball_positions) < 2:
            return None
        last_pos = self.missed_ball_positions[-1]
        prev_pos = self.missed_ball_positions[-2]
        delta_x = last_pos[0] - prev_pos[0]
        delta_y = last_pos[1] - prev_pos[1]
        return (last_pos[0] + delta_x, last_pos[1] + delta_y)

    def process_frame(self, frame, frame_num, use_existing_detections=False, detections=None):
        track = {
            'players': {},
            'referees': {},
            'balls': {}
        }
        # Thao tác nếu có detection trước 
        if use_existing_detections and detections is not None:
            for det in detections:               
                bbox = det[0]
                cls = det[2]
                track_id = det[3] if len(det) > 3 else -1
                x, y, w, h = bbox
                if cls == 0:  # Ball class
                    track['balls'].setdefault(frame_num, {})[track_id] = bbox

                elif cls == 1 or cls == 2 :  # Player class
                    track['players'].setdefault(frame_num, {})[track_id] = bbox
                    color = (255, 0, 0)
                elif cls == 3:  # Referee class
                    track['referees'].setdefault(frame_num, {})[track_id] = bbox
                    color = (0, 0, 255)
            return frame, detections,track
        #Thao tác nếu chưa có detection 
        detections = []
        ball_detected = False
        results = self.model(frame)
        for i in results[0].boxes.data.cpu().numpy():  
            x1, y1, x2, y2, conf, cls = i
            track_id = int(conf * 1000)  # Placeholder for track_id
            if int(cls) == 0:  # Ball class is 0
                if conf >= self.ball_conf_threshold:
                    ball_detected = True
                    self.missed_ball_positions.append(((x1 + x2) // 2, (y1 + y2) // 2))
                else:
                    continue
            width = x2 - x1
            height = y2 - y1
            detections.append(([x1, y1, width, height], conf, int(cls), track_id))
        

        if not ball_detected and self.missed_ball_positions:
            interpolated_pos = self.interpolate_ball_position()
            if interpolated_pos:
                x, y = interpolated_pos
                detections.append(([x - 10, y - 10, 20, 20], 0.2, 0, 1))

        tracks = self.deepsort.update_tracks(detections, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            ltrb = track.to_ltrb()  
            x1, y1, x2, y2 = map(int, ltrb)
            width, height = x2 - x1, y2 - y1
            bbox = [x1, y1, width, height]
            cls = track.class_id if hasattr(track, 'class_id') else 0

            if cls == 0:  # Ball
                track['balls'].setdefault(frame_num, {})[track_id] = bbox
            elif cls == 1 or cls == 2:  # Player
                track['players'].setdefault(frame_num, {})[track_id] = bbox
            elif cls == 2:  # Referee
                track['referees'].setdefault(frame_num, {})[track_id] = bbox
            return frame, detections, track
    def merge_tracks(self, all_tracks):
        merged_tracks = {'players': {},
                        'referees': {},
                        'balls': {}
                        }
    
        for track in all_tracks:
            for object_type in ['players', 'referees', 'balls']:
                for frame_num, objs in track[object_type].items():
                    if frame_num not in merged_tracks[object_type]:
                        merged_tracks[object_type][frame_num] = {}
    
                    for player_id, player_data in objs.items():
                        if isinstance(player_data, list):  # Fix nếu player_data là list
                            player_data = {'bbox': player_data}
                        merged_tracks[object_type][frame_num][player_id] = player_data
    
        return merged_tracks
    

    def track_video(self, video_frames):
        output_frames = []
        all_tracks = []
        if self.detections:
            print("Using existing detections.")
            for frame_num, (frame, detections) in enumerate(zip(video_frames, self.detections)):
                processed_frame, _ ,track= self.process_frame(frame, frame_num, use_existing_detections=True, detections=detections) # type: ignore
                output_frames.append(processed_frame)
                all_tracks.append(track)

            tracks = self.merge_tracks(all_tracks)
            return output_frames,tracks
            
        all_detections = []
        for frame_num, frame in enumerate(video_frames): 
            processed_frame, detections,track = self.process_frame(frame, frame_num) # type: ignore
            output_frames.append(processed_frame)
            all_detections.append(detections)
            all_tracks.append(track)
        self.save_detections(all_detections)
        tracks = self.merge_tracks(all_tracks)

        return output_frames,tracks
    def draw_annotation(self,frame_num, frame, tracks):
        # Vòng lặp cho players
        for player_id, track in tracks['players'].get(frame_num).items():
            bbox = track.get('bbox')
            team = track.get('team')
            team_color = tuple(map(int, track.get('team_color')))
            control_ball = track.get('has_ball')
            if len(bbox) == 4 and control_ball == False:
                x1, y1, width, height = map(int, bbox)
                x2 = x1 + width
                y2 = y1 + height
                cv2.ellipse(
                frame,
                center=(x1 + width//2,y1+height),
                axes=(int(width), int(0.35*width)),
                angle=0.0,
                startAngle=-45,
                endAngle=235,
                color = team_color,
                thickness=5,
                lineType=cv2.LINE_4
                )
            if len(bbox) == 4 and control_ball == True: 
                x1, y1, width, height = map(int, bbox)
                x2 = x1 + width
                y2 = y1 + height
                cv2.ellipse(
                frame,
                center=(x1 + width//2,y1+height),
                axes=(int(width), int(0.35*width)),
                angle=0.0,
                startAngle=-45,
                endAngle=235,
                color = (0,0,255),
                thickness=5,
                lineType=cv2.LINE_4
                )

        # Vòng lặp cho referees
        for referee_id, track in tracks['referees'].get(frame_num, {}).items():
            
            bbox = track.get('bbox')

            if len(bbox) == 4:
                x1, y1, width, height = map(int, bbox)
                x2 = x1 + width
                y2 = y1 + height
                cv2.rectangle(frame, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=5)

        # Vòng lặp cho balls
        for ball_id, track in tracks['balls'].get(frame_num).items():
            bbox = track.get('bbox')

            if len(bbox) == 4:
                x1, y1, width, height = map(int, bbox)
                triangle_pts = np.array([
                    [x1 + 10 // 2, y1 + 10], 
                    [x1, y1 - 10 // 2], 
                    [x1 + 10, y1 - 10 // 2]
                ], np.int32)
                cv2.polylines(frame, [triangle_pts], isClosed=True, color=(0, 255, 0), thickness=5)
