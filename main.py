import cv2
import numpy as np
from utils import read_video, save_video
from trackers import Tracker
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
def merge_ball_tracks(tracks, fixed_ball_id=1):
    merged_ball_track = {}

    for frame_num, ball_tracks in tracks['balls'].items():
        for _, track in ball_tracks.items():
            merged_ball_track[frame_num] = {fixed_ball_id: track}

    tracks['balls'] = merged_ball_track
    return tracks
def main(): 
    # Đọc video
    video_frames = read_video(r"input_video\input.mp4")
    
    tracker = Tracker(r"models\bestv5.pt")
    video, tracks = tracker.track_video(video_frames)

    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], tracks['players'][0])

    for frame_num, player_track in tracks['players'].items(): 
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num], track['bbox'], player_id)
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

    # Assign Ball Aquisition
    player_assigner =PlayerBallAssigner()
    team_ball_control= []
    tracks = merge_ball_tracks(tracks, fixed_ball_id=1)
    for frame_num, player_track in tracks['players'].items():
        ball_bbox = tracks['balls'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)
        for player_id, track in player_track.items():
            track['has_ball'] = False
        if assigned_player != -1:
            print(frame_num)
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            team_ball_control.append(2 if frame_num ==0 else team_ball_control[-1])
    team_ball_control= np.array(team_ball_control)

    # Vẽ annotation lên video và lưu
    for i, frame in enumerate(video_frames):
        percent = np.count_nonzero( team_ball_control[:i]== 1)
        cv2.putText(
        frame,                          # Ảnh hoặc frame
        f'Team 1 : {round(100*percent/(i+1),2)}%',         # Văn bản
        (1500, 900),                      # Vị trí (x, y)
        cv2.FONT_HERSHEY_SIMPLEX,        # Font
        1,                               # Kích thước chữ
        (255, 255, 255),                     # Màu chữ (BGR) - xanh lá
        2,                               # Độ dày chữ
        cv2.LINE_AA                      # Kiểu chữ mượt
        )
        cv2.putText(
        frame,                          # Ảnh hoặc frame
        f'Team 1 : {round(100-(100*(percent/(i+1))),2)}%',         # Văn bản
        (1500, 1000),                      # Vị trí (x, y)
        cv2.FONT_HERSHEY_SIMPLEX,        # Font
        1,                               # Kích thước chữ
        (255, 255, 255),                     # Màu chữ (BGR) - xanh lá
        2,                               # Độ dày chữ
        cv2.LINE_AA                      # Kiểu chữ mượt
        )
        tracker.draw_annotation(i, frame, tracks)
    save_video(video, r"output_videos\output_video.avi")

if __name__ == "__main__": 
    main()
