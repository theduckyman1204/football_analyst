import sys 
sys.path.append('../')
from utils import get_center_of_bbox, measure_distance # type: ignore

class PlayerBallAssigner():
    def __init__(self):
        self.max_player_ball_distance = 70  # ngưỡng khoảng cách tối đa để gán bóng cho cầu thủ
    
    def assign_ball_to_player(self, players, ball_bbox):
        if not isinstance(players, dict):
            print("Error: 'players' should be a dictionary.")
            return None

        ball_position = get_center_of_bbox(ball_bbox)

        min_distance = 10000
        assigned_player = -1

        for player_id, player in players.items():
            player_bbox = player.get('bbox', [0, 0, 0, 0])
            x1, y1, w, h = map(int, player_bbox)
            x2, y2 = x1 + w, y1 + h

            # Tính khoảng cách từ hai chân cầu thủ đến vị trí quả bóng
            distance_left = measure_distance((x1, y2), ball_position)
            distance_right = measure_distance((x2, y2), ball_position)
            distance = min(distance_left, distance_right)

            # Kiểm tra khoảng cách có nhỏ hơn ngưỡng tối đa không
            if distance < self.max_player_ball_distance and distance < min_distance:
                min_distance = distance
                assigned_player = player_id

        return assigned_player
