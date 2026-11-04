class SignalEvaluator:
    def __init__(self, config: dict, path_network):
        self.config = config
        self.path_network = path_network

    def evaluate_fall_risk(self, landing_box: list, footprint_pct: float):
        h = landing_box[3] - landing_box[1]
        fp_y1 = landing_box[3] - (h * footprint_pct)
        footprint_box = [[landing_box[0], fp_y1], [landing_box[2], landing_box[3]]]

        level, interior_overlap, strip_overlap, corridor_id = self.path_network.evaluate_footprint(footprint_box)
        return level, footprint_box, interior_overlap, strip_overlap, corridor_id

    def check_amr_approach(self, object_box: list, amr_tracks: dict, frame_width: int):
        for amr_id, track_history in amr_tracks.items():
            if len(track_history) < self.config['min_track_age_frames']:
                continue

            obj_cx = (object_box[0] + object_box[2]) / 2.0
            dists = [abs((t[0] + t[2]) / 2.0 - obj_cx) for t in track_history[-12:]]

            if len(dists) >= 3:
                is_decreasing_steps = sum(dists[i] > dists[i+1] for i in range(len(dists)-1))
                step_ratio = is_decreasing_steps / float(len(dists)-1)
                net_drop = (dists[0] - dists[-1]) / float(frame_width)

                if step_ratio >= self.config['distance_drop_step_pct'] and net_drop >= self.config['net_distance_drop_pct']:
                    return True, amr_id, dists[-1]

        return False, None, None