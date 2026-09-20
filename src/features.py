def extract_features(hand_landmarks):
    wrist = hand_landmarks[0]

    features = []

    for landmark in hand_landmarks:
        relative_x = landmark.x - wrist.x
        relative_y = landmark.y - wrist.y
        relative_z = landmark.z - wrist.z

        features.extend([
            relative_x,
            relative_y,
            relative_z
        ])

    return features