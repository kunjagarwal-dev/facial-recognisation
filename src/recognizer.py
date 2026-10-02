import torch

def recognize_face(embedding, database, threshold=0.9):
    best_match = None
    best_distance = float("inf")

    for name, stored_embedding in database.items():
        distance = torch.dist(embedding, stored_embedding).item()
        if distance < best_distance:
            best_distance = distance
            best_match = name

    if best_distance < threshold:
        return best_match, best_distance
    else:
        return "Unknown", best_distance