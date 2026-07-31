import torch


def random_derangement(video_lengths, device=None):
    """Randomly permute frames within each packed video."""
    lengths = video_lengths.tolist() if torch.is_tensor(video_lengths) else video_lengths
    permutations = []
    offset = 0
    for length in lengths:
        permutations.append(torch.randperm(length, device=device) + offset)
        offset += length
    return torch.cat(permutations)


if __name__ == "__main__":
    video_lengths = [5, 3, 4]
    permutations = random_derangement(video_lengths)
    print(permutations)
