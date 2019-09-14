import torch

"""
    From a list of samples from the dataset,
    returns the batched images and targets.
    This should be passed to the DataLoader
"""


def BatchCollator(batch):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    images = []
    targets = []
    for elm in batch:
        images.append(elm[0].to(device))
        for i in range(len(elm[1]["boxes"])):
            elm[1]["boxes"][i] = elm[1]["boxes"][i].to(device)
            elm[1]["masks"][i] = elm[1]["masks"][i].to(device)
            elm[1]["labels"][i] = elm[1]["labels"][i].to(device)
        targets.append(elm[1])
    return images, targets
