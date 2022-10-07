import torch


def onehot_encode(label, device):
  
    eye = torch.eye(10, device=device)
    return eye[label].view(-1, 10, 1, 1) 

def concat_image_label(image, label, device):

    B, _, H, W = image.shape
    oh_label = onehot_encode(label, device)
    oh_label = oh_label.expand(B, 10, H, W)
    return torch.cat((image, oh_label), dim=1)


def concat_noise_label(noise, label, device):
  
    oh_label = onehot_encode(label, device)
    return torch.cat((noise, oh_label), dim=1)