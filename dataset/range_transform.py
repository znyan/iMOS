import numpy as np
import torchvision.transforms as transforms
import torchvision.transforms.functional as F

# im_mean = (124, 116, 104)

# im_normalization = transforms.Normalize(
#                     mean=[0.485, 0.456, 0.406],
#                     std=[0.229, 0.224, 0.225]
#                 )

# inv_im_trans = transforms.Normalize(
#                 mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
#                 std=[1/0.229, 1/0.224, 1/0.225])

im_mean = (128, 128, 128)

im_normalization = transforms.Normalize(
                    mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5]
                )

inv_im_trans = transforms.Normalize(
                mean=[-1.0, -1.0, -1.0],
                std=[2.0, 2.0, 2.0])

class SquarePad:
	def __call__(self, image):
		w, h = image.size
		max_wh = np.max([w, h])
		hp = int((max_wh - w) / 2)
		vp = int((max_wh - h) / 2)
		padding = (hp, vp, hp, vp)
		return F.pad(image, padding, 0, 'constant')

class Norm:
    def __call__(self, image):
        return (image - image.mean(dim=[1, 2]).unsqueeze(1).unsqueeze(1)) / (image.std(dim=[1, 2]).unsqueeze(1).unsqueeze(1) + 1e-5)