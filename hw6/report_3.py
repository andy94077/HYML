import os, argparse
import pandas as pd
from PIL import Image
import numpy as np

import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dataset import Adverdataset

device = torch.device("cuda")

class IterativeFGSM:
	def __init__(self, model, data_dir):
		self.model = model
		self.model.cuda()
		self.model.eval()
		self.mean = [0.485, 0.456, 0.406]
		self.std = [0.229, 0.224, 0.225]

		self.normalize = transforms.Normalize(self.mean, self.std, inplace=False)
		transform = transforms.Compose([                
						transforms.Resize((224, 224), interpolation=3),
						transforms.ToTensor(),
						self.normalize
					])
		self.dataset = Adverdataset(data_dir, transform)
		print(f'\033[32;1mtrainX: {(len(self.dataset),)+self.dataset.X[0].size}, trainY: {self.dataset.label.shape}\033[0m')
		self.loader = torch.utils.data.DataLoader(self.dataset,	batch_size = 1, shuffle = False)

	def fgsm_attack(self, image, epsilon, data_grad):
		sign_data_grad = data_grad.sign()
		perturbed_image = image + epsilon * sign_data_grad
		return perturbed_image
	
	def iterative_fgsm_attack(self, image, epsilon, data_grad, lr=0.01):
		sign_data_grad = data_grad.sign()
		perturbed_image = image + (lr * sign_data_grad).clamp(-epsilon, epsilon)
		return perturbed_image
		

	def attack(self, epsilon):
		'''
		@return:
			list of [original_img (H, W, C), (original top_3_category value (3,), original top_3_category (3,)), (attacked top_3_category value (3,), attacked top_3_category (3,))]
		'''
		result = []
		wrong, fail, success = 0, 0, 0
		cnt = 0
		for (data, target) in self.loader:
			if cnt >= 5:
				break
			cnt += 1
			data, target = data.to(device), target.to(device)
			data_raw = data
			data.requires_grad = True

			output = self.model(data)
			init_pred = output.max(1, keepdim=True)[1]

			data_raw = data_raw * torch.tensor(self.std, device=device).view(3, 1, 1) + torch.tensor(self.mean, device=device).view(3, 1, 1)
			data_raw = np.clip(data_raw.squeeze().detach().cpu().numpy() * 255, 0, 255).astype(np.uint8)
			output_numpy = F.softmax(output).detach().cpu().numpy().ravel()
			top_3_idx = output_numpy.argsort()[:-4:-1]
			result.append([data_raw.transpose((1, 2, 0)), (output_numpy[top_3_idx], top_3_idx)])
			if init_pred.item() != target.item():
				wrong += 1
				output_numpy = F.softmax(output).detach().cpu().numpy().ravel()
				top_3_idx = output_numpy.argsort()[:-4:-1]
				result[-1].append((output_numpy[top_3_idx], top_3_idx))
				continue
			
			perturbed_data = data
			for _ in range(10): 
				self.model.zero_grad()
				output = self.model(perturbed_data)
				if output.max(1, keepdim=True)[1].item() != target.item(): # attack succeed
					break
				loss = F.cross_entropy(output, target)
				self.model.zero_grad()
				loss.backward()
				data_grad = perturbed_data.grad.detach()
				perturbed_data = self.fgsm_attack(perturbed_data, epsilon, data_grad).detach()

				perturbed_data.requires_grad = True
			output = self.model(perturbed_data)
			if output.max(1, keepdim=True)[1].item() == target.item():
				fail += 1
			else:
				success += 1 # attack succeed

			output_numpy = F.softmax(output).detach().cpu().numpy().ravel()
			top_3_idx = output_numpy.argsort()[:-4:-1]
			result[-1].append((output_numpy[top_3_idx], top_3_idx))
		
		return result

def L_infinity(X, perturbed_imgs):
	'''
	@param:
		X: list of PIL images
		perturbed_imgs: np.array of images, shape = (N, H, W, C)
	'''
	return np.mean(np.max(np.abs(np.array([np.array(x) for x in X], np.int32) - perturbed_imgs.astype(np.int32)), axis=(-1, -2, -3)))

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('data_dir')
	parser.add_argument('-g', '--gpu', default='0')
	args = parser.parse_args()

	data_dir = args.data_dir
	os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

	model = models.densenet121(pretrained=True)
	attacker = IterativeFGSM(model, data_dir)

	eps = 1.46484375e-01
	result = attacker.attack(eps)
	del result[1:3]

	df = pd.read_csv(os.path.join(data_dir, 'categories.csv'))
	fig, axs = plt.subplots(3, 3, figsize=(15, 15))
	for i, [img, ori, atk] in enumerate(result):
		axs[i, 0].imshow(img)
		for j, (top_3_value, top_3_idx) in enumerate([ori, atk]):
			axs[i, j + 1].bar(range(3), top_3_value, tick_label=[f'{name.split(",")[0]}({idx})' for name, idx in zip(df['CategoryName'][top_3_idx], top_3_idx)])
			axs[i, j+1].tick_params(labelsize=8)
			axs[i, j+1].set_title(['Original Image', 'Adversarial Image'][j])
			axs[i, j+1].set_xlabel(f'{df["CategoryName"][top_3_idx[0]].split(",")[0]} {top_3_value[0]*100:.2f}%')
	fig.savefig('report_3.jpg', bbox_inches='tight')

