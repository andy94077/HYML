import os, argparse
import pandas as pd
from PIL import Image
import numpy as np

import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms

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
		result = []
		wrong, fail, success = 0, 0, 0
		for (data, target) in self.loader:
			data, target = data.to(device), target.to(device)
			data_raw = data
			data.requires_grad = True

			output = self.model(data)
			init_pred = output.max(1, keepdim=True)[1]

			if init_pred.item() != target.item():
				wrong += 1
				data_raw = data_raw * torch.tensor(self.std, device=device).view(3, 1, 1) + torch.tensor(self.mean, device=device).view(3, 1, 1)
				data_raw = np.clip(data_raw.squeeze().detach().cpu().numpy() * 255, 0, 255).astype(np.uint8)
				result.append(data_raw)
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
				# perturbed_data = self.iterative_fgsm_attack(perturbed_data, epsilon, data_grad).detach()
				perturbed_data.requires_grad = True
				# eta = 0.1 * torch.sign(data_grad)
				# eta.clamp_(-epsilon, epsilon)

				# output = self.model(perturbed_data)
			output = self.model(perturbed_data)
			if output.max(1, keepdim=True)[1].item() == target.item():
				fail += 1
			else:
				success += 1 # attack succeed
			adv_ex = perturbed_data * torch.tensor(self.std, device=device).view(3, 1, 1) + torch.tensor(self.mean, device=device).view(3, 1, 1)
			adv_ex = np.clip(adv_ex.squeeze().detach().cpu().numpy() * 255, 0, 255).astype(np.uint8)
			result.append(adv_ex)
		final_acc = fail / len(self.loader)
		
		return np.array(result).transpose((0, 2, 3, 1)), final_acc

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
	parser.add_argument('output_dir')
	parser.add_argument('-g', '--gpu', default='0')
	args = parser.parse_args()

	data_dir = args.data_dir
	output_dir = args.output_dir
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)
	os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

	model = models.densenet121(pretrained=True)
	attacker = IterativeFGSM(model, data_dir)

	max_L_inf = 9.5
	best_perturbed_imgs = None
	best_eps, best_acc, best_L_inf = np.inf, 1., 0. # the lower best_acc is, the better the attack is
	T = 1
	left, right = 0, 2 * 1.46484375e-01
	try:
		for i in range(T):
			eps = (left + right) / 2
			perturbed_imgs, acc = attacker.attack(eps)
			L_inf = L_infinity(attacker.dataset.X, perturbed_imgs)
			print(f'epoch: {i + 1:02}, epsilon: {eps:.8e}, test acc: {acc:.5f}, L_infinity: {L_inf:.5f}')

			if L_inf <= max_L_inf:
				if L_inf > best_L_inf or L_inf == best_L_inf and acc < best_acc:
					best_eps, best_acc, best_L_inf = eps, acc, L_inf
					best_perturbed_imgs = perturbed_imgs.copy()
				left = eps
			else:
				right = eps
	except KeyboardInterrupt:
		pass

	if best_perturbed_imgs is not None:
		print(f'best epsilon: {best_eps:.8e}')
		print(f'L-infinity: {L_infinity(attacker.dataset.X, best_perturbed_imgs)}')
		for i, img in enumerate(best_perturbed_imgs):
			Image.fromarray(img).save(os.path.join(output_dir, f'{i:03}.png'))

