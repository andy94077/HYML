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

class FGSM:
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
		# 找出 gradient 的方向
		sign_data_grad = data_grad.sign()
		# 將圖片加上 gradient 方向乘上 epsilon 的 noise
		perturbed_image = image + epsilon * sign_data_grad
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
			
			loss = F.nll_loss(output, target)
			self.model.zero_grad()
			loss.backward()
			data_grad = data.grad.data
			perturbed_data = self.fgsm_attack(data, epsilon, data_grad)

			# 再將加入 noise 的圖片丟入 model 進行測試 得出相對應的 class        
			output = self.model(perturbed_data)
			final_pred = output.max(1, keepdim=True)[1]
		  
			if final_pred.item() == target.item(): # attack failed
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
	os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
	name = ['vgg16', 'vgg19', 'resnet50', 'resnet101', 'densenet121', 'densenet169']
	for model, n in zip([models.vgg16(pretrained=True), models.vgg19(pretrained=True), models.resnet50(pretrained=True), models.resnet101(pretrained=True), models.densenet121(pretrained=True), models.densenet169(pretrained=True)], name):
		attacker = FGSM(model, data_dir)
		eps = 0.1
		perturbed_imgs, acc = attacker.attack(eps)
		L_inf = L_infinity(attacker.dataset.X, perturbed_imgs)
		print(f'epsilon: {eps:.8e}, test acc: {acc:.5f}, L_infinity: {L_inf:.5f}')

		if not os.path.exists(output_dir+'_'+n):
			os.makedirs(output_dir+'_'+n)
		for i, img in enumerate(perturbed_imgs):
			Image.fromarray(img).save(os.path.join(output_dir+'_'+n, f'{i:03}.png'))

