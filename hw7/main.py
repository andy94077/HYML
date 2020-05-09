import os, argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import utils
from dataset import MyDataset
from StudentNet import StudentNet

def get_train_valid_dataset(data_dir):
	training_filenames, trainY = utils.load_train_filename_and_labels(data_dir)
	training_filenames, valid_filenames, trainY, validY = utils.train_test_split(training_filenames, trainY, split_ratio=0.1)
	
	trsfms = transforms.Compose([
				transforms.RandomCrop(256, pad_if_needed=True, padding_mode='symmetric'),
				transforms.RandomHorizontalFlip(),
				transforms.RandomRotation(15),
				transforms.ToTensor(),
				])
	return MyDataset(os.path.join(data_dir, 'training'), transforms=trsfms), MyDataset(os.path.join(data_dir, 'validation'), transforms=trsfms)
	#return MyDataset(filenames=training_filenames, labels=trainY, transforms=trsfms), MyDataset(filenames=valid_filenames, labels=validY, transforms=trsfms)

def knowledge_distillatin_loss(outputs, labels, teacher_outputs, T=20, alpha=0.5):
	hard_loss = F.cross_entropy(outputs, labels) * (1. - alpha)
	soft_loss = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(outputs / T, dim=1), F.softmax(teacher_outputs / T, dim=1)) * (alpha * T * T)
	return hard_loss + soft_loss

def run_epoch(optimizer, dataloader, teacher_net, student_net, update=True, alpha=0.5):
	total_num, total_hit, total_loss = 0, 0, 0
	for i, (inputs, hard_labels) in enumerate(dataloader):
		inputs = inputs.cuda()
		hard_labels = hard_labels.cuda()

		# 因為Teacher沒有要backprop，所以我們使用torch.no_grad
		# 告訴torch不要暫存中間值(去做backprop)以浪費記憶體空間
		with torch.no_grad():
			soft_labels = teacher_net(inputs)

		optimizer.zero_grad()
		if update:
			logits = student_net(inputs)
			loss = knowledge_distillatin_loss(logits, hard_labels, soft_labels, alpha=alpha)
			loss.backward()
			optimizer.step()
		else:
			# 只是算validation acc的話，就開no_grad節省空間
			with torch.no_grad():
				logits = student_net(inputs)
				loss = knowledge_distillatin_loss(logits, hard_labels, soft_labels, alpha=alpha)

		total_hit += torch.sum(torch.argmax(logits, dim=1) == hard_labels).item()
		total_num += len(inputs)

		total_loss += loss.item() * len(inputs)
	return total_loss / total_num, total_hit / total_num

def train(model_path, teacher_net, student_net, train_dataset, valid_dataset, epochs, batch_size):
	train_dataloader, valid_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True), DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

	optimizer = optim.AdamW(student_net.parameters(), lr=1e-3)

	teacher_net.eval()
	best_acc = 0
	for epoch in range(epochs):
		student_net.train()
		train_loss, train_acc = run_epoch(optimizer, train_dataloader, teacher_net, student_net, update=True)

		student_net.eval()
		valid_loss, valid_acc = run_epoch(optimizer, valid_dataloader, teacher_net, student_net, update=False)

		print(f'epoch: {epoch:03}, loss: {train_loss:6.4f}, acc: {train_acc:6.4f}, val_loss: {valid_loss:6.4f}, val_acc {valid_acc:6.4f}')
		if valid_acc > best_acc:
			print(f'val_acc improved from {best_acc:.4f} to {valid_acc:.4f}, saving model to {model_path}...\n')
			best_acc = valid_acc
			torch.save(student_net.state_dict(), model_path)

def predict(model, dataset, batch_size=256):
	trsfms = transforms.Compose([
				transforms.CenterCrop(256),
				transforms.ToTensor(),
				])
	dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
	model.eval()
	with torch.no_grad():
		y_pred = np.concatenate([model(x[0].cuda() if isinstance(x, tuple) else x.cuda()).detach().cpu().numpy() for i, x in enumerate(dataloader)], axis=0)
	return y_pred

def evaluate(model, dataset, batch_size=256):
	out = predict(model, dataset, batch_size)
	loss = F.cross_entropy(torch.tensor(out).cuda(), dataset.labels.cuda()).item()
	acc = np.mean(np.argmax(out, axis=1) == dataset.labels.numpy())
	return loss, acc

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('data_dir')
	parser.add_argument('teacher_model_path')
	parser.add_argument('model_path')
	parser.add_argument('-T', '--no-training', action='store_true')
	parser.add_argument('-s', '--test', nargs=2, type=str, help='predicted file')
	parser.add_argument('-e', '--ensemble', action='store_true', help='output npy file to ensemble later')
	parser.add_argument('-g', '--gpu', default='3')
	args = parser.parse_args()

	os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
	data_dir = args.data_dir
	teacher_model_path = args.teacher_model_path
	model_path = args.model_path
	training = not args.no_training
	test = args.test
	ensemble = args.ensemble

	student_net = StudentNet(base_channel_size=16).cuda()

	if training:
		train_dataset, valid_dataset = get_train_valid_dataset(data_dir)

		teacher_net = models.resnet18(pretrained=False, num_classes=11).cuda()
		teacher_net.load_state_dict(torch.load(teacher_model_path))

		train(model_path, teacher_net, student_net, train_dataset, valid_dataset, epochs=200, batch_size=128)
	else:
		print('\033[32;1mLoading Model\033[0m')
	
	student_net.load_state_dict(torch.load(model_path))
	if test:
		test_dataset = MyDataset(os.path.join(data_dir, 'testing'), transforms=trsfms)
		pred = predict(student_net, test_dataloader)
		if ensemble:
			np.save(test[1], pred)
		else:
			utils.generate_csv(pred, test[1])
	else:
		if not training:
			train_dataset, valid_dataset = get_train_valid_dataset(data_dir)
		print(f'\033[32;1mTraining score: {evaluate(student_net, train_dataset, batch_size=256)}\033[0m')
		print(f'\033[32;1mValidaiton score: {evaluate(student_net, valid_dataset, batch_size=256)}\033[0m')
		
