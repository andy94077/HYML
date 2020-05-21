import os, argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

def draw(title=None, xlabel=None, ylabel=None, has_legend=False, savefig=False):
	if title is not None:
		plt.title(title)
	if xlabel is not None:
		plt.xlabel(xlabel)
	if ylabel is not None:
		plt.ylabel(ylabel)
	if has_legend:
		plt.legend()
	if savefig is False:
		plt.show()
	else:
		plt.savefig(savefig, bbox_inches='tight')
	plt.close()

def plot(path):
	df = pd.read_csv(path)
	plt.plot()
parser = argparse.ArgumentParser()
parser.add_argument('model_path')
parser.add_argument('output', nargs=2, type=str, help='(train_loss and val_loss).jpg, bleu_score.jpg')
args = parser.parse_args()

df = pd.read_csv(os.path.join(args.model_path, 'logs.csv'))
plt.plot(df['steps']*2, df['train_loss'], label='train_loss')
plt.plot(df['steps']*2, df['val_loss'], label='val_loss')
draw(title='loss', xlabel='epoch', has_legend=True, savefig=args.output[0])

plt.plot(df['steps']*2, df['bleu_score'])
draw(title='BLEU@1 score', xlabel='epoch', savefig=args.output[1])
