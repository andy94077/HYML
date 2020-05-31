import os, argparse
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def draw(title=None, xlabel=None, ylabel=None, has_legend=False,savefig=False):
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
        plt.savefig(savefig)
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_dir')
    parser.add_argument('output_image')

    args = parser.parse_args()

    model_dir = args.model_dir.rstrip('/')
    output_image = args.output_image

    df = pd.read_csv(os.path.join(model_dir, 'logs.csv'))
    for col in ['G_loss', 'D_loss']:
        plt.plot(df['step'], df[col], label=col)
    draw(title=os.path.basename(model_dir), xlabel='steps', has_legend=True, savefig=output_image)