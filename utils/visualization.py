import matplotlib.pyplot as plt
import numpy as np
import tensorflow_addons as tfa
from tensorflow.keras import backend as K
from skimage.color import label2rgb


def plot_compare_curve(train_losses,val_losses,train_scores,val_scores,plot_title=None):
    fig,axes = plt.subplots(1,2,figsize=(12,6))
    if plot_title is not None:
        fig.suptitle(plot_title)

    axes[0].plot(range(len(train_losses)), train_losses,label='train')
    axes[0].plot(range(len(val_losses)), val_losses,label='val')
    axes[0].set_ylabel('loss')
    axes[0].legend()

    axes[1].plot(range(len(train_scores)), train_scores,label='train')
    axes[1].plot(range(len(val_scores)), val_scores,label='val')
    axes[1].set_ylabel('dice score')
    axes[1].legend()

    fig.text(0.5, 0.04, 'epochs', ha='center')
    plt.show()


def visualize_segmentation(model,X,Y,num_samples=10,seed=None, subfig_size = 5):
    if seed is not None:
        np.random.seed(seed)
    metric = tfa.metrics.F1Score(1,average='micro',threshold=0.5)
    ridx = np.random.choice(X.shape[0],num_samples,False)
    im = X[ridx][:,:,:,0]
    mk = Y[ridx][:,:,:,0]
    pr = model.predict(X[ridx])[:,:,:,0]
    pr = np.array(K.round(pr))
    
    fig,axes = plt.subplots(
        nrows=num_samples, 
        ncols=3,
        figsize=(subfig_size*3,subfig_size*num_samples))

    if num_samples < 2:
        for ax, col in zip(axes, ['Img', 'Ground Truth', 'Predicted Mask']):
            ax.set_title(col)
    else:
        for ax, col in zip(axes[0], ['Img', 'Ground Truth', 'Predicted Mask']):
            ax.set_title(col)

    for i,(ax1,ax2,ax3) in enumerate(axes):
        ax1.imshow(im[i],cmap='gray')
        ax2.imshow(label2rgb(mk[i],im[i], colors=['red',], alpha=1, bg_label=0, bg_color=None))
        ax3.imshow(label2rgb(pr[i],im[i], colors=['red',], alpha=1, bg_label=0, bg_color=None))
        
    plt.show()
    

def visualize_multi_segmentation(models,X,Y,num_samples=10,model_names=[],seed=None, subfig_size = 4):
    if seed is not None:
        np.random.seed(seed)
    n_model = len(models)

    metric = tfa.metrics.F1Score(1,average='micro',threshold=0.5)
    ridx = np.random.choice(X.shape[0],num_samples,False)

    im = X[ridx][:,:,:,0]
    mk = Y[ridx][:,:,:,0]
    pr = [model.predict(X[ridx])[:,:,:,0] for model in models]
    pr = [np.array(K.round(i)) for i in pr]
    
    fig,axes = plt.subplots(
        nrows = num_samples,
        ncols = 1 + n_model, 
        figsize=(subfig_size*(1 + n_model), subfig_size*num_samples))

    if num_samples < 2:
        for ax, col in zip(axes, ['Ground Truth']+model_names):
            ax.set_title(col)
    else:
        for ax, col in zip(axes[0], ['Ground Truth']+model_names):
            ax.set_title(col)

    for i,ax in enumerate(axes):
        ax[0].imshow(label2rgb(mk[i],im[i],colors=['red',],alpha=1,bg_label=0,bg_color=None))
        for j in range(n_model):
            ax[j+1].imshow(label2rgb(pr[j][i],im[i],colors=['red',],alpha=1,bg_label=0,bg_color=None))
    plt.show()