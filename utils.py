import numpy as np
import torch
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
import seaborn as sns
def load_normalize_data(data_dir,train_batch_size=128,test_batch_size=128):
    
    np.random.seed(1)
    _DATA_DIRECTORY_PATH = "./datasets/" + data_dir
    _DATA_FILE = _DATA_DIRECTORY_PATH + "/data.txt"

    if data_dir =="year_prediction_msd":
        raw_data=np.loadtxt(_DATA_FILE, delimiter=',', usecols=range(90))
        index_features = np.arange(1,90)
        index_target = np.array(0)
    else:
        raw_data = np.loadtxt(_DATA_FILE)
        _INDEX_FEATURES_FILE = _DATA_DIRECTORY_PATH + "/index_features.txt"
        _INDEX_TARGET_FILE = _DATA_DIRECTORY_PATH + "/index_target.txt"
        index_features = np.loadtxt(_INDEX_FEATURES_FILE)
        index_target = np.loadtxt(_INDEX_TARGET_FILE)

    raw_data = raw_data[np.random.permutation(np.arange(len(raw_data)))]
    raw_data = raw_data.astype(np.float32)

    X = raw_data[ : , [int(i) for i in index_features.tolist()] ]
    y = raw_data[ : , [int(index_target.tolist())]]

    end_train = int(0.9 * raw_data.shape[0])
    X_train,y_train = X[0:end_train, :], y[0:end_train]
    X_test, y_test = X[end_train:, : ], y[end_train: ]

    X_means, X_stds = X_train.mean(axis = 0), X_train.var(axis = 0)**0.5
    y_means, y_stds = y_train.mean(axis = 0), y_train.var(axis = 0)**0.5

    X_train = (X_train - X_means)/X_stds
    y_train = (y_train - y_means)/y_stds
    X_test = (X_test - X_means)/X_stds
    y_test = (y_test - y_means)/y_stds
    
    from torch.utils import data
    class Dataset(data.Dataset):

        def __init__(self, features, target):
            self.features = features
            self.target = target

        def __len__(self):
            return len(self.target)

        def __getitem__(self, index):
            X = self.features[index,:]
            y = self.target[index]
            return X, y

    train_dataset = Dataset(X_train, y_train)
    test_dataset = Dataset(X_test, y_test)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=train_batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=test_batch_size,shuffle=False)

    return train_loader, test_loader, y_stds


def train_dropout(model,train_loader, test_loader,epochs,num_samples,gamma, log_every, save_name, model_name):
    from IPython.display import clear_output
    from matplotlib.ticker import FuncFormatter
    loss_history=[]
    train_rmse_history, train_PICP_history, train_MPIW_history=[],[],[]
    test_rmse_history, test_PICP_history, test_MPIW_history=[],[],[]
    for epoch in trange(epochs):

        if (epoch % log_every == 0 ) or (epoch==0):
            train_rmse, train_PICP, train_MPIW,_,_,_,_,_= model.evaluate(train_loader, num_samples, gamma)
            test_rmse, test_PICP, test_MPIW,_,_,_,_,_= model.evaluate(test_loader, num_samples, gamma)
            train_rmse_history.append(train_rmse)
            train_PICP_history.append(train_PICP)
            train_MPIW_history.append(train_MPIW)

            test_rmse_history.append(test_rmse)
            test_PICP_history.append(test_PICP)
            test_MPIW_history.append(test_MPIW)
        for batch_idx, (X_train, y_train) in enumerate(train_loader):
            loss = model.fit(X_train,y_train)
            loss_history.append(loss)



        if (epoch % log_every==0) or (epoch==0):    
            clear_output(True)
            fig = plt.figure(figsize=[16,8])
            sns.set()
            sns.set_style("darkgrid")


            plt.subplot(2,2,1)
            plt.plot(test_rmse_history, label='Test RMSE: %.3f'%test_rmse)
            plt.plot(train_rmse_history, label='Train RMSE: %.3f'%train_rmse)
            plt.title("Root Mean Square Error")
            plt.xlabel('Epochs')
            plt.ylabel('RMSE')
            plt.legend(loc=0)
            plt.gca().get_xaxis().set_major_formatter(FuncFormatter(lambda x, p: format(int(x*log_every), ',')))

            plt.subplot(2,2,2)
            plt.plot(test_PICP_history, label='Test PICP: %.3f'%test_PICP)
            plt.plot(train_PICP_history, label='Train PICP: %.3f'%train_PICP)
            plt.title("Prediction Interval Coverage Probability")
            plt.xlabel('Epochs')
            plt.ylabel('PICP')
            plt.ylim(0.7,1.04)
            plt.legend(loc=0)
            plt.gca().get_xaxis().set_major_formatter(FuncFormatter(lambda x, p: format(int(x*log_every), ',')))


            plt.subplot(2,2,3)
            plt.plot(test_MPIW_history, label='Test MPIW: %.3f'%test_MPIW)
            plt.plot(train_MPIW_history, label='Train MPIW: %.3f'%train_MPIW)
            plt.xlabel('Epochs')
            plt.ylabel('MPIW')
            plt.legend(loc=0)
            plt.gca().get_xaxis().set_major_formatter(FuncFormatter(lambda x, p: format(int(x*log_every), ',')))


            plt.subplot(2,2,4)
            plt.plot(loss_history, label='Total Loss: %.3f'%loss)
            plt.legend() 
            plt.xlabel('Epochs')
            plt.ylabel('Total Loss')
            plt.legend(loc=0)
            plt.gca().get_xaxis().set_major_formatter(FuncFormatter(lambda x, p: format(int(x/len(train_loader)), ',')))
            plt.show()


    fig.savefig('figures/%s_%s.pdf'%(model_name, save_name), bbox_inches='tight')
    
    
def evaluate_model(model, test_loader,num_samples,gamma, save_name, model_name):
    
    rmse, PICP, MPIW, mean,pred_U, pred_L, features, target= model.evaluate(test_loader, num_samples, gamma)
    print(' RMSE = %.3f, PICP = %.3f, MPIW = %.3f' %(rmse, PICP, MPIW))
    
    n = len(target)
    if n > 50:
        n=50
        
    mean= mean.detach().numpy()[-n:]
    pred_U=pred_U.detach().numpy()[-n:]
    pred_L=pred_L .detach().numpy()[-n:]
    features=features.detach().numpy()[-n:]
    target=target.detach().numpy()[-n:]
    
    fig1, ax = plt.subplots(1)
    idx = np.argsort(target,axis=0)[:,0]

    ax.plot(np.arange(0,target.shape[0]), pred_U[idx,0],
           c='k',alpha=1.0,linewidth=0.7, label='Predicted Upper Bound')

    ax.plot(np.arange(0,target.shape[0]), pred_L[idx,0],
           c='k',alpha=1.0,linewidth=0.7, label='Predicted Lower Bound')
    ax.fill_between(np.arange(0,target.shape[0]),
                    pred_U[idx,0],
                   pred_L[idx,0],
                   color='0.9', alpha=1.0)
    ax.scatter(np.arange(0,target.shape[0]), target[idx,0], 
               c='b', s=10.0,marker='o',label='Target (True Value)')
    ax.scatter(np.arange(0,target.shape[0]), mean[idx,0],
               c='r', s=10.0,marker='X',label='Point Prediction ')
    ax.errorbar(np.arange(0,target.shape[0]),target[idx,0],
                yerr=[target[idx,0]-pred_L[idx,0], pred_U[idx,0]-target[idx,0]],
               ecolor='r', alpha=0.6, elinewidth=1.0,capzise=2.0,
               capthick=1., fmt='none',label='Prediction Interval')
    ax.set_xlabel('Samples ordered by y values')
    ax.set_ylabel('Y normalised')
    ax.legend(loc='upper left')
    ax.set_title('Dataset: %s {RMSE = %.3f, PICP = %.3f, MPIW = %.3f}' %(save_name,rmse, PICP, MPIW))
    sns.set()
    sns.set_style("darkgrid")
    sns.set_context("notebook", font_scale=1, 
                    rc={"lines.linewidth": 2,"font.zise":12, 
                        "text.color":"black","font.weight":'bold'})
    fig1.set_size_inches(16,4)
    fig1.savefig('figures/%s_%s_pi.pdf'%(model_name, save_name), bbox_inches='tight')
    
    num_vars = features.shape[-1]
    if num_vars >12:
        num_vars=12
    elif num_vars %2 != 0:
        num_vars-=1
    fig = plt.figure(figsize=[16,num_vars*2])
    sns.set()
    sns.set_style("darkgrid")
    sns.set_context("notebook", font_scale=1, 
                    rc={"lines.linewidth": 2,"font.zise":12, 
                        "text.color":"black","font.weight":'bold'})
    for var_plot in range(num_vars):
        plt.subplot(num_vars/2,2,var_plot+1)
        plt.scatter(features[:,var_plot], target[:,0],
                   c='b', s=10.0,marker='o',label='Target (True Value)')
        plt.scatter(features[:,var_plot], mean[:,0],
                   c='r', s=10.0,marker='X',label='Point Prediction ')
        plt.errorbar(features[:,var_plot],target[:,0],
                    yerr=[target[:,0]-pred_L[:,0], pred_U[:,0]-target[:,0]],
                   ecolor='r', alpha=0.6, elinewidth=1.0,capzise=2.0,
                   capthick=1., fmt='none',label='Prediction Interval')
        plt.xlabel('Input feature: %d'%(var_plot))
        plt.ylabel('Y normalised')
        plt.legend(loc=0)
#         plt.title('Dataset: %s {RMSE = %.3f, PICP = %.3f, MPIW = %.3f}' %(save_name,rmse, PICP, MPIW))
 

    fig.savefig('figures/%s_%s_pi_err_bar__var.pdf'%(model_name, save_name), bbox_inches='tight')

    
    
def train_Bayes(model,train_loader, test_loader,epochs,num_samples,gamma, log_every, save_name, model_name, qd=False):
    from IPython.display import clear_output
    from matplotlib.ticker import FuncFormatter
    loss_history, log_prior_history, log_variational_posterior_history, negative_log_likelihood_history=[],[],[],[]
    train_rmse_history, train_PICP_history, train_MPIW_history=[],[],[]
    test_rmse_history, test_PICP_history, test_MPIW_history=[],[],[]
    
    for epoch in trange(epochs):
        if (epoch % log_every == 0 ) or (epoch==0):
            train_rmse, train_PICP, train_MPIW,_,_,_,_,_= model.evaluate(train_loader, num_samples, gamma)
            test_rmse, test_PICP, test_MPIW,_,_,_,_,_= model.evaluate(test_loader, num_samples, gamma)
            train_rmse_history.append(train_rmse)
            train_PICP_history.append(train_PICP)
            train_MPIW_history.append(train_MPIW)

            test_rmse_history.append(test_rmse)
            test_PICP_history.append(test_PICP)
            test_MPIW_history.append(test_MPIW)
        for batch_idx, (X_train, y_train) in enumerate(train_loader):
            loss, log_prior, log_variational_posterior, negative_log_likelihood = model.fit(X_train,y_train, num_samples)
            loss_history.append(loss)
            log_prior_history.append(log_prior)
            log_variational_posterior_history.append(log_variational_posterior)
            negative_log_likelihood_history.append(negative_log_likelihood)


        if (epoch % log_every==0) or (epoch==0):    
            clear_output(True)
            fig = plt.figure(figsize=[16,8])
            sns.set()
            sns.set_style("darkgrid")


            plt.subplot(2,3,1)
            plt.plot(test_rmse_history, label='Test RMSE: %.3f'%test_rmse)
            plt.plot(train_rmse_history, label='Train RMSE: %.3f'%train_rmse)
            plt.title("Root Mean Square Error")
            plt.xlabel('Epochs')
            plt.ylabel('RMSE')
            plt.legend(loc=0)
            plt.gca().get_xaxis().set_major_formatter(FuncFormatter(lambda x, p: format(int(x*log_every), ',')))

            plt.subplot(2,3,2)
            plt.plot(test_PICP_history, label='Test PICP: %.3f'%test_PICP)
            plt.plot(train_PICP_history, label='Train PICP: %.3f'%train_PICP)
            plt.title("Prediction Interval Coverage Probability")
            plt.xlabel('Epochs')
            plt.ylabel('PICP')
            plt.ylim(0.7,1.04)
            plt.legend(loc=0)
            plt.gca().get_xaxis().set_major_formatter(FuncFormatter(lambda x, p: format(int(x*log_every), ',')))


            plt.subplot(2,3,3)
            plt.plot(test_MPIW_history, label='Test MPIW: %.3f'%test_MPIW)
            plt.plot(train_MPIW_history, label='Train MPIW: %.3f'%train_MPIW)
            plt.xlabel('Epochs')
            plt.ylabel('MPIW')
            plt.legend(loc=0)
            plt.gca().get_xaxis().set_major_formatter(FuncFormatter(lambda x, p: format(int(x*log_every), ',')))


            plt.subplot(2,3,4)
            plt.plot(loss_history, label='Total Loss: %.3f'%loss)
            plt.legend() 
            plt.xlabel('Epochs')
            plt.ylabel('Total Loss')
            plt.legend(loc=0)
            plt.gca().get_xaxis().set_major_formatter(FuncFormatter(lambda x, p: format(int(x/len(train_loader)), ',')))

            if qd :
                plt.subplot(2,3,5)
                plt.plot(negative_log_likelihood_history, label="quality_driven loss :%.3f"%negative_log_likelihood)
                plt.legend(loc=0)
                plt.xlabel('Epochs')
                plt.ylabel('QD loss')
                plt.gca().get_xaxis().set_major_formatter(FuncFormatter(lambda x, p: format(int(x/len(train_loader)), ',')))

            else:
                
                plt.subplot(2,3,5)
                plt.plot(negative_log_likelihood_history, label="Negative Log Likelihood:%.3f"%negative_log_likelihood)
                plt.legend(loc=0)
                plt.xlabel('Epochs')
                plt.ylabel('NLL')
                plt.gca().get_xaxis().set_major_formatter(FuncFormatter(lambda x, p: format(int(x/len(train_loader)), ',')))

            plt.subplot(2,3,6)
            plt.plot(log_variational_posterior_history, label="Log Variational Posterior:%.3f"%log_variational_posterior)
            plt.legend(loc=0)
            plt.xlabel('Epochs')
            plt.gca().get_xaxis().set_major_formatter(FuncFormatter(lambda x, p: format(int(x/len(train_loader)), ',')))
            plt.show()

    fig.savefig('figures/%s_%s.pdf'%(model_name, save_name), bbox_inches='tight')
