import torch
import argparse
from data_loader.channel_data import channel_dataset
from nets.FC_VAE import VAE
from train_and_test.training import train
from train_and_test.testing import test,test_latent_space,compute_mmd,auroc
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

from scipy.ndimage import gaussian_filter

def parse_args():
    parser = argparse.ArgumentParser(description='uai')
    parser.add_argument('--eps_tr', type=float, default=0, help='Contamination Ratio Training')
    parser.add_argument('--eps_te', type=float, default=0, help='Contamination Ratio Testing')
    parser.add_argument('--sigma', type=int, default=0.01, help='Likelihood Variance')
    parser.add_argument('--size_tr', type=int, default=10e2, help='size for training')
    parser.add_argument('--size_val', type=int, default=10e2, help='size for validation')
    parser.add_argument('--size_te', type=int, default=5*10e2, help='size for testing')
    parser.add_argument('--tr_batch_size', type=int, default=128, help='minibatchsize for training')
    parser.add_argument('--te_batch_size', type=int, default=128, help='minibatchsize for testing')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate for training')
    parser.add_argument('--total_epochs', type=int, default=200, help='total epochs for training')
    parser.add_argument('--m', type=int, default=1, help='number of multisample during training')
    parser.add_argument('--m_te', type=int, default=5, help='number of multisample for test')
    parser.add_argument('--t', type=float, default=1, help='t value for log-t')
    parser.add_argument('--beta', type=float, default=10., help='beta for KL term')
    parser.add_argument('--sigma_prior', type=float, default=1, help='prior for sigma')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--if_test', action='store_true', default=False, help='only testing')
    parser.add_argument('--num_bin', type=int, default=11, help='total number of bins for ECE')
    parser.add_argument('--BayesianDec', default=False, help='If True BNN for Dec')
    parser.add_argument('--BayesEnc', default=False, help='If True BNN for Enc')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device("cuda" if use_cuda else "cpu")
    return args

def main(args):
    MC_reps = 1
    torch.manual_seed(0)
    LOG_NLL,LOG_ACC,LOG_NLLT,TEST,MMD,AUROC = [], [], [], [], [],[]
    print('Called with args:')
    print(args)
    for rep in range(0, MC_reps):
        bnn = VAE(args.BayesEnc, args.BayesianDec, args.dim).to(args.device)
        train_dataset = channel_dataset(mode='train', seed=rep,snr=10,size=args.dim)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.tr_batch_size, shuffle=True, num_workers=0)
        val_dataset = channel_dataset(mode='test', seed=rep,snr=10,size=args.dim)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.te_batch_size, shuffle=True, num_workers=0)
        args.beta = 1. / (args.beta * len(train_dataset))
        if (not args.BayesianDec):
            args.m_te = 1
            args.beta = 0
        if args.if_test:
            if(args.BayesianDec):
                bnn = torch.load('saved_models/REP_' + str(rep) + '_temp_' + str(args.t) + '_m_' + str(args.m) + '_eps_' + str(args.eps_tr)+ '_dim_' + str(args.dim), map_location=torch.device('cpu'))
            else:
                bnn = torch.load('saved_models/REP_' + str(rep) + '_temp_' + str(args.t) + '_m_' + str(args.m) + '_eps_' + str(args.eps_tr)+ '_dim_' + str(args.dim)+'_freq', map_location=torch.device('cpu'))
            MMD_temp=0
            MMD_reps=1
            for MMD_rep in range(0,MMD_reps):
                test_dataset = channel_dataset(mode='test', seed=rep,snr=10,size=args.dim)
                test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.te_batch_size, shuffle=True, num_workers=0)
                test(args, bnn, test_loader)
                if (args.TEST):
                    TEST.append(test(args, bnn, test_loader))
                    test_latent_space(args, bnn, test_loader)
                if(args.AUROC):
                    AUROC.append(auroc(args, bnn, test_loader))
                if(args.MMD):
                    MMD_temp=MMD_temp+compute_mmd(args,bnn,test_loader)
                    MMD.append(MMD_temp/MMD_reps)
        else:
            # train
            test_nll, test_nllt, test_acc = train(args, bnn, train_loader, val_loader, rep)
            LOG_NLL.append(test_nll)
            LOG_NLLT.append(test_nllt)
            LOG_ACC.append(test_acc)
            if (args.BayesianDec):
                torch.save(bnn, 'saved_models/REP_' + str(rep) + '_temp_' + str(args.t) + '_m_' + str(args.m) + '_eps_' + str(args.eps_tr)+ '_dim_' + str(args.dim))
            else:
                torch.save(bnn, 'saved_models/REP_' + str(rep) + '_temp_' + str(args.t) + '_m_' + str(args.m) + '_eps_' + str(args.eps_tr) + '_dim_' + str(args.dim)+'_freq')
    if args.if_test:
        if (args.BayesianDec):
            if (args.TEST):
                np.save('logs/TEST' + str(args.t) + '_m_' + str(args.m) + '_eps_' + str(args.eps_tr) + '.npy', TEST)
            if (args.AUROC):
                np.save('logs/AUROC_' + str(args.t) + '_m_' + str(args.m) + '_eps_' + str(args.eps_tr) + '.npy', AUROC)
            if (args.MMD):
                np.save('logs/MMD_' + str(args.t) + '_m_' + str(args.m) + '_eps_' + str(args.eps_tr) + '.npy', MMD)
        else:
            if (args.TEST):
                np.save('logs/TEST' + str(args.t) + '_m_' + str(args.m) + '_eps_' + str(args.eps_tr) + '_freq.npy', TEST)
            if (args.AUROC):
                np.save('logs/AUROC_' + str(args.t) + '_m_' + str(args.m) + '_eps_' + str(args.eps_tr) + '_freq.npy', AUROC)
            if (args.MMD):
                np.save('logs/MMD_' + str(args.t) + '_m_' + str(args.m) + '_eps_' + str(args.eps_tr) + '_freq.npy', MMD)
    else:
        if(args.BayesianDec):
            np.save('logs/LOG_' + str(args.t) + '_eps_' + str(args.eps_tr) + '_m_' + str(args.m)+ '_dim_' + str(args.dim)+'_.npy', [LOG_NLL, LOG_NLLT, LOG_ACC])
        else:
            np.save('logs/LOG_' + str(args.t) + '_eps_' + str(args.eps_tr) + '_m_' + str(args.m) + '_dim_' + str(args.dim) + '_freq.npy', [LOG_NLL, LOG_NLLT, LOG_ACC])

if __name__ == '__main__':
    args = parse_args()
    main(args)


