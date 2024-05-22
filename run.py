import argparse
import os
import torch
from exp.exp_main import Exp_Main
import random
import numpy as np
from utils.str2bool import str2bool



fix_seed = 2023
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

parser = argparse.ArgumentParser(description=
    'Models for Long-range Time Series Forecasting')

# basic config
parser.add_argument('--is_training', type=int, default=1, help='status')
parser.add_argument('--task_id', type=str, default='test', help='task id')
parser.add_argument('--model', type=str, default='TPGN',
    help='model name, options: [TPGN, iTransformer, TimeMixer, FITS, ModernTCN, PDF, \
    WITRAN, CrossGNN, FourierGNN, Basisformer, \
    MICN, TimesNet, PatchTST, DLinear, NLinear, Linear, \
    FiLM, FEDformer, Pyraformer, Autoformer, Informer, Transformer]')

# supplementary config for FEDformer model
parser.add_argument('--version', type=str, default='Fourier',
                    help='for FEDformer, there are two versions to choose, options: [Fourier, Wavelets]')
parser.add_argument('--mode_select', type=str, default='random',
                    help='for FEDformer, there are two mode selection method, options: [random, low]')
parser.add_argument('--modes', type=int, default=64, help='modes to be selected random 64')
parser.add_argument('--L', type=int, default=3, help='ignore level')
parser.add_argument('--base', type=str, default='legendre', help='mwt base')
parser.add_argument('--cross_activation', type=str, default='tanh',
                    help='mwt cross atention activation function tanh or softmax')

# data loader
parser.add_argument('--data', type=str, default='electricity', help='dataset type')
parser.add_argument('--root_path', type=str, default='../Datasets/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='electricity.csv', help='data file')
parser.add_argument('--features', type=str, default='S',
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, '
                            'S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, '
                            'b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

# forecasting task
parser.add_argument('--seq_len', type=int, default=168, help='input sequence length')
parser.add_argument('--label_len', type=int, default=0, help='start token length, no use for TPGMNN')
parser.add_argument('--pred_len', type=int, default=168, help='prediction sequence length')

# model define
parser.add_argument('--bucket_size', type=int, default=4, help='for Reformer')
parser.add_argument('--n_hashes', type=int, default=4, help='for Reformer')
parser.add_argument('--enc_in', type=int, default=1, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=1, help='decoder input size')
parser.add_argument('--c_out', type=int, default=1, help='output size')
parser.add_argument('--d_model', type=int, default=32, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=0, help='num of heads, no use for TPGMNN')
parser.add_argument('--e_layers', type=int, default=0, help='num of encoder layers, no use for TPGMNN')
parser.add_argument('--d_layers', type=int, default=0, help='num of decoder layers, , no use for TPGMNN')
parser.add_argument('--d_ff', type=int, default=0, help='dimension of fcn, no use for TPGMNN')
parser.add_argument('--moving_avg', default=25, help='window size of moving average')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--distil', action='store_false',
                    help='whether to use distilling in encoder, using this argument means not using distilling',
                    default=True)
parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

# optimization
parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
parser.add_argument('--itr', type=int, default=5, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=25, help='train epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='mse', help='loss function')
parser.add_argument('--lradj', type=str, default='type4', help='adjust learning rate')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

# GPU  
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1', help='device ids of multi gpus')

# For Pyraformer
# supplementary config for Pyraformer model
parser.add_argument('--decoder', type=str, default='FC', help='Pyraformer decoder, options:[FC, attention]')
parser.add_argument('--d_bottleneck', type=int, default=128)
parser.add_argument('--window_size', type=str, default='[4, 4, 5]', help='The number of children of a parent node')
parser.add_argument('--inner_size', type=int, default=3, help='The number of ajacent nodes.')
parser.add_argument('--CSCM', type=str, default='Bottleneck_Construct', 
    help='CSCM structure, options:[Bottleneck_Construct, Conv_Construct, MaxPooling_Construct, AvgPooling_Construct]')
parser.add_argument('--truncate', action='store_true', default=False, 
    help='Whether to remove coarse-scale nodes from the attention structure')
parser.add_argument('--use_tvm', action='store_true', default=False, help='Whether to use TVM')
parser.add_argument('--embed_type', type=str, default='DataEmbedding', 
    help='Embedding type of different dataset, default for features:S is DataEmbedding, others is CustomEmbedding')

# For FiLM
parser.add_argument('--modes1', type=int, default=64, help='modes to be 64')
parser.add_argument('--ab', type=int, default=2, help='ablation version')
parser.add_argument('--wavelet', type=int, default=0, help='use wavelet')
parser.add_argument('--add_noise_vali',type=bool,default=False,help='add noise in vali')
parser.add_argument('--add_noise_train',type=bool,default=False,help='add noise in training')
parser.add_argument('--ours', default=True, action='store_true')
parser.add_argument('--version1', type=int, default=0, help='compression')
parser.add_argument('--seasonal',type=int,default=7)
parser.add_argument('--mode_type',type=int,default=0)

# For PatchTST
parser.add_argument('--fc_dropout', type=float, default=0.05, help='fully connected dropout')
parser.add_argument('--head_dropout', type=float, default=0.0, help='head dropout')
parser.add_argument('--patch_len', type=int, default=16, help='patch length')
parser.add_argument('--stride', type=int, default=8, help='stride')
parser.add_argument('--padding_patch', default='end', help='None: None; end: padding on the end')
parser.add_argument('--revin', type=int, default=1, help='RevIN; True 1 False 0')
parser.add_argument('--affine', type=int, default=0, help='RevIN-affine; True 1 False 0')
parser.add_argument('--subtract_last', type=int, default=0, help='0: subtract mean; 1: subtract last')
parser.add_argument('--decomposition', type=int, default=0, help='decomposition; True 1 False 0')
parser.add_argument('--kernel_size', type=int, default=25, help='decomposition-kernel')
parser.add_argument('--individual', type=int, default=0, help='individual head; True 1 False 0')

# For TimesNet
parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')

# For MICN
parser.add_argument('--mode', type=str, default='regre', 
    help='different mode of trend prediction block: [regre or mean]')
parser.add_argument('--conv_kernel', type=int, nargs='+', default=[12,16,24], 
    help='downsampling and upsampling convolution kernel_size')
parser.add_argument('--decomp_kernel', type=int, nargs='+', default=[12,16,24], 
    help='decomposition kernel_size')
parser.add_argument('--isometric_kernel', type=int, nargs='+', default=[12,16,24], 
    help='isometric convolution kernel_size')
parser.add_argument('--padding', type=int, default=0, help='padding type')

# For Crossformer
parser.add_argument('--win_size', type=int, default=2, help='window size for segment merge')
parser.add_argument('--baseline', action='store_true', 
    help='whether to use mean of past series as baseline for prediction', default=False)
parser.add_argument('--cross_factor', type=int, default=10, 
    help='num of routers in Cross-Dimension Stage of TSA (c)')
parser.add_argument('--seg_len', type=int, default='12', help='segment length (L_seg)')

# For Basisformer
parser.add_argument('--N', type=int, default=10, help='number of learnable basis')
parser.add_argument('--block_nums', type=int, default=2, help='number of blocks')
parser.add_argument('--bottleneck', type=int, default=2, help='reduction of bottleneck')
parser.add_argument('--map_bottleneck', type=int, default=20, help='reduction of mapping bottleneck')

# For CrossGNN
parser.add_argument('--blocks', type=int, default=1, help='gpu')
parser.add_argument('--tvechidden', type=int, default=50, help='gpu')
parser.add_argument('--nvechidden', type=int, default=20, help='gpu')
parser.add_argument('--use_tgcn', type=int, default=1, help='gpu')
parser.add_argument('--use_ngcn', type=int, default=1, help='gpu')
parser.add_argument('--anti_ood', type=int, default=1, help='gpu')
parser.add_argument('--scale_number', type=int, default=4, help='gpu')
parser.add_argument('--tk', type=int, default=10, help='gpu')

# For WITRAN
parser.add_argument('--WITRAN_deal', type=str, default='None', 
    help='WITRAN deal data type, options:[None, standard]')
parser.add_argument('--WITRAN_grid_cols', type=int, default=24, 
    help='Numbers of data grid cols for WITRAN')

# For iTransformer
parser.add_argument('--use_norm', type=int, default=True, help='use norm and denorm')

# For FITS
parser.add_argument('--cut_freq', type=int, default=0)
parser.add_argument('--base_T', type=int, default=24)
parser.add_argument('--H_order', type=int, default=6)

# For PDF
parser.add_argument('--add', action='store_true', default=False, help='add')
parser.add_argument('--wo_conv', action='store_true', default=False, help='without convolution')
parser.add_argument('--serial_conv', action='store_true', default=False, help='serial convolution')
parser.add_argument('--PDF_kernel_list', type=int, nargs='+', default=[3, 7, 9], help='kernel size list')
parser.add_argument('--PDF_patch_len', type=int, nargs='+', default=[16, 16, 16], help='patch high')
parser.add_argument('--PDF_period', type=int, nargs='+', default=[12, 24], help='period list')
parser.add_argument('--PDF_stride', type=int, nargs='+', default=None, help='stride')

# For TimeMixer
parser.add_argument('--only_use_down_sampling', type=bool, default=False, help='only use down sampling')
parser.add_argument('--pred_down_sampling', type=bool, default=False, help='whether to down sampling in pred seq')
parser.add_argument('--down_sampling_layers', type=int, default=3, help='num of down sampling layers')
parser.add_argument('--down_sampling_window', type=int, default=2, help='down sampling window size')
parser.add_argument('--down_sampling_method', type=str, default='avg', help='down sampling method')
parser.add_argument('--channel_independent', type=int, default=1, help='whether to channel independent; True 1 False 0')

# For ModernTCN
parser.add_argument('--stem_ratio', type=int, default=6, help='stem ratio')
parser.add_argument('--downsample_ratio', type=int, default=2, help='downsample_ratio')
parser.add_argument('--ffn_ratio', type=int, default=8, help='ffn_ratio')

parser.add_argument('--patch_size', type=int, default=8, help='the patch size')
parser.add_argument('--patch_stride', type=int, default=4, help='the patch stride')

parser.add_argument('--large_size', nargs='+',type=int, default=51, help='big kernel size')
parser.add_argument('--small_size', nargs='+',type=int, default=5, help='small kernel size for structral reparam')

parser.add_argument('--small_kernel_merged', type=str2bool, default=False, help='small_kernel has already merged or not')
parser.add_argument('--call_structural_reparam', type=bool, default=False, help='structural_reparam after training')
parser.add_argument('--use_multi_scale', type=str2bool, default=True, help='use_multi_scale fusion')

# For TPGN
parser.add_argument('--TPGN_period', type=int, default=24, help='TPGN_period')
parser.add_argument('--norm', type=int, default=0, help='norm')

args = parser.parse_args()

# Pyraformer parameters
args.window_size = eval(args.window_size)

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.dvices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

print('Args in experiment:')
print(args)

Exp = Exp_Main

if args.model == 'MICN':
    model_flag = args.model + '-' + args.mode
    decomp_kernel = []  # kernel of decomposition operation 
    isometric_kernel = []  # kernel of isometric convolution
    for ii in args.conv_kernel:
        if ii%2 == 0:   # the kernel of decomposition operation must be odd
            decomp_kernel.append(ii+1)
            isometric_kernel.append((args.seq_len + args.pred_len+ii) // ii) 
        else:
            decomp_kernel.append(ii)
            isometric_kernel.append((args.seq_len + args.pred_len+ii-1) // ii) 
    args.isometric_kernel = isometric_kernel  # kernel of isometric convolution
    args.decomp_kernel = decomp_kernel   # kernel of decomposition operation 
    print("isometric_kernel", isometric_kernel)
    print("decomp_kernel", decomp_kernel)
else:
    model_flag = args.model
print(model_flag)

eval_loss = []
test_loss = []

if args.is_training:
    for ii in range(args.itr):
        # setting record of experiments
        setting = '{}_{}_{}_modes{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
            args.task_id,
            model_flag,
            args.mode_select,
            args.modes,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.factor,
            args.embed,
            args.distil,
            args.des,
            ii)
        
        exp = Exp(args)  # set experiments
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        _, epoch_time_all_avg = exp.train(setting)

        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, epoch_time_all_avg)

        if args.do_predict:
            print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.predict(setting, True)

        
        torch.cuda.empty_cache()
        
else:
    ii = 0
    setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(args.model_id,
                                                                                                    model_flag,
                                                                                                    args.data,
                                                                                                    args.features,
                                                                                                    args.seq_len,
                                                                                                    args.label_len,
                                                                                                    args.pred_len,
                                                                                                    args.d_model,
                                                                                                    args.n_heads,
                                                                                                    args.e_layers,
                                                                                                    args.d_layers,
                                                                                                    args.d_ff,
                                                                                                    args.factor,
                                                                                                    args.embed,
                                                                                                    args.distil,
                                                                                                    args.des, ii)
    
    exp = Exp(args)  # set experiments
    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    exp.test(setting, test=1)
    
    torch.cuda.empty_cache()