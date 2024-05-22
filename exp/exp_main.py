import os
import time
import warnings
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import TPGN, iTransformer, FITS, PDF, TimeMixer, ModernTCN, \
    WITRAN, CrossGNN, FourierGNN, Basisformer, Crossformer, \
    MICN, TimesNet, PatchTST, DLinear, NLinear, Linear, \
    FiLM, Pyraformer, FEDformer, Autoformer, Informer, Reformer, Transformer
from utils.tools import EarlyStopping, adjust_learning_rate, visual, save_to_csv
from utils.metrics import metric

warnings.filterwarnings('ignore')


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'TPGN': TPGN,
            'iTransformer': iTransformer,
            'FITS': FITS,
            'PDF': PDF,
            'TimeMixer': TimeMixer,
            'ModernTCN': ModernTCN,
            'WITRAN': WITRAN,
            'CrossGNN': CrossGNN,
            'FourierGNN': FourierGNN,
            'Basisformer': Basisformer,
            'Crossformer': Crossformer,
            'MICN': MICN,
            'TimesNet': TimesNet,
            'PatchTST': PatchTST,
            'DLinear': DLinear,
            'NLinear': NLinear,
            'Linear': Linear,
            'FiLM': FiLM,
            'Pyraformer': Pyraformer,
            'FEDformer': FEDformer,
            'Autoformer': Autoformer,
            'Informer': Informer,
            'Reformer': Reformer,
            'Transformer': Transformer,
        }
        if self.args.model=='MICN':
            e_layers = self.args.e_layers
            model = model_dict[self.args.model].MICN(
                self.args.dec_in,
                self.args.c_out, 
                self.args.seq_len, 
                self.args.label_len,
                self.args.pred_len,
                self.args.d_model, 
                self.args.n_heads,
                self.args.d_layers,
                self.args.dropout,
                self.args.embed,
                self.args.freq,
                self.device,
                self.args.mode,
                self.args.decomp_kernel,
                self.args.conv_kernel,
                self.args.isometric_kernel,
            ).float()
        else:
            model = model_dict[self.args.model].Model(self.args).float()
        print('###### generator parameters:', sum(param.numel() for param in model.parameters()))

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def __multi_scale_process_inputs(self, batch_x, batch_x_mark, batch_y, batch_y_mark):
        if self.args.down_sampling_method == 'max':
            down_pool = torch.nn.MaxPool1d(self.args.down_sampling_window, return_indices=False)

        elif self.args.down_sampling_method == 'avg':
            down_pool = torch.nn.AvgPool1d(self.args.down_sampling_window)
        else:
            return batch_x, batch_x_mark, batch_y, batch_y_mark
        # B,T,C -> B,C,T
        batch_x = batch_x.permute(0, 2, 1)
        batch_y = batch_y.permute(0, 2, 1)

        batch_x_ori = batch_x
        batch_y_ori = batch_y
        batch_x_mark_ori = batch_x_mark
        batch_y_mark_ori = batch_y_mark

        batch_x_sampling_list = []
        batch_y_sampling_list = []
        batch_x_mark_list = []
        batch_y_mark_list = []
        batch_x_sampling_list.append(batch_x.permute(0, 2, 1))
        batch_y_sampling_list.append(batch_y.permute(0, 2, 1))
        batch_x_mark_list.append(batch_x_mark)
        batch_y_mark_list.append(batch_y_mark)

        for i in range(self.args.down_sampling_layers):
            batch_x_sampling = down_pool(batch_x_ori)
            batch_y_sampling = batch_y_ori

            batch_x_sampling_list.append(batch_x_sampling.permute(0, 2, 1))
            batch_y_sampling_list.append(batch_y_sampling.permute(0, 2, 1))

            batch_x_mark_list.append(batch_x_mark_ori[:, ::self.args.down_sampling_window, :])
            batch_y_mark_list.append(batch_y_mark_ori)

            batch_x_ori = batch_x_sampling
            batch_y_ori = batch_y_sampling

            batch_x_mark_ori = batch_x_mark_ori[:, ::self.args.down_sampling_window, :]
            batch_y_mark_ori = batch_y_mark_ori

        if self.args.only_use_down_sampling and self.args.down_sampling_layers == 1:
            return batch_x_sampling.permute(0, 2, 1), batch_x_mark[:, ::self.args.down_sampling_window, :], \
                batch_y_sampling.permute(0, 2, 1), batch_y_mark[:, ::self.args.down_sampling_window, :]
        # B,C,T -> B,T,C
        if self.args.down_sampling_layers == 1 and self.args.pred_down_sampling:
            batch_x = [batch_x.permute(0, 2, 1), batch_x_sampling.permute(0, 2, 1)]
            batch_y = batch_y_sampling.permute(0, 2, 1)
            batch_x_mark = [batch_x_mark, batch_x_mark[:, ::self.args.down_sampling_window, :]]
            batch_y_mark = [batch_y_mark, batch_y_mark[:, ::self.args.down_sampling_window, :]]
        else:
            batch_x = batch_x_sampling_list
            batch_y = batch_y.permute(0, 2, 1)
            batch_x_mark = batch_x_mark_list
            batch_y_mark = batch_y_mark

        return batch_x, batch_x_mark, batch_y, batch_y_mark

    
    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                        elif self.args.model == 'TimeMixer':
                            batch_x, batch_x_mark, batch_y, batch_y_mark = \
                                self.__multi_scale_process_inputs(batch_x, batch_x_mark, batch_y, batch_y_mark)
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model:
                        outputs = self.model(batch_x)
                    elif self.args.model == 'TimeMixer':
                        batch_x, batch_x_mark, batch_y, batch_y_mark = \
                            self.__multi_scale_process_inputs(batch_x, batch_x_mark, batch_y, batch_y_mark)
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        epoch_time_all_list = []

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            torch.autograd.set_detect_anomaly(True)
            epoch_time_start = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                        elif self.args.model == 'TimeMixer':
                            batch_x, batch_x_mark, batch_y, batch_y_mark = \
                                self.__multi_scale_process_inputs(batch_x, batch_x_mark, batch_y, batch_y_mark)
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model:
                        outputs = self.model(batch_x)
                    elif self.args.model == 'TimeMixer':
                            batch_x, batch_x_mark, batch_y, batch_y_mark = \
                                self.__multi_scale_process_inputs(batch_x, batch_x_mark, batch_y, batch_y_mark)
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    # print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    # print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    # with torch.autograd.detect_anomaly():
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    # with torch.autograd.detect_anomaly():
                    loss.backward()
                    model_optim.step()
            
            epoch_time_end = time.time()
            epoch_time_all = epoch_time_end - epoch_time_start

            epoch_time_all_list.append(epoch_time_all)

            print("Epoch: {} cost time: {}".format(epoch + 1, epoch_time_all))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)
            
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)
        
        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        epoch_time_all_avg = np.mean(epoch_time_all_list)

        return self.model, epoch_time_all_avg

    def test(self, setting, train_time, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            test_time_start = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                        elif self.args.model == 'TimeMixer':
                            batch_x, batch_x_mark, batch_y, batch_y_mark = \
                                self.__multi_scale_process_inputs(batch_x, batch_x_mark, batch_y, batch_y_mark)
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model:
                        outputs = self.model(batch_x)
                    elif self.args.model == 'TimeMixer':
                        batch_x, batch_x_mark, batch_y, batch_y_mark = \
                            self.__multi_scale_process_inputs(batch_x, batch_x_mark, batch_y, batch_y_mark)
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0

                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    if self.args.model == 'TimeMixer' and self.args.down_sampling_method and self.args.only_use_down_sampling == False and self.args.pred_down_sampling:
                        input = batch_x[-1].detach().cpu().numpy()
                        gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                        pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                        visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))
                        save_to_csv(gt, pd, os.path.join(folder_path, str(i) + '.csv'))
                    elif self.args.model == 'TimeMixer' and self.args.down_sampling_method and self.args.only_use_down_sampling == False and self.args.pred_down_sampling == False:
                        input = batch_x[0].detach().cpu().numpy()
                        gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                        pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                        visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))
                        save_to_csv(gt, pd, os.path.join(folder_path, str(i) + '.csv'))
                    else:
                        input = batch_x.detach().cpu().numpy()
                        gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                        pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                        visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))
                        save_to_csv(gt, pd, os.path.join(folder_path, str(i) + '.csv'))
            test_time_end = time.time()
        test_time = test_time_end - test_time_start
        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mae:{}, mse:{}, train_time:{}, test_time:{}'.format(mae, mse, train_time, test_time))
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mae:{}, mse:{}, train_time:{}, test_time:{}'.format(mae, mse, train_time, test_time))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)
        # time.sleep(100)
        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                        elif self.args.model == 'TimeMixer':
                            batch_x, batch_x_mark, batch_y, batch_y_mark = \
                                self.__multi_scale_process_inputs(batch_x, batch_x_mark, batch_y, batch_y_mark)
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model:
                        outputs = self.model(batch_x)
                    elif self.args.model == 'TimeMixer':
                        batch_x, batch_x_mark, batch_y, batch_y_mark = \
                            self.__multi_scale_process_inputs(batch_x, batch_x_mark, batch_y, batch_y_mark)
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)
        
        return
