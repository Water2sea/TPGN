import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from layers.Embed import WITRAN_Temporal_Embedding

class WITRAN_2DPSGMU_Encoder(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, water_rows, water_cols, res_mode='none'):
        super(WITRAN_2DPSGMU_Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.water_rows = water_rows
        self.water_cols = water_cols
        self.res_mode = res_mode
        # parameter of row cell
        self.W_first_layer = torch.nn.Parameter(torch.empty(6 * hidden_size, input_size + 2 * hidden_size))
        self.W_other_layer = torch.nn.Parameter(torch.empty(num_layers - 1, 6 * hidden_size, 4 * hidden_size))
        self.B = torch.nn.Parameter(torch.empty(num_layers, 6 * hidden_size))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, +stdv)

    def linear(self, input, weight, bias, batch_size, slice, Water2sea_slice_num):
        a = F.linear(input, weight)
        if slice < Water2sea_slice_num:
            a[:batch_size * (slice + 1), :] = a[:batch_size * (slice + 1), :] + bias
        return a

    def forward(self, input, batch_size, input_size, flag):
        if flag == 1: # cols > rows
            input = input.permute(2, 0, 1, 3)
        else:
            input = input.permute(1, 0, 2, 3)
        Water2sea_slice_num, _, Original_slice_len, _ = input.shape
        Water2sea_slice_len = Water2sea_slice_num + Original_slice_len - 1
        hidden_slice_row = torch.zeros(Water2sea_slice_num * batch_size, self.hidden_size).to(input.device)
        hidden_slice_col = torch.zeros(Water2sea_slice_num * batch_size, self.hidden_size).to(input.device)
        input_transfer = torch.zeros(Water2sea_slice_num, batch_size, Water2sea_slice_len, input_size).to(input.device)
        for r in range(Water2sea_slice_num):
            input_transfer[r, :, r:r+Original_slice_len, :] = input[r, :, :, :]
        hidden_row_all_list = []
        hidden_col_all_list = []
        for layer in range(self.num_layers):
            if layer == 0:
                a = input_transfer.reshape(Water2sea_slice_num * batch_size, Water2sea_slice_len, input_size)
                W = self.W_first_layer
            else:
                a = F.dropout(output_all_slice, self.dropout, self.training)
                if layer == 1:
                    layer0_output = a
                W = self.W_other_layer[layer-1, :, :]
                hidden_slice_row = hidden_slice_row * 0
                hidden_slice_col = hidden_slice_col * 0
            B = self.B[layer, :]
            # start every for all slice
            output_all_slice_list = []
            for slice in range (Water2sea_slice_len):
                # gate generate
                gate = self.linear(torch.cat([hidden_slice_row, hidden_slice_col, a[:, slice, :]], 
                    dim = -1), W, B, batch_size, slice, Water2sea_slice_num)
                # gate
                sigmod_gate, tanh_gate = torch.split(gate, 4 * self.hidden_size, dim = -1)
                sigmod_gate = torch.sigmoid(sigmod_gate)
                tanh_gate = torch.tanh(tanh_gate)
                update_gate_row, output_gate_row, update_gate_col, output_gate_col = sigmod_gate.chunk(4, dim = -1)
                input_gate_row, input_gate_col = tanh_gate.chunk(2, dim = -1)
                # gate effect
                hidden_slice_row = torch.tanh(
                    (1-update_gate_row)*hidden_slice_row + update_gate_row*input_gate_row) * output_gate_row
                hidden_slice_col = torch.tanh(
                    (1-update_gate_col)*hidden_slice_col + update_gate_col*input_gate_col) * output_gate_col
                # output generate
                output_slice = torch.cat([hidden_slice_row, hidden_slice_col], dim = -1)
                # save output
                output_all_slice_list.append(output_slice)
                # save row hidden
                if slice >= Original_slice_len - 1:
                    need_save_row_loc = slice - Original_slice_len + 1
                    hidden_row_all_list.append(
                        hidden_slice_row[need_save_row_loc*batch_size:(need_save_row_loc+1)*batch_size, :])
                # save col hidden
                if slice >= Water2sea_slice_num - 1:
                    hidden_col_all_list.append(
                        hidden_slice_col[(Water2sea_slice_num-1)*batch_size:, :])
                # hidden transfer
                hidden_slice_col = torch.roll(hidden_slice_col, shifts=batch_size, dims = 0)
            if self.res_mode == 'layer_res' and layer >= 1: # layer-res
                output_all_slice = torch.stack(output_all_slice_list, dim = 1) + layer0_output
            else:
                output_all_slice = torch.stack(output_all_slice_list, dim = 1)
        hidden_row_all = torch.stack(hidden_row_all_list, dim = 1)
        hidden_col_all = torch.stack(hidden_col_all_list, dim = 1)
        hidden_row_all = hidden_row_all.reshape(batch_size, self.num_layers, Water2sea_slice_num, hidden_row_all.shape[-1])
        hidden_col_all = hidden_col_all.reshape(batch_size, self.num_layers, Original_slice_len, hidden_col_all.shape[-1])
        if flag == 1:
            return output_all_slice, hidden_col_all, hidden_row_all
        else:
            return output_all_slice, hidden_row_all, hidden_col_all

class Model(nn.Module):
    def __init__(self, configs, WITRAN_dec='Concat', WITRAN_res='none', WITRAN_PE='add'):
        super(Model, self).__init__()
        self.standard_batch_size = configs.batch_size
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.dec_in = configs.dec_in
        self.c_out = configs.c_out
        self.d_model = configs.d_model
        self.num_layers = configs.e_layers
        self.dropout = configs.dropout
        self.WITRAN_dec = WITRAN_dec
        self.WITRAN_deal = configs.WITRAN_deal
        self.WITRAN_res = WITRAN_res
        self.PE_way = WITRAN_PE
        self.WITRAN_grid_cols = configs.WITRAN_grid_cols
        self.WITRAN_grid_enc_rows = int(configs.seq_len / self.WITRAN_grid_cols)
        self.WITRAN_grid_dec_rows = int(configs.pred_len / self.WITRAN_grid_cols)
        self.device = configs.gpu
        if configs.freq== 'h':
            Temporal_feature_dim = 4
        # Encoder
        self.encoder_2d = WITRAN_2DPSGMU_Encoder(self.enc_in + Temporal_feature_dim, self.d_model, self.num_layers, 
            self.dropout, self.WITRAN_grid_enc_rows, self.WITRAN_grid_cols, self.WITRAN_res)
        # Embedding
        self.dec_embedding = WITRAN_Temporal_Embedding(Temporal_feature_dim, configs.d_model,
            configs.embed, configs.freq, configs.dropout)
        
        if self.PE_way == 'add':
            if self.WITRAN_dec == 'FC':
                self.fc_1 = nn.Linear(self.num_layers * (self.WITRAN_grid_enc_rows + self.WITRAN_grid_cols) * self.d_model, 
                    self.pred_len * self.d_model)
            elif self.WITRAN_dec == 'Concat':
                self.fc_1 = nn.Linear(self.num_layers * 2 * self.d_model, self.WITRAN_grid_dec_rows * self.d_model)
            self.fc_2 = nn.Linear(self.d_model, self.c_out)
        else:
            if self.WITRAN_dec == 'FC':
                self.fc_1 = nn.Linear(self.num_layers * (self.WITRAN_grid_enc_rows + self.WITRAN_grid_cols) * self.d_model, 
                    self.pred_len * self.d_model)
            elif self.WITRAN_dec == 'Concat':
                self.fc_1 = nn.Linear(self.num_layers * 2 * self.d_model, self.WITRAN_grid_dec_rows * self.d_model)
            self.fc_2 = nn.Linear(self.d_model * 2, self.c_out)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        if self.WITRAN_deal == 'standard':
            seq_last = x_enc[:,-1:,:].detach()
            x_enc = x_enc - seq_last
        
        x_input_enc = torch.cat([x_enc, x_mark_enc], dim = -1)
        batch_size, _, input_size = x_input_enc.shape
        x_input_enc = x_input_enc.reshape(batch_size, self.WITRAN_grid_enc_rows, self.WITRAN_grid_cols, input_size)

        if self.WITRAN_grid_enc_rows <= self.WITRAN_grid_cols:
            flag = 0
        else: # need permute
            flag = 1
        
        _, enc_hid_row, enc_hid_col = self.encoder_2d(x_input_enc, batch_size, input_size, flag)
        dec_T_E = self.dec_embedding(x_mark_dec)

        if self.WITRAN_dec == 'FC':
            hidden_all = torch.cat([enc_hid_row, enc_hid_col], dim = 2)
            hidden_all = hidden_all.reshape(hidden_all.shape[0], -1)
            last_output = self.fc_1(hidden_all)
            last_output = last_output.reshape(last_output.shape[0], self.pred_len, -1)
            
        elif self.WITRAN_dec == 'Concat':
            enc_hid_row = enc_hid_row[:, :, -1:, :].expand(-1, -1, self.WITRAN_grid_cols, -1)
            output = torch.cat([enc_hid_row, enc_hid_col], dim = -1).permute(0, 2, 1, 3)
            output = output.reshape(output.shape[0], 
                output.shape[1], output.shape[2] * output.shape[3])
            last_output = self.fc_1(output)
            last_output = last_output.reshape(last_output.shape[0], last_output.shape[1], 
                self.WITRAN_grid_dec_rows, self.d_model).permute(0, 2, 1, 3)
            last_output = last_output.reshape(last_output.shape[0], 
                last_output.shape[1] * last_output.shape[2], last_output.shape[3])
            
        if self.PE_way == 'add':
            last_output = last_output + dec_T_E
            if self.WITRAN_deal == 'standard':
                last_output = self.fc_2(last_output) + seq_last
            else:
                last_output = self.fc_2(last_output)
        else:
            if self.WITRAN_deal == 'standard':
                last_output = self.fc_2(torch.cat([last_output, dec_T_E], dim = -1)) + seq_last
            else:
                last_output = self.fc_2(torch.cat([last_output, dec_T_E], dim = -1))
        
        return last_output