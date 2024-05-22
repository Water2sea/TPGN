import torch
import torch.nn as nn
from layers.Pyraformer_EncDec import EncoderLayer, Decoder, Predictor
from layers.Pyraformer_EncDec import Bottleneck_Construct, Conv_Construct, MaxPooling_Construct, AvgPooling_Construct
from layers.Pyraformer_EncDec import get_mask, get_subsequent_mask, refer_points, get_k_q, get_q_k
from layers.Embed import DataEmbedding, CustomEmbedding


class Encoder(nn.Module):
    """ A encoder model with self attention mechanism. """

    def __init__(self, opt):
        super().__init__()

        self.d_model = opt.d_model
        self.model_type = opt.model
        self.window_size = opt.window_size
        self.truncate = opt.truncate

        if opt.use_gpu:
            device = torch.device("cuda")
        else:
            device = torch.device('cpu')

        if opt.decoder == 'attention':
            self.mask, self.all_size = get_mask(opt.seq_len, opt.window_size, opt.inner_size, device)
        else:
            self.mask, self.all_size = get_mask(opt.seq_len+1, opt.window_size, opt.inner_size, device)
        self.decoder_type = opt.decoder
        if opt.decoder == 'FC':
            self.indexes = refer_points(self.all_size, opt.window_size, device)

        if opt.use_tvm:
            assert len(set(self.window_size)) == 1, "Only constant window size is supported."
            padding = 1 if opt.decoder == 'FC' else 0
            q_k_mask = get_q_k(opt.seq_len + padding, opt.inner_size, opt.window_size[0], device)
            k_q_mask = get_k_q(q_k_mask)
            self.layers = nn.ModuleList([
                EncoderLayer(opt.d_model, opt.d_ff, opt.n_heads, opt.d_model // opt.n_heads, opt.d_model // opt.n_heads, dropout=opt.dropout, \
                    normalize_before=False, use_tvm=True, q_k_mask=q_k_mask, k_q_mask=k_q_mask) for i in range(opt.e_layers)
                ])
        else:
            self.layers = nn.ModuleList([
                EncoderLayer(opt.d_model, opt.d_ff, opt.n_heads, opt.d_model // opt.n_heads, opt.d_model // opt.n_heads, dropout=opt.dropout, \
                    normalize_before=False) for i in range(opt.e_layers)
                ])

        if opt.embed_type == 'CustomEmbedding':
            self.enc_embedding = CustomEmbedding(opt.enc_in, opt.d_model, opt.covariate_size, opt.seq_num, opt.dropout)
        else:
            self.enc_embedding = DataEmbedding(opt.enc_in, opt.d_model, 'timeF', opt.freq, opt.dropout)

        self.conv_layers = eval(opt.CSCM)(opt.d_model, opt.window_size, opt.d_bottleneck)

    def forward(self, x_enc, x_mark_enc):

        #import pdb
        #pdb.set_trace()
        seq_enc = self.enc_embedding(x_enc, x_mark_enc)

        mask = self.mask.repeat(len(seq_enc), 1, 1).to(x_enc.device)
        seq_enc = self.conv_layers(seq_enc)

        for i in range(len(self.layers)):
            seq_enc, _ = self.layers[i](seq_enc, mask)

        if self.decoder_type == 'FC':
            indexes = self.indexes.repeat(seq_enc.size(0), 1, 1, seq_enc.size(2)).to(seq_enc.device)
            indexes = indexes.view(seq_enc.size(0), -1, seq_enc.size(2))
            all_enc = torch.gather(seq_enc, 1, indexes)
            seq_enc = all_enc.view(seq_enc.size(0), self.all_size[0], -1)
        elif self.decoder_type == 'attention' and self.truncate:
            seq_enc = seq_enc[:, :self.all_size[0]]

        return seq_enc


class Model(nn.Module):
    """ A sequence to sequence model with attention mechanism. """

    def __init__(self, opt):
        super().__init__()

        self.predict_step = opt.pred_len
        self.d_model = opt.d_model
        self.input_size = opt.seq_len
        self.decoder_type = opt.decoder
        self.channels = opt.enc_in

        self.encoder = Encoder(opt)
        if opt.decoder == 'attention':
            mask = get_subsequent_mask(opt.seq_len, opt.window_size, opt.pred_len, opt.truncate)
            self.decoder = Decoder(opt, mask)
            self.predictor = Predictor(opt.d_model, opt.c_out)
        elif opt.decoder == 'FC':
            self.predictor = Predictor(4 * opt.d_model, opt.pred_len * opt.c_out)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, pretrain=False):
        """
        Return the hidden representations and predictions.
        For a sequence (l_1, l_2, ..., l_N), we predict (l_2, ..., l_N, l_{N+1}).
        Input: event_type: batch*seq_len;
               event_time: batch*seq_len.
        Output: enc_output: batch*seq_len*model_dim;
                type_prediction: batch*seq_len*num_classes (not normalized);
                time_prediction: batch*seq_len.
        """
        
        if self.decoder_type == 'attention':
            enc_output = self.encoder(x_enc, x_mark_enc)
            dec_enc = self.decoder(x_dec[:, -self.predict_step:, :], x_mark_dec[:, -self.predict_step:, :], enc_output)

            if pretrain:
                dec_enc = torch.cat([enc_output[:, :self.input_size], dec_enc], dim=1)
                pred = self.predictor(dec_enc)
            else:
                pred = self.predictor(dec_enc)
        
        elif self.decoder_type == 'FC':
            # Add a predict token into the history sequence
            predict_token = torch.zeros(x_enc.size(0), 1, x_enc.size(-1), device=x_enc.device)
            x_enc = torch.cat([x_enc, predict_token], dim=1)
            x_mark_enc = torch.cat([x_mark_enc, x_mark_dec[:, -self.predict_step:-self.predict_step+1, :]], dim=1)
            
            enc_output = self.encoder(x_enc, x_mark_enc)[:, -1, :]
            pred = self.predictor(enc_output).view(enc_output.size(0), self.predict_step, -1)

        return pred

