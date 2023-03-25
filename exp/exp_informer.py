from data.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred
from exp.exp_basic import Exp_Basic
from models.model import Informer, InformerStack

from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric

import numpy as np

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

import os
import time

import warnings
warnings.filterwarnings('ignore')


class Exp_Informer(Exp_Basic):
    def __init__(self, args):
        super(Exp_Informer, self).__init__(args)
        self.root_path = args.root_path
        self.subject = args.subject
        self.set_type = 'E'

    def _build_model(self):
        if self.args.model_flag == 0:
            model_dict = {
                'informer': Informer,
                'informerstack': InformerStack,
            }
            if self.args.model == 'informer' or self.args.model == 'informerstack':
                e_layers = self.args.e_layers if self.args.model == 'informer' else self.args.s_layers
                model = model_dict[self.args.model](
                    self.args.enc_in,
                    self.args.dec_in,
                    self.args.c_out,
                    self.args.seq_len,
                    self.args.label_len,
                    self.args.pred_len,
                    self.args.factor,
                    self.args.d_model,
                    self.args.n_heads,
                    e_layers,
                    self.args.d_layers,
                    self.args.d_ff,
                    self.args.dropout,
                    self.args.attn,
                    self.args.embed,
                    self.args.freq,
                    self.args.activation,
                    self.args.output_attention,
                    self.args.distil,
                    self.args.mix,
                    self.device
                ).float()

            if self.args.use_multi_gpu and self.args.use_gpu:
                model = nn.DataParallel(model, device_ids=self.args.device_ids)
        else:
            print('载入模型')
            model = torch.load(self.args.model_Save_Path)
        return model

    def _get_data(self, flag):
        args = self.args

        data_dict = {
            'ETTh1': Dataset_ETT_hour,
            'ETTh2': Dataset_ETT_hour,
            'ETTm1': Dataset_ETT_minute,
            'ETTm2': Dataset_ETT_minute,
            'WTH': Dataset_Custom,
            'ECL': Dataset_Custom,
            'Solar': Dataset_Custom,
            'custom': Dataset_Custom,
        }
        Data = data_dict[self.args.data]
        timeenc = 0 if args.embed != 'timeF' else 1

        if flag == 'test':
            shuffle_flag = False
            drop_last = True
            batch_size = args.batch_size
            freq = args.freq
        elif flag == 'pred':
            shuffle_flag = False
            drop_last = False
            batch_size = 1
            freq = args.detail_freq
            Data = Dataset_Pred
        else:
            shuffle_flag = True
            drop_last = True
            batch_size = args.batch_size
            freq = args.freq
        data_set = Data(
            subject=args.subject,
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            inverse=args.inverse,
            timeenc=timeenc,
            freq=freq,
            cols=args.cols
        )
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)

        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim
    
    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        self.model.eval()
        total_loss = []
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
            pred, true, no_use = self._process_one_batch(
                vali_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
            loss = criterion(pred.detach().cpu(), true.detach().cpu())
            total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')         # 调用训练
        vali_data, vali_loader = self._get_data(flag='val')             # 调用测试

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        
        train_steps = len(train_loader)
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                # print('exp_informer', 'batch_x', batch_x.shape, 'batch_y', batch_y.shape)
                model_optim.zero_grad()
                pred, true, no_use = self._process_one_batch(
                    train_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
                loss = criterion(pred, true)
                train_loss.append(loss.item())
                
                if (i+1) % 100==0:
                    speed = (time.time()-time_now)/iter_count
                    left_time = speed*((self.args.train_epochs - epoch)*train_steps - i)
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss))
            adjust_learning_rate(model_optim, epoch+1, self.args)

        return self.model

    def test(self, setting, SavePath, TestPath, VIO):
        test_Save_len = 200          # 每一组记录的数据
        self.model.eval()
        # result save       建立文件夹
        folder_path = SavePath + VIO +'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        torch.save(self.model, self.args.model_Save_Path)
        preds = []
        trues = []
        Point_List = []

        Data_Get = TestPath + VIO + '.pt'      # 载入对应数据路径
        print(Data_Get)
        Test_Tensor = torch.load(Data_Get)                      # 载入数据
        channel_num, Data_Get_Len = Test_Tensor.shape  # 获取数据通道数，持续时间
        test_index_len = Data_Get_Len - self.args.seq_len - self.args.pred_len  # 索引长度
        Test_data_x = torch.clone(Test_Tensor.t())                  # 样本数据
        Test_data_y = torch.clone(Test_Tensor.t())  # 样本数据
        time_Len = self.args.seq_len + self.args.pred_len                 # 实际的时间序列
        time_index = torch.linspace(1, time_Len, time_Len)          # 线性产生标签时间数据
        Test_data_stamp = time_index / time_Len - 0.5                   # 时间编码
        seq_x_mark = torch.unsqueeze(Test_data_stamp[0:self.args.seq_len], dim=1)
        seq_y_mark = torch.unsqueeze(Test_data_stamp[self.args.seq_len - self.args.label_len:self.args.seq_len], dim=1)
        seq_x_mark_clone = torch.clone(torch.unsqueeze(seq_x_mark, dim=0))
        seq_y_mark_clone = torch.clone(torch.unsqueeze(seq_y_mark, dim=0))
        if self.args.test_len > Data_Get_Len:
            Cir_Num = int(Data_Get_Len / self.args.label_len) - 10                 # 可进行完整循环的次数
        else:
            Cir_Num = int(self.args.test_len / self.args.label_len) - 10     # 样本循环次数
        for Cir_i in range(Cir_Num):                            # 进行数据循环得到结果
            if Cir_i % 10000 == 0:
                print("完成测试", round(Cir_i / Cir_Num * 100, 2), "%")
            Lable_be = 0
            s_begin = Cir_i * self.args.label_len + self.args.test_Start_Flag  # 0
            s_end = s_begin + self.args.seq_len  # self.seq_len
            r_begin = s_end - self.args.label_len - Lable_be  # self.seq_len - self.label_len
            r_end = s_end - Lable_be  #
            seq_x = torch.unsqueeze(Test_data_x[s_begin:s_end, :4], dim=0)          # 训练集
            seq_y = torch.unsqueeze(Test_data_y[r_begin:r_end, :], dim=0)         # 标签
            if Cir_i % test_Save_len == 0:
                seq_x_batch = torch.clone(seq_x)
                seq_y_batch = torch.clone(seq_y)
                seq_x_mark_batch = torch.clone(seq_x_mark_clone)
                seq_y_mark_batch = torch.clone(seq_y_mark_clone)
            else:
                seq_x_batch = torch.cat((seq_x_batch, seq_x), dim=0)
                seq_y_batch = torch.cat((seq_y_batch, seq_y), dim=0)
                seq_x_mark_batch = torch.cat((seq_x_mark_batch, seq_x_mark_clone), dim=0)
                seq_y_mark_batch = torch.cat((seq_y_mark_batch, seq_y_mark_clone), dim=0)
            if (Cir_i % test_Save_len == test_Save_len - 1) or (Cir_i == Cir_Num - 1):
                pred, true, point_use = self._process_one_batch(1, seq_x_batch, seq_y_batch, seq_x_mark_batch, seq_y_mark_batch)           # 获取预测与真实值
                for Batch_i in range(point_use.shape[0]):

                    Point_List.append(point_use[Batch_i].detach().cpu().numpy())         # 节点数据列表
                    preds.append(pred[Batch_i].detach().cpu().numpy())
                    trues.append(true[Batch_i].detach().cpu().numpy())

        preds = np.array(preds)
        trues = np.array(trues)
        Point_List = np.array(Point_List)
        print('test shape:', preds.shape, trues.shape, Point_List.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        Point_List = Point_List.reshape(-1, Point_List.shape[-2], Point_List.shape[-1])
        print('test shape:', preds.shape, trues.shape, Point_List.shape)
        Point_List = np.array(Point_List)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))

        np.save(folder_path+'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path+'pred.npy', preds)
        np.save(folder_path+'true.npy', trues)
        np.save(folder_path + 'Point.npy', Point_List)

        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')
        
        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path+'/'+'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        self.model.eval()
        
        preds = []
        
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(pred_loader):
            pred, true, no_use = self._process_one_batch(
                pred_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
            preds.append(pred.detach().cpu().numpy())

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        
        # result save
        folder_path = './results/' + setting +'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        np.save(folder_path+'real_prediction.npy', preds)
        
        return

    def _process_one_batch(self, dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark):
        batch_x = batch_x.float().to(self.device)           # 放置gpu
        batch_y = batch_y.float()
        batch_x_mark = batch_x_mark.float().to(self.device)         # 时间标志位
        batch_y_mark = batch_y_mark.float().to(self.device)
        dec_inp = torch.zeros([batch_y.shape[0], self.args.label_len, batch_x.shape[2]]).float().to(self.device)
        outputs, outpus_clone = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

        f_dim = -1 if self.args.features == 'MS' else 0
        batch_y = batch_y[:, :, batch_x.shape[2]:].to(self.device)
        return outputs, batch_y, outpus_clone

