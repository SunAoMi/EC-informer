"""--model "informer" --data "ETTh1" --do_predict"""

import argparse
import os
import torch
from exp.exp_informer import Exp_Informer
import shutil
import time


parser = argparse.ArgumentParser(description='[Informer] Long Sequences Forecasting')

parser.add_argument('--model', type=str, required=True, default='informer', help='model of experiment, options: [informer, informerstack, informerlight(TBD)]')
parser.add_argument('--subject', type=str, default='A01', help='被选择的实验对象')

parser.add_argument('--data', type=str, required=True, default='ETTh1', help='data')
parser.add_argument('--root_path', type=str, default=r'L:\tensorflow脑机接口\Informer2020脑机通道补全\预测样本', help='数据载入根目录')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
parser.add_argument('--features', type=str, default='MS', help='预测任务，选项:[M, S, MS];M:多元预测多元，S:一元预测一元，MS:多元预测一元')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='h', help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

parser.add_argument('--train_epochs', type=int, default=15, help='train epochs')
parser.add_argument('--batch_size', type=int, default=128, help='batch size of train input data')
parser.add_argument('--seq_len', type=int, default=48, help='input sequence length of Informer encoder')
parser.add_argument('--label_len', type=int, default=24, help='start token length of Informer decoder')
parser.add_argument('--pred_len', type=int, default=0, help='prediction sequence length')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--learning_rate', type=float, default=0.001, help='optimizer learning rate')
parser.add_argument('--enc_in', type=int, default=25, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=25, help='decoder input size')
parser.add_argument('--c_out', type=int, default=25, help='output size')
parser.add_argument('--d_model', type=int, default=128, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=16, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--s_layers', type=str, default='3,2,1', help='num of stack encoder layers')
parser.add_argument('--d_ff', type=int, default=512, help='dimension of fcn')
parser.add_argument('--factor', type=int, default=8, help='probsparse attn factor')
parser.add_argument('--padding', type=int, default=0, help='padding type')
parser.add_argument('--distil', action='store_false', help='是否在编码器中使用蒸馏，使用这个参数意味着不使用蒸馏', default=True)
parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
parser.add_argument('--attn', type=str, default='prob', help='attention used in encoder, options:[prob, full]')
parser.add_argument('--embed', type=str, default='timeF', help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu',help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
parser.add_argument('--do_predict', action='store_true', help='是否预测不可见的未来数据')
parser.add_argument('--mix', action='store_false', help='use mix attention in generative decoder', default=True)
parser.add_argument('--cols', type=str, nargs='+', help='certain cols from the data files as the input features')
parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
parser.add_argument('--patience', type=int, default=3, help='early stopping patience')

parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='mse', help='loss function')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
parser.add_argument('--inverse', action='store_true', help='逆输出数据', default=False)
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

args = parser.parse_args()

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False  # 确认显卡

if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

data_parser = {
    'ETTh1': {'data': 'ETTh1.csv', 'T': 'OT', 'M': [5, 5, 5], 'S': [1, 1, 1], 'MS': [4, 4, 1]},
    'ETTh2': {'data': 'ETTh2.csv', 'T': 'OT', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1]},
    'ETTm1': {'data': 'ETTm1.csv', 'T': 'OT', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1]},
    'ETTm2': {'data': 'ETTm2.csv', 'T': 'OT', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1]},
    'WTH': {'data': 'WTH.csv', 'T': 'WetBulbCelsius', 'M': [12, 12, 12], 'S': [1, 1, 1],
            'MS': [12, 12, 1]},
    'ECL': {'data': 'ECL.csv', 'T': 'MT_320', 'M': [321, 321, 321], 'S': [1, 1, 1],
            'MS': [321, 321, 1]},
    'Solar': {'data': 'solar_AL.csv', 'T': 'POWER_136', 'M': [137, 137, 137], 'S': [1, 1, 1],
              'MS': [137, 137, 1]},
}

if args.data in data_parser.keys():
    data_info = data_parser[args.data]
    args.data_path = data_info['data']
    args.target = data_info['T']
    args.enc_in, args.dec_in, args.c_out = data_info[
        args.features]

args.s_layers = [int(s_l) for s_l in args.s_layers.replace(' ', '').split(',')]
args.detail_freq = args.freq
args.freq = args.freq[-1:]

print('Args in experiment:')
print(args)

Exp = Exp_Informer
SavePath = r''
setting = '{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_at{}_fc{}_eb{}_dt{}_mx{}_{}_{}'.format(
    args.model, args.data, args.features,
    args.seq_len, args.label_len, args.pred_len,
    args.d_model, args.n_heads, args.e_layers, args.d_layers, args.d_ff, args.attn, args.factor,
    args.embed, args.distil, args.mix, args.des, 1)
Cicle_Name = ['A01']
Channel_Flag = {'test': 1}
Save_Path = r'Root'
Date_Get_Path = r'Root'

Save_times = 21
All_Time = 5

# 主函数开始
if __name__ == "__main__":
    Start_time = time.time()
    NewStart = Start_time
    for All_i in range(All_Time):
        Save_Path = r'Root' + str(Save_times)
        # 添加文件夹
        if not os.path.exists(Save_Path):
            os.mkdir(Save_Path)
        Save_times = Save_times + 1

        for Cicle_Name_Key in Channel_Flag:
            # 多被试输出
            if Cicle_Name_Key == 'test' and Channel_Flag[Cicle_Name_Key] == 1:
                Data_Get_Root = Date_Get_Path + '\\' + '多被试输出'
                filepath = Save_Path + '\\' + Cicle_Name_Key
                # 添加文件夹
                if not os.path.exists(filepath):
                    os.mkdir(filepath)
                else:
                    shutil.rmtree(filepath, ignore_errors=True)
                    os.mkdir(filepath)
                # 各进行训练
                for Cicle_Name_Aim in range(len(Cicle_Name)):

                    Aim_name = Cicle_Name[Cicle_Name_Aim]                               # 目标名称
                    print("当前进行--", Cicle_Name_Key, '||目标对象--', Aim_name)
                    # 添加路径
                    args.subject = Aim_name             # 目标对象
                    args.root_path = Data_Get_Root
                    exp = Exp(args)
                    print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
                    exp.train(setting)
                    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))

                    Save_Path_Root = filepath + '\\' + Aim_name
                    # 添加文件夹
                    if not os.path.exists(Save_Path_Root):
                        os.mkdir(Save_Path_Root)
                    else:
                        shutil.rmtree(Save_Path_Root, ignore_errors=True)
                        os.mkdir(Save_Path_Root)
                    TestPath = Data_Get_Root + '\\' + Aim_name
                    exp.test(setting, Save_Path_Root, TestPath)  # 开始验证              可以画图
                    torch.cuda.empty_cache()  # 释放显存

                    print("已训练时间", round(time.time() - Start_time, 2), 's', '||单次训练时间', round(time.time() - NewStart, 2), 's')
                    NewStart = time.time()


