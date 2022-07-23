import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import re,ast
import time
import multiprocessing as mp
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_colwidth',1000) #完整显示，不省略
pd.set_option('display.expand_frame_repr', False) #显示全部列并且字段不换行

'''数据读取'''

# blocks=pd.read_csv("/Users/lizhihan/Desktop/Bitcoin data/data_blockchair/download/blockchair_bitcoin_blocks_20220531.tsv", delimiter = '\t')
tx=pd.read_csv("/Users/lizhihan/Desktop/Bitcoin data/data_blockchair/download/blockchair_bitcoin_transactions_20220531.tsv", delimiter = '\t')
input_data=pd.read_csv("/Users/lizhihan/Desktop/Bitcoin data/data_blockchair/download/blockchair_bitcoin_inputs_20220531.tsv", delimiter = '\t')
output_data=pd.read_csv("/Users/lizhihan/Desktop/Bitcoin data/data_blockchair/download/blockchair_bitcoin_outputs_20220531.tsv", delimiter = '\t')
ad_info=pd.read_csv('/Users/lizhihan/Desktop/Bitcoin data/data_blockchair/ad_info_738690-738789.csv')
# blocks=blocks[['id', 'hash', 'time', 'transaction_count',  'input_count', 'output_count',
#                'input_total', 'input_total_usd', 'output_total', 'output_total_usd']]

tx=tx[['block_id', 'hash', 'time', 'is_coinbase', 'input_count',
         'output_count','input_total', 'input_total_usd', 'output_total', 'output_total_usd']]
tx.columns=['block_id', 'tx_hash', 'tx_time', 'is_coinbase', 'input_count',
         'output_count','input_total', 'input_total_usd', 'output_total', 'output_total_usd']

input_data=input_data[['spending_block_id','transaction_hash', 'time', 'value', 'value_usd',
       'recipient', 'type', 'is_from_coinbase', 'is_spendable',
       'spending_transaction_hash', 'spending_time', 'spending_value_usd', 'lifespan']]
input_data.columns=['block_id','transaction_hash', 'time', 'address_value', 'value_usd',
       'address_hash', 'type', 'is_from_coinbase', 'is_spendable',
       'tx_hash', 'spending_time', 'spending_value_usd', 'lifespan']

output_data=output_data[['block_id','transaction_hash', 'time', 'value', 'value_usd',
       'recipient', 'type', 'is_from_coinbase', 'is_spendable']]
output_data.columns=['block_id','tx_hash', 'time', 'address_value', 'value_usd',
       'address_hash', 'type', 'is_from_coinbase', 'is_spendable']

tx=tx[(tx['block_id']>=738690)&(tx['block_id']<=738789)]
input_data=input_data[(input_data['block_id']>=738690)&(input_data['block_id']<=738789)]
output_data=output_data[(output_data['block_id']>=738690)&(output_data['block_id']<=738789)]

tx=tx[tx['is_coinbase']==0]
input_data=input_data[input_data['is_from_coinbase']==0]
output_data=output_data[output_data['is_from_coinbase']==0]

'''数据口径统一
    -- 数据集tx中某些交易并不在input_data与output_data中，因此分析的交易先取这3个数据的交集
    -- 有些交易无法解码输入或者输出，因此在取交集后还要剔除无法解码输出或输入的交易，然后剩下的交易用于多输入启发式聚类和找零地址识别
'''

tx_list1=tx['tx_hash'].tolist()
tx_list2=input_data['tx_hash'].tolist()
tx_list3=output_data['tx_hash'].tolist()
tx_list=list(set(tx_list1)&set(tx_list2)&set(tx_list3)) #交易取3个数据的交集
tx_df=pd.DataFrame({'tx_hash':tx_list})

tx=pd.merge(tx_df,tx,on='tx_hash',how='left')
input_data=pd.merge(tx_df,input_data,on='tx_hash',how='left')
output_data=pd.merge(tx_df,output_data,on='tx_hash',how='left')

print('交易总数：',len(tx_list))

#剔除无法解码输出或输入的交易
# tx_invalid=set()
# for t in tx_list:
#     if input_data[input_data['tx_hash']==t].shape[0]==0 or output_data[output_data['tx_hash']==t].shape[0]==0:
#         tx_invalid=tx_invalid|{t}
# tx_list=list(set(tx_list)-tx_invalid)

'''地址脚本类型处理'''
print('执行时间：', time.strftime('%Y-%m-%d %A %H:%M:%S', time.localtime()))
ad_list=list(set(input_data['address_hash'].tolist()+output_data['address_hash'].tolist()))
print('地址总数：',len(ad_list))

script_type=[]
for ad in ad_list:
    if ad[0]=='1':
        script_type.append('P2PKH')
    elif ad[0]=='3':
        script_type.append('P2SH')
    elif ad[:3]=='bc1':
        script_type.append('Bech32')
    else:
        script_type.append('Address which can not be parsed') #无法解析的地址
ad_dataset=pd.DataFrame({'address_hash':ad_list,'script_type':script_type})
ad_script_dic={ad_list[i]:script_type[i] for i in range(len(ad_list))}
input_data=pd.merge(input_data,ad_dataset,on='address_hash',how='left')
output_data=pd.merge(output_data,ad_dataset,on='address_hash',how='left')

input_data.dropna(axis=0, subset=['script_type'], inplace=True)
output_data.dropna(axis=0, subset=['script_type'], inplace=True)
print('输入地址总数：',len(set(input_data['address_hash'].tolist())))
print('输出地址总数：',len(set(output_data['address_hash'].tolist())))


'''找零地址识别'''

def change_address_identify(tx,input_data,output_data,ad_info,tx_list):


    start_time = time.time()
    h2=[] #剔除coinbase交易，不是self——change，输出中唯一首次出现的地址
    h3=[] #f1基础上考虑地址重用
    h4=[] #找零地址与输入脚本相同
    h5=[] #收款人更可能接收一个整数付款
    h6=[] #找零地址收款额小于所有输入

    script_fea1=[] #每个交易的所有输入类型是否相同
    condition1=[] #启发式1，保存用脚本类型识别的找零地址
    condition2=[] #启发式2，保存基于整数值接收识别的找零地址
    condition3=[] #启发式3，保存基于找零地址接收金额小于所有输入原则，识别的找零地址
    h7=[] #用于保存最终识别出的找零地址
    tx_type=[] #存储每个交易的类型
    k=0
    for t in  tx_list:
        t_info = tx[tx['tx_hash'] == t]  # 每个交易的总体信息
        input_count=t_info['input_count'].values[0] # 每个交易付款地址数
        output_count = t_info['output_count'].values[0]  # 每个交易的接收地址数
        tx_input = input_data[input_data['tx_hash'] == t]  # 每个交易所有输入
        input_address = tx_input['address_hash'].tolist()  # 所有的输入地址

        '''交易类型判断'''
        if input_count==0:
            tx_type.append('无法识别输入')
        elif output_count==0:
            tx_type.append('无法识别输出')
        elif output_count==1:
            tx_type.append('单输出')
        elif input_count==1:
            if output_count==2:
                tx_type.append('单输入2输出')
            else:
                tx_type.append('单输入多输出')
        elif output_count==2:
            tx_type.append('多输入2输出')
        else:
            tx_type.append('多输入多输出')

        '''交易输出数'''
        if output_count > 1:  # 只有输出地址数大于1才识别找零地址
            tx_output = output_data[output_data['tx_hash'] == t]  # 每个交易的所有输出
            t_time = t_info['tx_time'].values[0]  # 交易发生时间
            input_script = tx_input['script_type'].tolist()  # 所有输入的脚本类型
            output_script = tx_output['script_type'].tolist()  # 所有输出的脚本类型
            intput_sending = tx_input['address_value'].tolist()  # 所有的输入地址付款额
            output_address = tx_output['address_hash'].tolist()  # 所有的输出地址
            output_receive = tx_output['address_value'].tolist()  # 所有的输出地址收款额
            output_receive_usd = tx_output['value_usd'].tolist()  # 所有的输出地址收款额换算为美元
            output_dict = {output_address[k]: output_receive[k] for k in range(len(output_address))}  # 构建输出地址与BTC收款额的字典
            output_dict_usd={output_address[k]: output_receive_usd[k] for k in range(len(output_address))}

            '''h2: 剔除coinbase交易和self—change交易后，输出中唯一首次出现的地址'''

            self_change_tx=[] #保存self——change交易
            ad_h2 = []  # 保存所有首次出现的地址
            for ad in output_address:
                if ad in input_address: #是否self——change交易
                    self_change_tx.append('self-change交易')
                    break
                else:
                    first_receive_time = ad_info[ad_info['address_hash'] == ad]['first_seen_receiving'].values[0]  # 地址第一次收款时间
                    if t_time == first_receive_time:  # 第一次收款时间与交易发生时间相同，说明地址是第一次出现
                        ad_h2.append(ad)
            if self_change_tx !=[] and self_change_tx[0]=='self-change交易':
                h2.append('self-change交易')
            elif len(ad_h2) == 0:
                h2.append('没有新地址')
            elif len(ad_h2) == 1:
                h2.append(ad_h2[0])
            else:
                h2.append('有多个新地址')

            '''h3：h2基础上考虑只有找零地址不会重用'''
            ad_h3 = []
            if self_change_tx !=[] and self_change_tx[0]=='self-change交易':
                h3.append('self-change交易')
            elif len(ad_h2) == 0:
                h3.append('没有新地址')
            else:
                for ad in output_address:
                    if ad_info[ad_info['address_hash'] == ad]['output_count'].values[0] == 1:  # 地址是新地址且未被重用
                        ad_h3.append(ad)
                if len(ad_h3) == 0:
                    h3.append('新地址都被重用')
                elif len(ad_h3) == 1:
                    h3.append(ad_h3[0])
                else:
                    h3.append('多个新地址都没有被重用')

            '''h4:找零地址与输入脚本相同'''
            ad_h4 = []
            if len(set(input_script)) == 1:  # 所有输入脚本是否相同
                if len(set(output_script)) > 1:  # 输出地址脚本类型数是否大于1，等于1无论输出有几个地址，脚本判断方法都失效
                    input_script_type = input_script[0]  # 输入脚本的类型
                    for ad in output_address:
                        if ad_script_dic[ad] == input_script_type:  # 判断每个输出的类型与输入是否相同
                            ad_h4.append(ad)
                    if len(ad_h4) == 1:  # 唯一与输入脚本相同的地址
                        h4.append(ad_h4[0])  # 该地址可能为找零地址
                    elif len(ad_h4) == 0:
                        h4.append('没有与输入脚本类型相同的输出')
                    else:
                        h4.append('与输入脚本类型相同的地址不唯一')
                else:
                    h4.append('输出地址只有一个类型')
            else:
                h4.append('输入地址有多个类型')

            ''' h5:对只有两个输出的交易中，找零地址具有更多小数位'''
            ad_h5 = []  # 保存具有更多小数位的地址
            for i in range(len(output_receive)):
                if output_receive[i] % 1000 != 0:
                    ad_h5.append(output_address[i])
            if len(ad_h5) == 0:
                h5.append('收款都是整数')
            elif len(ad_h5) == 1:
                h5.append(ad_h5[0])
            else:
                h5.append('有多个非整数收款地址')

            '''h6:找零地址收款额小于所有输入'''
            if input_count>1:
                ad_h6 = []  # 保存小于所有输入的地址
                smallest = min(intput_sending)  # 所有的输入地址付款额的最小值
                for i in range(len(output_receive)):
                    if output_receive[i] < smallest:
                        ad_h6.append(output_address[i])
                if len(ad_h6) == 0:
                    h6.append('输出地址收款都不小于所有输入')
                elif len(ad_h6) == 1:
                    h6.append(ad_h6[0])
                else:
                    h6.append('有多个地址收款小于所有输入')
            else:
                h6.append('单输入交易')

            '''本文提出的启发式'''
            ad_condidate = []  # 保存输出中未重用的新地址
            for ad in output_address:
                if ad_info[ad_info['address_hash'] == ad]['output_count'].values[0] == 1: #是否为新地址且未重用
                    ad_condidate.append(ad)

            if self_change_tx !=[] and self_change_tx[0]=='self-change交易': #剔除self-change交易
                condition1.append('self-change交易')
                condition2.append('self-change交易')
                condition3.append('self-change交易')
                script_fea1.append(-1)
            elif len(ad_condidate)==0: # 剔除不存在未重用的新地址交易
                condition1.append('输出中不存在未重用的新地址')
                condition2.append('输出中不存在未重用的新地址')
                condition3.append('输出中不存在未重用的新地址')
                script_fea1.append(-1)
            else:
                '''condition1：输入脚本类型相同，唯一相同类型的输出是找零地址'''
                if len(set(input_script)) == 1: # 所有输入脚本是否相同
                    script_fea1.append(1)
                    if len(set(output_script)) > 1:  # 输出地址脚本类型数是否大于1，等于1无论输出有几个地址，脚本判断方法都失效
                        ad_equal = []  # 保存与输入脚本相同的地址
                        input_script_type = input_script[0]  # 输入脚本的类型
                        for ad in ad_condidate:
                            if ad_script_dic[ad] == input_script_type:  # 判断每个输出的类型与输入是否相同
                                ad_equal.append(ad)
                        if len(ad_equal) == 1:  # 唯一与输入脚本相同的地址
                            condition1.append(ad_equal[0])  # 该地址可能为找零地址
                        elif len(ad_equal) == 0:
                            condition1.append('没有与输入脚本相同的输出')
                        else:
                            condition1.append('与输入脚本相同的地址不唯一')
                    else:
                        condition1.append('输出地址只有一个类型')
                else:
                    script_fea1.append(0)
                    condition1.append('输入地址脚本有多个类型')

                '''condition2：唯一不是整数输出的地址是找零地址
                condition3：唯一小于所有输入的地址是找零地址
                '''
                ad_non_integer = []  # 保存收款为非整数的地址
                ad_small = []  # 保存小于所有输入的输出地址
                smallest = min(intput_sending)  # 所有的输入地址付款额的最小值
                for ad in ad_condidate:
                    if (output_dict[ad] % 1000 != 0) and (output_dict_usd[ad] %100!=0):  # h2启发式
                        ad_non_integer.append(ad)
                    if output_dict[ad] < smallest:  # h3启发式
                        ad_small.append(ad)

                '''condition2启发式'''
                if len(ad_non_integer) == 1:  # 唯一非整数收款的地址
                    address_others=list(set(output_address)-set(ad_non_integer))
                    integer_num=0 #统计整数收款地址
                    for ad in address_others:
                        if (output_dict[ad] % 1000 == 0) or (output_dict_usd[ad] %100==0): #其他地址是否为整数收款
                            integer_num+=1
                    if integer_num==len(address_others):
                        condition2.append(ad_non_integer[0])
                    else:
                        condition2.append('其他地址收款并不全为整数')
                elif len(ad_non_integer) == 0:
                    condition2.append('没有非整数收款的输出地址')
                else:
                    condition2.append('有多个非整数收款的输出地址')

                '''condition3启发式'''
                if input_count>1:
                    if len(ad_small) == 1:  # 唯一小于所有输入的地址且收款交易数为1
                        condition3.append(ad_small[0])
                    elif len(ad_small) == 0:
                        condition3.append('没有输出地址收款小于所有输入')
                    else:
                        condition3.append('小于所有输入的输出地址不唯一')
                else:
                    condition3.append('单输入交易')

            '''判断以上不同方法识别出的是否是同一个，如果不是同一个则不标记找零地址'''
            if condition1[-1]==condition2[-1]==condition3[-1]:
                h7.append(condition1[-1])
            else:
                ad_set = set()
                if condition1[-1] in output_address:
                    ad_set = ad_set | {condition1[-1]}
                if condition2[-1] in output_address:
                    ad_set = ad_set | {condition2[-1]}
                if condition3[-1] in output_address:
                    ad_set = ad_set | {condition3[-1]}
                if len(ad_set) == 1:  # ad_set集合长度为1，说明识别的地址是唯一
                    h7.append(list(ad_set)[0])
                elif len(ad_set) == 0: # ad_set集合长度为0，h1-h3都未识别出找零地址
                    h7.append('未重用的新地址不满足条件1-3')
                else:
                    h7.append('产生冲突，条件1-3的识别结果的找零地址不相同')

        else:
            h2.append('只有一个输出地址')
            h3.append('只有一个输出地址')
            h4.append('只有一个输出地址')
            h5.append('只有一个输出地址')
            h6.append('只有一个输出地址')
            script_fea1.append(-1)
            condition1.append('只有一个输出地址')
            condition2.append('只有一个输出地址')
            condition3.append('只有一个输出地址')
            h7.append('只有一个输出地址')
        k+=1
        if k%100==0:
            print('进程{}执行{}次花费时间为：{}s'.format(os.getpid(),k,time.time()-start_time))
    print('执行花费总时间为:',time.time()-start_time)

    '''数据保存'''
    data = pd.DataFrame({'tx_hash': tx_list, 'tx_type': tx_type, 'h2': h2, 'h3': h3, 'h4': h4, 'h5': h5, 'h6': h6,
                         'script_fea1': script_fea1, 'condition1': condition1, 'condition2': condition2,
                         'condition3': condition3, 'h7': h7})
    data = pd.merge(data, tx[['tx_hash', 'block_id']], on='tx_hash', how='left')
    data.to_csv('/Users/lizhihan/Desktop/Bitcoin_data/data_blockchair/result'+str(os.getpid())+'.csv', index=False, encoding='gbk')

# if __name__ == '__main__':
#     tx_list1 = tx_list[:20000]
#     tx_list2 = tx_list[20000:40000]
#     tx_list3 = tx_list[40000:60000]
#     tx_list4 = tx_list[60000:80000]
#     tx_list5 = tx_list[80000:100000]
#     tx_list6 = tx_list[100000:120000]
#     tx_list7 = tx_list[120000:140000]
#     tx_list8 = tx_list[140000:160000]
#     tx_list9 = tx_list[160000:180000]
#     tx_list10 = tx_list[180000:]
#
#     '''创建子进程'''
#     task1_process = mp.Process(target=change_address_identify,args=(tx,input_data,output_data,ad_info,tx_list1))
#     task2_process = mp.Process(target=change_address_identify, args=(tx, input_data, output_data, ad_info, tx_list2))
#     task3_process = mp.Process(target=change_address_identify, args=(tx, input_data, output_data, ad_info, tx_list3))
#     task4_process = mp.Process(target=change_address_identify, args=(tx, input_data, output_data, ad_info, tx_list4))
#     task5_process = mp.Process(target=change_address_identify, args=(tx, input_data, output_data, ad_info, tx_list5))
#     task6_process = mp.Process(target=change_address_identify, args=(tx, input_data, output_data, ad_info, tx_list6))
#     task7_process = mp.Process(target=change_address_identify, args=(tx, input_data, output_data, ad_info, tx_list7))
#     task8_process = mp.Process(target=change_address_identify, args=(tx, input_data, output_data, ad_info, tx_list8))
#     task9_process = mp.Process(target=change_address_identify, args=(tx, input_data, output_data, ad_info, tx_list9))
#     task10_process = mp.Process(target=change_address_identify, args=(tx, input_data, output_data, ad_info, tx_list10))
#
#     '''启动子进程'''
#     task1_process.start()
#     task2_process.start()
#     task3_process.start()
#     task4_process.start()
#     task5_process.start()
#     task6_process.start()
#     task7_process.start()
#     task8_process.start()
#     task9_process.start()
#     task10_process.start()

'''交易、地址数随交易的变化趋势'''


data=pd.read_csv('/Users/lizhihan/Desktop/Bitcoin_data/data_blockchair/result.csv',encoding='gbk')
data_tx_type=data[['block_id','tx_hash','tx_type']]
block_list=list(set(data['block_id'].to_list()))
block_list.sort()

tx_ad_num=pd.DataFrame(columns=['tx_num','ad_num_input','ad_num_output'])
for i in block_list:
    tx_block_i=tx[tx['block_id']<=i]
    input_data_block_i = input_data[input_data['block_id'] <= i]
    output_data_block_i = output_data[output_data['block_id'] <= i]
    tx_num=len(set(tx_block_i['tx_hash'].tolist()))
    ad_num_input=len(set(input_data_block_i['address_hash'].tolist()))
    ad_num_output = len(set(output_data_block_i['address_hash'].tolist()))
    tx_ad_num.loc[i]=[tx_num,ad_num_input,ad_num_output]

tx_ad_num.to_csv('/Users/lizhihan/Desktop/Bitcoin data/data_blockchair/tx_ad_num.csv',index=True)

tx_ad_num=pd.read_csv('/Users/lizhihan/Desktop/Bitcoin data/data_blockchair/tx_ad_num.csv')
fig = plt.figure(figsize=(36,10),dpi=120,frameon=False)
ax1 = fig.add_subplot(1,3,1)
ax1.plot(tx_ad_num['tx_num'])
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.patch.set_facecolor('gray')
ax1.patch.set_alpha(0.1)
plt.xlabel('Block Height',fontsize=25)
plt.ylabel('Number of transaction',fontsize=25)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

# plt.grid(True, linestyle = "--",color = "gray", linewidth = "0.5",axis = 'x')

ax2 = fig.add_subplot(1,3,2)
ax2.plot(tx_ad_num['ad_num_input'])
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.patch.set_facecolor('gray')
ax2.patch.set_alpha(0.1)
plt.xlabel('Block Height',fontsize=25)
plt.ylabel('Number of Input Address',fontsize=25)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

ax3 = fig.add_subplot(1,3,3)
ax3.plot(tx_ad_num['ad_num_output'])
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.patch.set_facecolor('gray')
ax3.patch.set_alpha(0.1)
plt.xlabel('Block Height',fontsize=25)
plt.ylabel('Number of Output Address',fontsize=25)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.savefig('/Users/lizhihan/Desktop/Bitcoin_data/可视化/fig2.png',dpi=200)
plt.show()


'''交易类型分布'''
type_count=pd.DataFrame(columns=['单输出','单输入2输出','多输入2输出','多输入多输出','单输入多输出']) #统计各类型频数
tx_type_distribution=pd.DataFrame(columns=['单输出','单输入2输出','多输入2输出','多输入多输出','单输入多输出']) #统计各类型频率

for i in block_list:
    block_tx_type=data_tx_type[data_tx_type['block_id']<=i]
    tx_type_count=block_tx_type['tx_type'].value_counts()
    tx_type_count=np.array([tx_type_count['单输出'],tx_type_count['单输入2输出'],tx_type_count['多输入2输出'],tx_type_count['多输入多输出'],tx_type_count['单输入多输出']])
    sample=tx_type_count/np.sum(tx_type_count)
    type_count.loc[i]=tx_type_count
    tx_type_distribution.loc[i]=sample #逐行增加


tx_type_distribution=np.around(tx_type_distribution,decimals=3)
columns=['Single Output','Single input and 2 outputs','Multiple inputs and 2 outputs','Multiple inputs and outputs','Single input and multiple outputs']
type_count.columns=columns
tx_type_distribution.columns=columns

type_count.to_csv('/Users/lizhihan/Desktop/Bitcoin_data/data_blockchair/tx_type_count.csv')
tx_type_distribution.to_csv('/Users/lizhihan/Desktop/Bitcoin_data/data_blockchair/tx_type_distribution_csv')


'''画面积图'''
# tx_type_distribution.plot.area(alpha = 0.5)

fig, ax = plt.subplots(figsize=(10,4))
ax.spines['top'].set_visible(False) #去掉上边框
ax.spines['right'].set_visible(False) #去掉右边框
tx_type_distribution.plot(kind='area',ax=ax[0])
# Single_Output=tx_type_distribution['Single Output']
# Single_in_2_out=tx_type_distribution['Single input and 2 outputs']
# Multiple_in_2_out=tx_type_distribution['Multiple inputs and 2 outputs']
# Multiple_in_out=tx_type_distribution['Multiple inputs and outputs']
# Single_in_multi_out=tx_type_distribution['Single input and multiple outputs']
# plt.stackplot(tx_type_distribution.index,Single_Output,Single_in_2_out,Multiple_in_2_out,Multiple_in_out,Single_in_multi_out)
# plt.legend(tx_type_distribution.columns,loc = 10)
# plt.xlim([738690,738789])
# plt.xticks([738699,738719,738739,738759,738779])
plt.show()


'''统计每中启发式方法识别的找零地址数'''

def non_address(change_ads,addresses_set): #获取每个启发式方法中的未成功识别的结果
    non_ad=[]
    for ad in change_ads:
        if ad not in addresses_set:
            non_ad.append(ad)
    return set(non_ad)

def identity_num(data,h,elements_removed): #从识别结果中剔除未识别的部分
    change_ads=set(data[h].tolist())
    change_ads=change_ads-elements_removed
    return change_ads

def change_ad_count(output_data,data,block_list):

    output_ad_set=set(output_data['address_hash'].tolist())
    c1_ad_list=list(set(data['condition1'].tolist()))
    c2_ad_list=list(set(data['condition2'].tolist()))
    c3_ad_list=list(set(data['condition3'].tolist()))
    h7_ad_list=list(set(data['h7'].tolist()))
    h2_ad_list=list(set(data['h2'].tolist()))
    h3_ad_list=list(set(data['h3'].tolist()))
    h4_ad_list=list(set(data['h4'].tolist()))
    h5_ad_list=list(set(data['h5'].tolist()))
    h6_ad_list=list(set(data['h6'].tolist()))

    c1_non_ad=non_address(c1_ad_list,output_ad_set)
    c2_non_ad=non_address(c2_ad_list,output_ad_set)
    c3_non_ad=non_address(c3_ad_list,output_ad_set)
    h7_non_ad=non_address(h7_ad_list,output_ad_set)
    h2_non_ad=non_address(h2_ad_list,output_ad_set)
    h3_non_ad=non_address(h3_ad_list,output_ad_set)
    h4_non_ad=non_address(h4_ad_list,output_ad_set)
    h5_non_ad=non_address(h5_ad_list,output_ad_set)
    h6_non_ad=non_address(h6_ad_list,output_ad_set)

    change_ad_num=pd.DataFrame(columns=['H2','H3','H4','H5','H6','H7','c1','c2','c3'])
    for i in block_list:
        block_tx_type =data[data['block_id']<=i]
        c1=len(identity_num(block_tx_type,'condition1',c1_non_ad))
        c2 = len(identity_num(block_tx_type, 'condition2', c2_non_ad))
        c3 = len(identity_num(block_tx_type, 'condition3', c3_non_ad))
        h7=len(identity_num(block_tx_type,'h7',h7_non_ad))
        h2=len(identity_num(block_tx_type,'h2',h2_non_ad))
        h3=len(identity_num(block_tx_type,'h3',h3_non_ad))
        h4=len(identity_num(block_tx_type,'h4',h4_non_ad))
        h5=len(identity_num(block_tx_type,'h5',h5_non_ad))
        h6=len(identity_num(block_tx_type,'h6',h6_non_ad))
        sample=[h2,h3,h4,h5,h6,h7,c1,c2,c3]
        change_ad_num.loc[i]=sample
    return change_ad_num

data1=data[data['tx_type']=='单输入2输出']
data2=data[data['tx_type']=='多输入2输出']
data3=data[data['tx_type']=='多输入多输出']
data4=data[data['tx_type']=='单输入多输出']

change_ad_num_all=change_ad_count(output_data,data,block_list)
change_ad_num_1=change_ad_count(output_data,data1,block_list)
change_ad_num_2=change_ad_count(output_data,data2,block_list)
change_ad_num_3=change_ad_count(output_data,data3,block_list)
change_ad_num_4=change_ad_count(output_data,data4,block_list)
change_ad_num_all.to_csv('/Users/lizhihan/Desktop/Bitcoin_data/data_blockchair/change_ad_num.csv',index=True)
change_ad_num_1.to_csv('/Users/lizhihan/Desktop/Bitcoin_data/data_blockchair/change_ad_num_单输入2输出.csv',index=True)
change_ad_num_2.to_csv('/Users/lizhihan/Desktop/Bitcoin_data/data_blockchair/change_ad_num_多输入2输出.csv',index=True)
change_ad_num_3.to_csv('/Users/lizhihan/Desktop/Bitcoin_data/data_blockchair/change_ad_num_多输入多输出.csv',index=True)
change_ad_num_4.to_csv('/Users/lizhihan/Desktop/Bitcoin_data/data_blockchair/change_ad_num_单输入多输出.csv',index=True)

'''线图'''
# change_ad_num.plot(grid=True,dpi=120,facecolor='gray')
# plt.figure(dpi=120,facecolor='gray',frameon=False,grid=True)
# plt.plot(change_ad_num)
# plt.xlim([738690,738789])
# plt.xticks([738699,738719,738739,738759,738779])
# plt.xlabel('Continuous Block Height')
# plt.ylabel('Cumulative Number of Change Addresses')
# plt.show()

'''不同方法识别率'''

output_ad_set=set(output_data['address_hash'].tolist())
c1_ad_list=list(set(data['h1'].tolist()))
c2_ad_list=list(set(data['h2'].tolist()))
c3_ad_list=list(set(data['h2'].tolist()))
h7_ad_list=list(set(data['change_address'].tolist()))
h2_ad_list=list(set(data['f1'].tolist()))
h3_ad_list=list(set(data['f2'].tolist()))
h4_ad_list=list(set(data['f3'].tolist()))
h5_ad_list=list(set(data['f4'].tolist()))
h6_ad_list=list(set(data['f5'].tolist()))

c1_non_ad=non_address(c1_ad_list,output_ad_set)
c2_non_ad=non_address(c2_ad_list,output_ad_set)
c3_non_ad=non_address(c3_ad_list,output_ad_set)
h7_non_ad=non_address(h7_ad_list,output_ad_set)
h2_non_ad=non_address(h2_ad_list,output_ad_set)
h3_non_ad=non_address(h3_ad_list,output_ad_set)
h4_non_ad=non_address(h4_ad_list,output_ad_set)
h5_non_ad=non_address(h5_ad_list,output_ad_set)
h6_non_ad=non_address(h6_ad_list,output_ad_set)

def h_identity_rate(data,h,tx_type_freq,elements_removed):
    change_ads=set(data[h].tolist())
    change_ads=list(change_ads-elements_removed) #剔除未识别的部分
    df1 =data[data[h].isin(change_ads)] #得到包含找零地址的部分
    tx_type_count = df1['tx_type'].value_counts()
    sample=[] #保存不同类型识别的找零地址数
    index_type=list(tx_type_count.index)
    if '单输入2输出' in index_type:
        sample.append(tx_type_count['单输入2输出'])
    else:
        sample.append(0)
    if '多输入2输出' in index_type:
        sample.append(tx_type_count['多输入2输出'])
    else:
        sample.append(0)
    if '多输入多输出' in index_type:
        sample.append(tx_type_count['多输入多输出'])
    else:
        sample.append(0)
    if '单输入多输出' in index_type:
        sample.append(tx_type_count['单输入多输出'])
    else:
        sample.append(0)
    tx_type_count = np.array(sample)
    tx_type_count = tx_type_count / tx_type_freq
    return np.around(tx_type_count,decimals=3)


tx_type_freq=data_tx_type['tx_type'].value_counts()
tx_type_freq=np.array([tx_type_freq['单输入2输出'],tx_type_freq['多输入2输出'],tx_type_freq['多输入多输出'],tx_type_freq['单输入多输出']])
h_ours_tx=h_identity_rate(data,'change_address',tx_type_freq,h7_non_ad)
h_f1_tx=h_identity_rate(data,'f1',tx_type_freq,h2_non_ad)
h_f2_tx=h_identity_rate(data,'f2',tx_type_freq,h3_non_ad)
h_f3_tx=h_identity_rate(data,'f3',tx_type_freq,h4_non_ad)
h_f4_tx=h_identity_rate(data,'f4',tx_type_freq,h5_non_ad)
h_f5_tx=h_identity_rate(data,'f5',tx_type_freq,h6_non_ad)
h_tx_type=pd.DataFrame([h_ours_tx,h_f1_tx,h_f2_tx,h_f3_tx,h_f4_tx,h_f5_tx],columns=['Single input and 2 outputs', 'Multiple inputs and 2 outputs', 'Multiple inputs and outputs','Single input and multiple outputs'],index=['H7','H2','H3','H4','H5','H6'])

#画柱状图
h_tx_type.plot(kind='bar',colormap='Blues_r')
plt.show()

'''分析每种方法重合度与差异'''

start_time = time.time()
output_addresses = set(output_data['address_hash'].tolist())
h7_ad_list = data['change_address'].tolist()
h2_ad_list = data['f1'].tolist()
h3_ad_list = data['f2'].tolist()
h4_ad_list = data['f3'].tolist()
h5_ad_list = data['f4'].tolist()
h6_ad_list = data['f5'].tolist()

h7_vs_f1_overlap = []
h7_vs_f1_not_overlap = []
h7_vs_f2_overlap = []
h7_vs_f2_not_overlap = []
h7_vs_f3_overlap = []
h7_vs_f3_not_overlap = []
h7_vs_f4_overlap = []
h7_vs_f4_not_overlap = []
h7_vs_f5_overlap = []
h7_vs_f5_not_overlap = []
h7_change_ad=identity_num(data,'change_address',h7_non_ad)
for i in range(len(h7_ad_list)):
    h7_ad = h7_ad_list[i]
    h2_ad = h2_ad_list[i]
    h3_ad = h3_ad_list[i]
    h4_ad = h4_ad_list[i]
    h5_ad = h5_ad_list[i]
    h6_ad = h6_ad_list[i]
    if h2_ad in output_addresses:
        if h2_ad == h7_ad:
            h7_vs_f1_overlap.append(h7_ad)
        elif h2_ad not in h7_change_ad: #h7未覆盖的找零地址
            if h7_ad not in h7_change_ad: #h7_ad也不是找零地址
                h7_vs_f1_not_overlap.append(h7_ad)
            else:
                h7_vs_f1_not_overlap.append('识别结果不同') #与h7识别结果不同
    if h3_ad in output_addresses:
        if h3_ad == h7_ad:
            h7_vs_f2_overlap.append(h7_ad)
        elif h3_ad not in h7_change_ad: #h7未覆盖的找零地址
            if h7_ad not in h7_change_ad: #h7_ad也不是找零地址
                h7_vs_f2_not_overlap.append(h7_ad)
            else:
                h7_vs_f2_not_overlap.append('识别结果不同')
    if h4_ad in output_addresses:
        if h4_ad == h7_ad:
            h7_vs_f3_overlap.append(h7_ad)
        elif h4_ad not in h7_change_ad: #h7未覆盖的找零地址
            if h7_ad not in h7_change_ad: #h7_ad也不是找零地址
                h7_vs_f3_not_overlap.append(h7_ad)
            else:
                h7_vs_f3_not_overlap.append('识别结果不同')
    if h5_ad in output_addresses:
        if h5_ad == h7_ad:
            h7_vs_f4_overlap.append(h7_ad)
        elif h5_ad not in h7_change_ad: #h7未覆盖的找零地址
            if h7_ad not in h7_change_ad: #h7_ad也不是找零地址
                h7_vs_f4_not_overlap.append(h7_ad)
            else:
                h7_vs_f4_not_overlap.append('识别结果不同')
    if h6_ad in output_addresses:
        if h6_ad == h7_ad:
            h7_vs_f5_overlap.append(h7_ad)
        elif h6_ad not in h7_change_ad: #h7未覆盖的找零地址
            if h7_ad not in h7_change_ad: #h7_ad也不是找零地址
                h7_vs_f5_not_overlap.append(h7_ad)
            else:
                h7_vs_f5_not_overlap.append('识别结果不同')
    if i % 10000 == 0:
        print('已花费{}秒，剩余{}个交易'.format(time.time() - start_time, len(h7_ad_list) - i))

# print(len(h7_vs_f1_overlap),len(h7_vs_f1_not_overlap))
# print(len(h7_vs_f2_overlap),len(h7_vs_f2_not_overlap))
# print(len(h7_vs_f3_overlap),len(h7_vs_f3_not_overlap))
# print(len(h7_vs_f4_overlap),len(h7_vs_f4_not_overlap))
# print(len(h7_vs_f5_overlap),len(h7_vs_f5_not_overlap))

def frequency(f_not_covered):
    index_type=f_not_covered.index
    sample_freq=[]
    if 'self-change交易' in index_type:
        sample_freq.append(f_not_covered['self-change交易'])
    else:
        sample_freq.append(0)
    if '输出中不存在未重用的新地址' in index_type:
        sample_freq.append(f_not_covered['输出中不存在未重用的新地址'])
    else:
        sample_freq.append(0)
    if '识别结果不同' in index_type:
        sample_freq.append(f_not_covered['识别结果不同'])
    else:
        sample_freq.append(0)
    if '未重用的新地址不满足条件1-3' in index_type:
        sample_freq.append(f_not_covered['未重用的新地址不满足条件1-3'])
    else:
        sample_freq.append(0)
    if '产生冲突，条件1-3的识别结果的找零地址不相同' in index_type:
        sample_freq.append(f_not_covered['产生冲突，条件1-3的识别结果的找零地址不相同'])
    else:
        sample_freq.append(0)
    return np.array(sample_freq)

f1_not_covered=pd.Series(h7_vs_f1_not_overlap).value_counts()
f2_not_covered=pd.Series(h7_vs_f2_not_overlap).value_counts()
f3_not_covered=pd.Series(h7_vs_f3_not_overlap).value_counts()
f4_not_covered=pd.Series(h7_vs_f4_not_overlap).value_counts()
f5_not_covered=pd.Series(h7_vs_f5_not_overlap).value_counts()

df_not_covered=pd.DataFrame(columns=['self-change交易','输出中不存在未重用的新地址','识别结果不同','未重用的新地址不满足条件1-3','产生冲突，条件1-3的识别结果的找零地址不相同'])

df_not_covered.loc['f1_not_covered']=frequency(f1_not_covered)/np.sum(frequency(f1_not_covered))
df_not_covered.loc['f2_not_covered']=frequency(f2_not_covered)/np.sum(frequency(f2_not_covered))
df_not_covered.loc['f3_not_covered']=frequency(f3_not_covered)/np.sum(frequency(f3_not_covered))
df_not_covered.loc['f4_not_covered']=frequency(f4_not_covered)/np.sum(frequency(f4_not_covered))
df_not_covered.loc['f5_not_covered']=frequency(f5_not_covered)/np.sum(frequency(f5_not_covered))

df_not_covered.plot(kind='bar',stacked=True)
plt.show()

'''将同一个交易的所有输入地址进行聚集'''
tx_adset_dict = {}  # 存储每个交易的输入地址
start_time=time.time()
count_rest=0
count_total=len(tx_list)
for t in tx_list:
    tx_input = input_data[input_data['tx_hash'] == t]  # 每个交易所有输入
    input_ad_set = set(tx_input['address_hash'].tolist())
    tx_adset_dict[t] = input_ad_set
    count_rest+=1
    print('剩余{}个交易'.format(count_total-count_rest))

'''对交易编码，并将输入地址有交集的交易编码进行统一'''
def tx_encoding(tx_adset_dict): #交易哈希，交易输入，截止的区块高度
    t_list=list(tx_adset_dict.keys())
    txin_user_id = {t_list[i]: i for i in range(len(t_list))}  # 对交易进行编码,即交易编码看作是输入用户的ID
    l1 = len(t_list)
    '''将任意两个输入有交集的交易编码进行统一'''
    while l1>0:
        tx_hash_l1=t_list[l1-1]
        l2=l1-1
        while l2>0:
            tx_hash_l2 =t_list[l2-1]
            if len(tx_adset_dict[tx_hash_l1] & tx_adset_dict[tx_hash_l2])!=0: #查看任意两个交易tx_hash_l1与tx_hash_l2输入地址集是否有交集
                txin_user_id[tx_hash_l2]=txin_user_id[tx_hash_l1] #有交集，则将交易tx_hash_l2的编码统一为tx_hash_l1的编码
                tx_adset_dict[tx_hash_l1]=tx_adset_dict[tx_hash_l1]|tx_adset_dict[tx_hash_l2] #输入地址有共同交集的，则将输入地址进行合并
                t_list.remove(tx_hash_l2)
                l1-=1
            l2-=1
            print('进程{},外层还剩下{}次'.format(os.getpid(),l1))
        l1-=1
    return txin_user_id

def run1(tx_adset,k):
    tx_adset_k=tx_adset[tx_adset['block_id']<=k]
    tx_list=tx_adset_k['tx_hash'].tolist()
    ad_set=tx_adset_k['adset'].tolist()
    tx_adset_dict={tx_list[i]:ad_set[i] for i in range(len(tx_list))} #每个交易的输入地址集合
    tx_coding_result=tx_encoding(tx_adset_dict) #基于交易的编码
    tx_list2=list(tx_coding_result.keys())
    user_id=[tx_coding_result[t] for t in tx_list2]
    input_user_id=pd.DataFrame({'tx_hash':tx_list2,'user_id':user_id})
    input_user_id.to_csv('C:\\Users\\lenovo\\Desktop\\input_user_id_'+str(k)+'.csv',index=False)
    return

tx_adset=pd.read_csv('/Users/lizhihan/Desktop/Bitcoin data/data_blockchair/tx_adset.csv')
tx_adset['adset']=tx_adset['adset'].apply(lambda x:ast.literal_eval(re.search('({.+})', x).group(0))) #将字符串类型的集合转换为集合
tx_adset=pd.merge(tx_adset,tx,on='tx_hash',how='left')
tx_adset=tx_adset[['tx_hash','adset','block_id']]
height_k=[738699,738709,738719,738729,738739,738749,738759,738769,738779,738789]


if __name__ == '__main__':
    for k in height_k:
        task_process=mp.Process(target=run1,args=(tx_adset,k))
        task_process.start()



'''实体识别：找零地址启发式+多输入启发式'''

def multi_input_address_clustering(input_data,output_data,txin_user_id): #多输入启发式聚类
    ad_user_id = {}  # 存储每个ad的用户编码
    input_addresses = list(set(input_data['address_hash'].tolist()))
    output_addresses = list(set(output_data['address_hash'].tolist()))
    ad_list = list(set(input_addresses + output_addresses))

    '''对输入地址编码'''
    input_data = pd.merge(input_data, txin_user_id, on='tx_hash', how='left')
    userid_list = input_data['user_id'].tolist()
    input_addresses = input_data['address_hash'].tolist()
    for i in range(len(input_addresses)):
        ad_user_id[input_addresses[i]] = userid_list[i]  # 对每个输入地址进行id编码

    '''对输出中的剩下地址进行编码'''
    ad_list2 = set(ad_list) - set(list(ad_user_id.keys()))  # 获取剩下还未编码的地址
    id = 0
    for ad in ad_list2:
        ad_user_id[ad] = txin_user_id.shape[0] + id
        id += 1
    input_data.drop(['user_id'], axis=1, inplace=True)  # 删除字段'user_id'，便于下次计算
    return ad_user_id

def change_ad_h(input_data,output_data,tx_change_ad_dict,txin_user_id): #找零地址启发式聚类
    ad_user_id={} #存储每个ad的用户编码
    start_time=time.time()
    print('执行时间：', time.strftime('%Y-%m-%d %A %H:%M:%S', time.localtime()))
    input_addresses=list(set(input_data['address_hash'].tolist()))
    output_addresses=list(set(output_data['address_hash'].tolist()))
    ad_list=list(set(input_addresses+output_addresses))

    '''对找零地址编码：将输入中包含找零地址的交易输入与以及找零地址对应的交易同一个为一个实体'''
    t_list = txin_user_id['tx_hash'].tolist()
    user_id = txin_user_id['user_id'].tolist()
    txin_user_id = {t_list[i]: user_id[i] for i in range(len(t_list))}
    k = 0
    for key in tx_change_ad_dict.keys():  # 遍历有找零地址的所有交易
        address = tx_change_ad_dict[key]  # 获取交易的找零地址
        if address in input_addresses:  # 找零地址是否参与过付款
            t_list1 = list(set(input_data[input_data['address_hash'] == address]['tx_hash'].tolist()))  # 获取参与付款的所有交易
            for t in t_list1:
                txin_user_id[t] = txin_user_id[key]  # 将参与付款的所有交易id统一为key的编码
        else:
            ad_user_id[address] = txin_user_id[key]  # 将找零地址与对应交易的用户id统一
        k += 1
        if k % 1000 == 0:
            print('已花费{}秒，剩余{}个交易'.format(time.time() - start_time, len(tx_change_ad_dict.keys()) - k))

    '''对输入地址编码'''
    txin_user_id=pd.DataFrame({'tx_hash':t_list,'user_id':user_id})
    input_data = pd.merge(input_data, txin_user_id, on='tx_hash', how='left')
    userid_list = input_data['user_id'].tolist()
    input_addresses = input_data['address_hash'].tolist()
    for i in range(len(input_addresses)):
        ad_user_id[input_addresses[i]] = userid_list[i]  # 对每个输入地址进行id编码

    '''对输出中的剩下地址进行编码'''
    ad_list2=set(ad_list)-set(list(ad_user_id.keys())) #获取剩下还未编码的地址
    id=0
    for ad in ad_list2:
        ad_user_id[ad]=len(t_list)+id
        id+=1
    input_data.drop(['user_id'],axis=1,inplace=True) #删除字段'user_id'，便于下次计算
    return ad_user_id


'''多输入启发式+找零地址启发式'''

height_k=[738699,738709,738719,738729,738739,738749,738759,738769,738779,738789]

clustering_result=pd.DataFrame(columns=['H1','H2','H3','H4','H5','H6','H7'])

for height in height_k:
    tx_coding_result = pd.read_csv(
        '/Users/lizhihan/Desktop/data_server/input_user_id_' + str(height) + '.csv')

    data_height=data[data['block_id']<=height]
    input_data_height = input_data[input_data['block_id'] <= height]
    output_data_height = output_data[output_data['block_id'] <= height]

    tx_list = data_height['tx_hash'].tolist()
    output_addresses = set(output_data_height['address_hash'].tolist())
    h7_change_ad = data_height['h7'].tolist()
    h2_change_ad = data_height['h2'].tolist()
    h3_change_ad = data_height['h3'].tolist()
    h4_change_ad = data_height['h4'].tolist()
    h5_change_ad = data_height['h5'].tolist()
    h6_change_ad = data_height['h6'].tolist()

    tx_change_ad_h7_dict = {tx_list[i]: h7_change_ad[i] for i in range(len(tx_list)) if
                              h7_change_ad[i] in output_addresses}
    tx_change_ad_h2_dict = {tx_list[i]: h2_change_ad[i] for i in range(len(tx_list)) if
                            h2_change_ad[i] in output_addresses}
    tx_change_ad_h3_dict = {tx_list[i]: h3_change_ad[i] for i in range(len(tx_list)) if
                            h3_change_ad[i] in output_addresses}
    tx_change_ad_h4_dict = {tx_list[i]: h4_change_ad[i] for i in range(len(tx_list)) if
                            h4_change_ad[i] in output_addresses}
    tx_change_ad_h5_dict = {tx_list[i]: h5_change_ad [i] for i in range(len(tx_list)) if
                            h5_change_ad [i] in output_addresses}
    tx_change_ad_h6_dict = {tx_list[i]: h6_change_ad[i] for i in range(len(tx_list)) if
                            h6_change_ad[i] in output_addresses}

    multi_input_result=multi_input_address_clustering(input_data_height,output_data_height,tx_coding_result) #多输入启发式聚类结果
    h7_clustering_result=change_ad_h(input_data_height,output_data_height,tx_change_ad_h7_dict,tx_coding_result) #找零地址启发式
    h2_clustering_result=change_ad_h(input_data_height,output_data_height,tx_change_ad_h2_dict,tx_coding_result)
    h3_clustering_result=change_ad_h(input_data_height,output_data_height,tx_change_ad_h3_dict,tx_coding_result)
    h4_clustering_result=change_ad_h(input_data_height,output_data_height,tx_change_ad_h4_dict,tx_coding_result)
    h5_clustering_result=change_ad_h(input_data_height,output_data_height,tx_change_ad_h5_dict,tx_coding_result)
    h6_clustering_result=change_ad_h(input_data_height,output_data_height,tx_change_ad_h6_dict,tx_coding_result)

    clustering_result.loc[height]=[len(set(multi_input_result.values())),len(set(h2_clustering_result.values())),len(set(h3_clustering_result.values())),len(set(h4_clustering_result.values())),len(set(h5_clustering_result.values())),len(set(h6_clustering_result.values())),len(set(h7_clustering_result.values()))]

    print({'H1:{},H2:{},H3:{},H4:{},H5:{},H6:{},本文提出的启发式H7:{}'.format(len(set(multi_input_result.values())),len(set(h2_clustering_result.values())),len(set(h3_clustering_result.values())),len(set(h4_clustering_result.values())),len(set(h5_clustering_result.values())),len(set(h6_clustering_result.values())),len(set(h7_clustering_result.values())))})

clustering_result.to_csv('/Users/lizhihan/Desktop/Bitcoin_data/data_blockchair/clustering_result.csv')

'''地址减少率'''

def ad_reduce_rate(clustering_result,input_data,output_data):
    r=pd.DataFrame(columns=['H1','H2','H3','H4','H5','H6','H7'])
    for height in clustering_result.index:
        ad_num=len(set(input_data[input_data['block_id']<=height]['address_hash'].tolist()))+len(set(output_data[output_data['block_id']<=height]['address_hash'].tolist()))
        h1_cluster_num=clustering_result.loc[height]['H1']
        h2_cluster_num = clustering_result.loc[height]['H2']
        h3_cluster_num = clustering_result.loc[height]['H3']
        h4_cluster_num = clustering_result.loc[height]['H4']
        h5_cluster_num = clustering_result.loc[height]['H5']
        h6_cluster_num = clustering_result.loc[height]['H6']
        h7_cluster_num = clustering_result.loc[height]['H7']
        r.loc[height]=(ad_num-np.array([h1_cluster_num,h2_cluster_num,h3_cluster_num,h4_cluster_num,h5_cluster_num,h6_cluster_num,h7_cluster_num]))/ad_num
    return r

clustering_result=pd.read_csv('/Users/lizhihan/Desktop/Bitcoin_data/data_blockchair/clustering_result.csv',index_col=0)
r=ad_reduce_rate(clustering_result,input_data,output_data)
r.to_csv('/Users/lizhihan/Desktop/Bitcoin_data/data_blockchair/地址减少率.csv')

'''找零地址与区块之间的回归关系'''
import statsmodels.api as sm
from statsmodels.formula.api import ols

change_ad_num=pd.read_excel('/Users/lizhihan/Desktop/Bitcoin_data/data_blockchair/change_ad_num.xls')
lm_h7=ols('H7~block_num',data=change_ad_num).fit()
lm_h2=ols('H2~block_num',data=change_ad_num).fit()
lm_h3=ols('H3~block_num',data=change_ad_num).fit()
lm_h4=ols('H4~block_num',data=change_ad_num).fit()
lm_h5=ols('H5~block_num',data=change_ad_num).fit()
lm_h6=ols('H6~block_num',data=change_ad_num).fit()

print(lm_h2.summary())
print(lm_h3.summary())
print(lm_h4.summary())
print(lm_h5.summary())
print(lm_h6.summary())
print(lm_h7.summary())















