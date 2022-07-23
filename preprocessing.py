import pandas as pd
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import warnings
warnings.filterwarnings('ignore')

def preprocessing(read_path,save_path):

    tx=pd.read_csv(read_path+"/blockchair_bitcoin_transactions_20220531.tsv", delimiter = '\t')
    input_data=pd.read_csv(read_path+"/blockchair_bitcoin_inputs_20220531.tsv", delimiter = '\t')
    output_data=pd.read_csv(read_path+"/blockchair_bitcoin_outputs_20220531.tsv", delimiter = '\t')

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

    tx_list1=tx['tx_hash'].tolist()
    tx_list2=input_data['tx_hash'].tolist()
    tx_list3=output_data['tx_hash'].tolist()
    tx_list=list(set(tx_list1)&set(tx_list2)&set(tx_list3)) #交易取3个数据的交集
    tx_df=pd.DataFrame({'tx_hash':tx_list})

    tx=pd.merge(tx_df,tx,on='tx_hash',how='left')
    input_data=pd.merge(tx_df,input_data,on='tx_hash',how='left')
    output_data=pd.merge(tx_df,output_data,on='tx_hash',how='left')

    tx.to_csv(save_path+'/transactions_data.csv',index=False)
    input_data.to_csv(save_path + '/input_data.csv',index=False)
    output_data.to_csv(save_path + '/output_data.csv',index=False)

    ad_list = list(set(input_data['address_hash'].tolist() + output_data['address_hash'].tolist()))
    print('Total number of transactions：', len(tx_list))
    print('Total number of addresses：', len(ad_list))
    print('Total number of input addresses：', len(set(input_data['address_hash'].tolist())))
    print('Total number of output addresses：', len(set(output_data['address_hash'].tolist())))

    return


if __name__ == '__main__':
    read_path=''
    save_path=''
    preprocessing(read_path,save_path)




