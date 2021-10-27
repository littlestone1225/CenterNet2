import os,sys,argparse
#create on 20211001

# Read config_file
aoi_dir = '/home/aoi/AOI_PCB_CenterNet2/ctr2_pcb/previous_stuff'
sys.path.append(os.path.join(aoi_dir, "config"))
from config import read_config_yaml, write_config_yaml
config_file = os.path.join(aoi_dir, 'config/config.yaml')
config = read_config_yaml(config_file)

def metric_to_dict(line_list):
    data={}
    for line in line_list:
        ele_list = line.replace(':',',').split(',')
        if len(ele_list)<16:continue
        model_name = ele_list[1]
        data[model_name]={}
        data[model_name]['bridge']=int(ele_list[3])
        data[model_name]['empty']=int(ele_list[5])
        data[model_name]['score_threshold']=float(ele_list[7])
        data[model_name]['critical recall']=float(ele_list[9])
        data[model_name]['normal recall']=float(ele_list[11])
        data[model_name]['fp rate']=float(ele_list[15])
    return data

def fp_rate_filter(dictionary, fpr_base):
    remove_list = []
    range_ = fpr_base*0.1*0.5
    upper_limit = fpr_base+range_
    for model in dictionary:
        if dictionary[model]['fp rate']>upper_limit:remove_list.append(model)
    for name in remove_list:del dictionary[name]
        
    return dictionary

def criticalLoss_filter(inputs, top_k = 5):
    if isinstance(inputs, dict):
        if len(inputs)<top_k:return inputs
        dicts = sorted(inputs.items(), key=lambda d: d[1]['critical recall'], reverse=True)
    
        cri_recall_value = dicts[0][1]['critical recall']
        topk_dicts = list(filter(lambda n: n[1]['critical recall']==cri_recall_value, dicts))
    
        if len(topk_dicts)<top_k:topk_dicts = dicts[:top_k] #at least get the highest 5
        return dict(topk_dicts)
        
    elif isinstance(inputs, list):
        if len(inputs)<top_k:return inputs
        lists = sorted(inputs, key=lambda d: d['critical_recall'], reverse=True)

        cri_recall_value = lists[0]['critical_recall']
        topk_lists = list(filter(lambda n: n['critical_recall']==cri_recall_value, lists))
    
        if len(topk_lists)<top_k:topk_lists = lists[:top_k] #at least get the highest 5
        return topk_lists   

def bridge_filter(inputs):
    if isinstance(inputs, dict):
        dicts = sorted(inputs.items(), key=lambda d: d[1]['bridge'])
        best_model_dict = dicts[:1] #get the least one
        return dict(best_model_dict)
    elif isinstance(inputs, list):
        lists = sorted(inputs, key=lambda d: d['label_fn_dict']['bridge'])
        best_model_list = lists[:1] #get the least one
        return best_model_list

def select_model(model_metrics, fpr=0.6, val=False, top_k = 5):
    metrics_dict = {}
    model_list = []
    out_json_path = config['out_json_path']
    exp_name = config['model_file_dir'][config['model_file_dir'].rfind('/')+1:].replace('.yaml','')
    inference_result_dir = os.path.join(out_json_path, exp_name)

    if val:txt_name = 'val_results.txt'
    else:txt_name = 'test_results.txt'
    record_path = os.path.join(inference_result_dir, txt_name)
    print('record_path: ', record_path)
    
    f = open(record_path, 'r')
    lines = f.readlines()
    metrics_dict = metric_to_dict(lines)
    print('metrics_dict: ', metrics_dict)
    metrics_dict1 = fp_rate_filter(metrics_dict, fpr)
    print('metrics_dict1: ', metrics_dict1)
    metrics_dict2 = criticalLoss_filter(metrics_dict1, top_k = top_k)
    filtered_dict = metrics_dict2
    print('1_filtered_dict: ', filtered_dict)
    
    if not val:
        metrics_dict3 = bridge_filter(metrics_dict2)
        filtered_dict = metrics_dict3
        print('2_filtered_dict: ', filtered_dict)
    
    model_list = modeldict_tolist(filtered_dict)
    print('model_list: ', model_list)
    model_list_pth = [x+'.pth' for x in model_list]
    print('model_list_pth: ', model_list_pth)
    
    if not val:
        print('3_filtered_dict: ', filtered_dict)
        for model in filtered_dict:
            print('best_model: ',model)
            best_conf = filtered_dict[model]['score_threshold']
        return model_list_pth, best_conf
    else:return model_list_pth

def select_best_model_aifs(eval_result_list, top_k):
    sup_model_list = criticalLoss_filter(eval_result_list, top_k = top_k)
    best_model_list = bridge_filter(sup_model_list)
    
    return best_model_list[0]['model_iter']

def modeldict_tolist(dicts):
    model_list = []
    for model in dicts:
        model_list.append(model)
    return model_list

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description='xxx')
    parser.add_argument('--fpr',type=float,default=0.6,help='(default=%(default)d)')
    parser.add_argument('--path',default='/home/aoi/AOI_PCB_CenterNet2/ctr2_pcb/th_record',type=str,help='(default=%(default)s)')
    parser.add_argument('--txt',default='v6ex_aug_bs8_baseline_sat_norpre04_multith_20211001.txt',type=str,help='(default=%(default)s)')
    #parser.add_argument('--lr',default=0.05,type=float,required=False,help='(default=%(default)f)')
    args=parser.parse_args()
    
    metric_txt_path = os.path.join(args.path, args.txt)
    metrics_dict = {}
    f = open(metric_txt_path, 'r')
    lines = f.readlines()
    metrics_dict = metric_to_dict(lines)
    metrics_dict1 = fp_rate_filter(metrics_dict, args.fpr)
    metrics_dict2 = criticalLoss_filter(metrics_dict1, top_k = 5)
    print('metrics_dict2: ', metrics_dict2)
    """
    metrics_dict3 =  bridge_filter(metrics_dict2)
    print('metrics_dict3: ', metrics_dict3)
    model_list = modeldict_tolist(metrics_dict3)
    print('model_list: ', model_list)
    """