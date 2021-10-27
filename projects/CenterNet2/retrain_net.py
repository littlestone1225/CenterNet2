import logging
import os,sys
from collections import OrderedDict
import torch
from torch.nn.parallel import DistributedDataParallel
import time
import datetime
import json

from fvcore.common.timer import Timer
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
)
from detectron2.engine import default_argument_parser, default_setup, launch

from detectron2.evaluation import (
    COCOEvaluator,
    LVISEvaluator,
    inference_on_dataset,
    print_csv_format,
)
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.events import (
    CommonMetricPrinter,
    EventStorage,
    JSONWriter,
    TensorboardXWriter,
)
from detectron2.modeling.test_time_augmentation import GeneralizedRCNNWithTTA
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.data.build import build_detection_train_loader

from centernet.config import add_centernet_config
from centernet.data.custom_build_augmentation import build_custom_augmentation
from detectron2.data.datasets import register_coco_instances

aoi_dir_name = os.getenv('AOI_DIR_NAME')
assert aoi_dir_name != None, "Environment variable AOI_DIR_NAME is None"
current_dir = os.path.dirname(os.path.abspath(__file__))
idx = current_dir.find(aoi_dir_name) + len(aoi_dir_name)
aoi_dir = current_dir[:idx]
"""
sys.path.append(os.path.join(aoi_dir, "ctr2_pcb/previous_stuff"))
from validation.validation import gen_inference_json
from validation.model_select import select_model
from validation.multi_th_eval import evaluation_through_model_iter
from data_map.pcb_retrain_v5  import retrain_data_map
"""
sys.path.append(os.path.join(aoi_dir, "validation"))
from validation import validate_new_models_on_val, validate_new_models_on_test_and_fn, validate_old_model_on_test_and_fn
from evaluation import evaluate_new_models_on_val, evaluate_new_models_on_val, \
    evaluate_new_models_on_test_and_fn_by_fp_rate, evaluate_old_model_on_test_and_fn_by_fp_rate
from tools.model_select import select_best_model_aifs #select_model
#from model_select import select_model
#from multi_th_eval import evaluation_through_model_iter

logger = logging.getLogger("detectron2")

def do_test(cfg, model):
    results = OrderedDict()
    for dataset_name in cfg.DATASETS.TEST:
        mapper = None if cfg.INPUT.TEST_INPUT_TYPE == 'default' else \
            DatasetMapper(
                cfg, False, augmentations=build_custom_augmentation(cfg, False))
        data_loader = build_detection_test_loader(cfg, dataset_name, mapper=mapper)
        output_folder = os.path.join(
            cfg.OUTPUT_DIR, "inference_{}".format(dataset_name))
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type

        if evaluator_type == "lvis":
            evaluator = LVISEvaluator(dataset_name, cfg, True, output_folder)
        elif evaluator_type == 'coco':
            evaluator = COCOEvaluator(dataset_name, cfg, True, output_folder)
        else:
            assert 0, evaluator_type
            
        results[dataset_name] = inference_on_dataset(
            model, data_loader, evaluator)
        if comm.is_main_process():
            logger.info("Evaluation results for {} in csv format:".format(
                dataset_name))
            print_csv_format(results[dataset_name])
    if len(results) == 1:
        results = list(results.values())[0]
    return results

def do_train(cfg, model, resume=False):
    model.train()
    optimizer = build_optimizer(cfg, model)
    scheduler = build_lr_scheduler(cfg, optimizer)

    checkpointer = DetectionCheckpointer(
        model, cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler
    )

    start_iter = (
        checkpointer.resume_or_load(
            cfg.MODEL.WEIGHTS, resume=resume,
            ).get("iteration", -1) + 1
    )
    if cfg.SOLVER.RESET_ITER:
        logger.info('Reset loaded iteration. Start training from iteration 0.')
        start_iter = 0
    max_iter = cfg.SOLVER.MAX_ITER if cfg.SOLVER.TRAIN_ITER < 0 else cfg.SOLVER.TRAIN_ITER

    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter
    )

    writers = (
        [
            CommonMetricPrinter(max_iter),
            JSONWriter(os.path.join(cfg.OUTPUT_DIR, "metrics.json")),
            TensorboardXWriter(cfg.OUTPUT_DIR),
        ]
        if comm.is_main_process()
        else []
    )

    mapper = DatasetMapper(cfg, True) if cfg.INPUT.CUSTOM_AUG == '' else \
        DatasetMapper(cfg, True, augmentations=build_custom_augmentation(cfg, True))
    if cfg.DATALOADER.SAMPLER_TRAIN in ['TrainingSampler', 'RepeatFactorTrainingSampler']:
        data_loader = build_detection_train_loader(cfg, mapper=mapper)
    else:
        from centernet.data.custom_dataset_dataloader import  build_custom_train_loader
        data_loader = build_custom_train_loader(cfg, mapper=mapper)


    logger.info("Starting training from iteration {}".format(start_iter))
    with EventStorage(start_iter) as storage:
        step_timer = Timer()
        data_timer = Timer()
        start_time = time.perf_counter()
        for data, iteration in zip(data_loader, range(start_iter, max_iter)):
            data_time = data_timer.seconds()
            storage.put_scalars(data_time=data_time)
            step_timer.reset()
            iteration = iteration + 1
            storage.step()
            loss_dict = model(data)

            losses = sum(
                loss for k, loss in loss_dict.items())
            assert torch.isfinite(losses).all(), loss_dict

            loss_dict_reduced = {k: v.item() \
                for k, v in comm.reduce_dict(loss_dict).items()}
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            if comm.is_main_process():
                storage.put_scalars(
                    total_loss=losses_reduced, **loss_dict_reduced)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            storage.put_scalar(
                "lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)

            step_time = step_timer.seconds()
            storage.put_scalars(time=step_time)
            data_timer.reset()
            scheduler.step()

            if (
                cfg.TEST.EVAL_PERIOD > 0
                and iteration % cfg.TEST.EVAL_PERIOD == 0
                and iteration != max_iter
            ):
                do_test(cfg, model)
                comm.synchronize()

            if iteration - start_iter > 5 and \
                (iteration % 20 == 0 or iteration == max_iter):
                for writer in writers:
                    writer.write()
            periodic_checkpointer.step(iteration)

        total_time = time.perf_counter() - start_time
        logger.info(
            "Total training time: {}".format(
                str(datetime.timedelta(seconds=int(total_time)))))

def setup(args, yaml_config):
    """
    Create configs and perform basic setups.
    """

    cfg = get_cfg()
    add_centernet_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    if args.aifs:cfg.OUTPUT_DIR = yaml_config['centernet2_model_output_dir']
    else:cfg.OUTPUT_DIR = yaml_config['model_file_dir']
    if 'centernet2_old_model_file_path' in yaml_config:
        cfg.MODEL.WEIGHTS = yaml_config['centernet2_old_model_file_path']
    if ('retrain_task' in yaml_config) and (not args.aifs):cfg.DATASETS.TRAIN = retrain_data_map(yaml_config['retrain_task'])
    if '/auto' in cfg.OUTPUT_DIR:
        file_name = os.path.basename(args.config_file)[:-5]
        cfg.OUTPUT_DIR = cfg.OUTPUT_DIR.replace('/auto', '/{}'.format(file_name))
        logger.info('OUTPUT_DIR: {}'.format(cfg.OUTPUT_DIR))
    cfg.freeze()
    default_setup(cfg, args)
    return cfg

def performance_review(cfg, yaml_config, args, model_file_dir):
    test_dat_type = 'val'
    gen_inference_json(cfg, test_dat_type)    #gen val json
    model_metrics = evaluation_through_model_iter(score_threshold = args.max_conf_th, min_thresh = args.min_conf_th, test_dat=test_dat_type)   #eval val data 
    model_val_list = select_model('', fpr=args.fpr, val=True, top_k = args.top_k) #pick top k model
    logger.info("Model_val_list:\n{}".format(model_val_list))
    
    test_dat_type = 'test'
    gen_inference_json(cfg, test_dat_type, model_val_list) #gen test json   
    model_metrics = evaluation_through_model_iter(score_threshold = args.max_conf_th, \
        min_thresh = args.min_conf_th, test_dat=test_dat_type, model_list = model_val_list) #eval test data
    best_model, best_conf_th = select_model(model_metrics, fpr=args.fpr, val=False, top_k = args.top_k) #choose best model
    best_model_path = os.path.join(model_file_dir, best_model[0]) 
    
    #To-do: rename best model
    if ('retrain_task' in yaml_config) and isinstance(yaml_config['retrain_task'], int) and yaml_config['retrain_task']>0:
        test_dat_type = 'fn_test'
        gen_inference_json(cfg, test_dat_type, best_model)    #gen val json
        model_metrics = evaluation_through_model_iter(score_threshold = best_conf_th, min_thresh = best_conf_th, test_dat=test_dat_type, model_list = best_model)   #eval val data
        
    return best_model_path
    
def aifs_performance_review(args):
    model_type='CenterNet2'
    #Val
    validate_new_models_on_val(model_type = model_type)
    val_best_model_iter_list = evaluate_new_models_on_val(fp_rate=args.fpr, return_type='best_models')
    validate_new_models_on_test_and_fn(model_type, val_best_model_iter_list)
    #Test and fn
    test_best_model_iter_list = evaluate_new_models_on_test_and_fn_by_fp_rate(val_best_model_iter_list, \
                                                                  fp_rate=args.fpr, return_type='best_models')
    
    new_eval_result_list = evaluate_new_models_on_test_and_fn_by_fp_rate(test_best_model_iter_list, \
                                                                  fp_rate=args.fpr, return_type='eval_result')
    
    
    best_model_iter = select_best_model_aifs(new_eval_result_list, args.top_k)
    
    best_eval_result = None
    for eval_result in new_eval_result_list:
        if eval_result['model_iter'] == best_model_iter:
            best_eval_result = eval_result
            break   
    
    #validate_old_model_on_test_and_fn(model_type = model_type)
    eval_result_list = evaluate_old_model_on_test_and_fn_by_fp_rate(fp_rate=args.fpr, return_type='eval_result')
    eval_result_list.append(best_eval_result)
    best_model_iter = select_best_model_aifs(eval_result_list, 1)
    if best_model_iter == best_eval_result['model_iter']:is_replaced = True
    else:is_replaced = False
    
    return best_model_iter, is_replaced

def main(args):
    if args.aifs:
        config_file = os.path.join(aoi_dir, 'config/config.yaml')
        print('aoi_dir: ', aoi_dir)
        sys.path.append(os.path.join(aoi_dir, "config"))
        from config import read_config_yaml, write_config_yaml
        config = read_config_yaml(config_file)
        model_file_dir = config['centernet2_model_output_dir']
        args.config_file = config['centernet2_yaml_file_path']
        
        pcb_data_dir = config['pcb_data_dir']
        train_json_file_path = os.path.join(pcb_data_dir, 'annotations/train.json')
        pcb_train_data_dir = os.path.join(pcb_data_dir, 'train')
        register_coco_instances("pcb_data_train", {}, train_json_file_path, pcb_train_data_dir)
    else:
        config_file = os.path.join(args.aoi_dir, 'config/config.yaml')
        from config import read_config_yaml, write_config_yaml
        config = read_config_yaml(config_file)
        model_file_dir = config['model_file_dir']
        args.config_file = config['yaml_file_path']
    
    cfg = setup(args, config)
    model = build_model(cfg)
    logger.info("Model:\n{}".format(model))
    
    if args.eval_only:
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        if cfg.TEST.AUG.ENABLED:
            logger.info("Running inference with test-time augmentation ...")
            model = GeneralizedRCNNWithTTA(cfg, model, batch_size=1)

        return do_test(cfg, model)

    distributed = comm.get_world_size() > 1
    if distributed:
        model = DistributedDataParallel(
            model, device_ids=[comm.get_local_rank()], broadcast_buffers=False,
            find_unused_parameters=True
        )
    
    do_train(cfg, model, resume=args.resume)
    if not args.aifs:
        best_model = performance_review(cfg, config, args, config['model_file_dir'])
        best_model_path = os.path.join(config['model_file_dir'], best_model)
        logger.info("Best_model_path:\n{}".format(best_model_path))
    else:
        #best_model = aifs_performance_review(cfg, config, args, config['centernet2_model_output_dir'])
        best_model_iter, Is_replaced = aifs_performance_review(args)     
    
        test_data_dir = config['test_data_dir']
        retrain_data_val_dir = config['retrain_data_val_dir']

        if Is_replaced: 
            centernet2_model_output_version = config['centernet2_model_output_version']

            config['centernet2_best_model_file_path'] = os.path.join(config['centernet2_model_output_dir'], "{}.pth".format(best_model_iter))
            config['centernet2_label_dir'] = [os.path.join(test_data_dir, '{}_inference_result'.format(centernet2_model_output_version), \
                                         best_model_iter, 'labels'),
                                         os.path.join(retrain_data_val_dir, '{}_inference_result'.format(centernet2_model_output_version), \
                                         best_model_iter, 'labels')]

        else:
            centernet2_old_model_file_path = config['centernet2_old_model_file_path']
            centernet2_old_model_file_id = os.path.splitext(os.path.basename(centernet2_old_model_file_path))[0]

            config['centernet2_best_model_file_path'] = centernet2_old_model_file_path
            config['centernet2_label_dir'] = [os.path.join(test_data_dir, 'CenterNet2_old_inference_result', centernet2_old_model_file_id, 'labels'),
                                             os.path.join(retrain_data_val_dir, 'CenterNet2_old_inference_result', centernet2_old_model_file_id, 'labels')]
        
        best_model_path = config['centernet2_best_model_file_path']
                                         
    logger.info("Best_model_path:\n{}".format(best_model_path))
    write_config_yaml(config_file, config)

if __name__ == "__main__":
    args = default_argument_parser()
    args.add_argument('--manual_device', default='')
    args.add_argument('--fpr',type=float,default=0.6,help='(default=%(default)d)')
    args.add_argument('--max_conf_th',type=float,default=0.3,help='(default=%(default)d)')
    args.add_argument('--min_conf_th',type=float,default=0.01,help='(default=%(default)d)')
    args.add_argument('--top_k',type=int,default=5,help='(default=%(default)d)')
    args.add_argument('--aoi_dir',default='/home/aoi/AOI_PCB_CenterNet2/ctr2_pcb/previous_stuff',type=str,help='(default=%(default)s)')
    args.add_argument('--config_file',default='',type=str,help='(default=%(default)s)')
    args.add_argument('--aifs', action='store_true', help='')
    args = args.parse_args()
    
    if args.manual_device != '':
        os.environ['CUDA_VISIBLE_DEVICES'] = args.manual_device
    args.dist_url = 'tcp://127.0.0.1:{}'.format(
        torch.randint(11111, 60000, (1,))[0].item())
    print("Command Line Args:", args)    
    
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )