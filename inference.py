from feature_extraction.extract_main import generate
import argparse
import os
import time

import numpy as np
import torch

from libs import models
from libs.config import get_config
from libs.postprocess import PostProcessor
from debug_model import visualize_pred

def get_arguments() -> argparse.Namespace:
    """
    parse all the arguments from command line inteface
    return a list of parsed arguments
    """

    parser = argparse.ArgumentParser(
        description="save predictions from Action Segmentation Networks."
    )
    parser.add_argument(
        "--config",
        type=str,
        help="path to a config file about the experiment on action segmentation",
        default='./config/config.yaml'
    )
    parser.add_argument(
        "--cpu", action="store_true", help="Add --cpu option if you use cpu."
    )

    parser.add_argument('--videopath', type=str, default="test_inference")
    parser.add_argument('--outputpath', type=str, default="test_inference")
    parser.add_argument('--pretrainedpath', type=str, default="feature_extraction/pretrained/i3d_r50_kinetics.pth")
    parser.add_argument('--frequency', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--sample_mode', type=str, default="center_crop")
    args = parser.parse_args()
    
    return args

def inference_video(args, model, device, boundary_th, result_path, npy_file=None):
    postprocessor = PostProcessor("refinement_with_boundary", boundary_th)
    
    # feature = generate(args.videopath, str(args.outputpath), args.pretrainedpath, args.frequency, args.batch_size, args.sample_mode)[0]
    # feature = feature.reshape(-1, 2048).T
    feature = np.load(npy_file)
    feature = torch.from_numpy(feature).float()
    feature = feature[:, :: 2] # downsamp
    feature = feature.unsqueeze(0)
    mask = [[[1 for i in range(feature.shape[2])]]]
    mask = torch.tensor(mask, dtype=torch.bool)
    model.eval()

    with torch.inference_mode():

        feature = feature.to(device)
        output_cls, output_bound = model(feature)

        output_cls = output_cls.to("cpu").data.numpy()
        output_bound = output_bound.to("cpu").data.numpy()

        refined_pred = postprocessor(
            output_cls, boundaries=output_bound, masks=mask)
        pred = output_cls.argmax(axis=1)
        pred_path = result_path + npy_file[-12:-4] + '_refined_pred.npy'
        
        # np.save(result_path + npy_file[-12:-4] + '_pred.npy', pred[0])
        np.save(pred_path, refined_pred[0])

        return pred_path

def predict_file(npy_file, result_path):
    args = get_arguments()
    config = get_config(args.config)

    # result_path = './test_inference/'
    # video_path = 'test_inference/rgb-01-1.avi'
    # pred_path = result_path + '_refined_pred.npy'
    # gt_path = result_path + '_refined_pred.npy'

    # npy_file = 'test_inference/rgb-01-1.npy'
    # cpu or gpu
    if args.cpu:
        device = "cpu"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            torch.backends.cudnn.benchmark = True
    print(device)
    n_classes = 19
    
    model = models.ActionSegmentRefinementFramework(
        in_channel=config.in_channel,
        n_features=config.n_features,
        n_classes=n_classes,
        n_stages=config.n_stages,
        n_layers=config.n_layers,
        n_stages_asb=config.n_stages_asb,
        n_stages_brb=config.n_stages_brb,
    )
    
    model.to(device)
    state_dict_cls = torch.load(os.path.join(result_path, "model_70.prm"))
    model.load_state_dict(state_dict_cls)

    start_time = time.time()
    pred_path = inference_video(args, model, device, config.boundary_th, result_path, npy_file=npy_file)
    # visualize_pred(gt_path, pred_path, video_path)
    print("Done in {0}.".format(time.time() - start_time))

    return pred_path

if __name__ == '__main__':
    npy_file = 'samples/rgb-01-1.npy'
    result_path = 'samples'
    predict_file(npy_file, result_path)

