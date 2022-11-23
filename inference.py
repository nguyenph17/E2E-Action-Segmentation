from feature_extraction.extract_main import generate
import argparse
import os
import time

import numpy as np
import torch

from libs import models
from libs.config import get_config
from libs.postprocess import PostProcessor


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
        default='./result/50salads/dataset-50salads_split-1/config.yaml'
    )
    parser.add_argument(
        "--cpu", action="store_true", help="Add --cpu option if you use cpu."
    )

    parser.add_argument('--videopath', type=str, default="/mnt/f/ActionSegment/e2e_action_segmentation/test_result_inference/")
    parser.add_argument('--outputpath', type=str, default="/mnt/f/ActionSegment/e2e_action_segmentation/test_result_inference/")
    parser.add_argument('--pretrainedpath', type=str, default="pretrained/i3d_r50_kinetics.pth")
    parser.add_argument('--frequency', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--sample_mode', type=str, default="center_crop")
    args = parser.parse_args()
    
    return args

def inference_video(args, model, device, boundary_th, result_path):
    postprocessor = PostProcessor("refinement_with_boundary", boundary_th)
    
    feature = generate(args.videopath, str(args.outputpath), args.pretrainedpath, args.frequency, args.batch_size, args.sample_mode)[0]
    feature = feature.reshape(-1, 2048).T
    # feature = np.load('dataset/50salads/features/rgb-01-1.npy')
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

        np.save(result_path + 'pred.npy', pred[0])
        np.save(result_path + '_refined_pred.npy', refined_pred[0])

def main():
    args = get_arguments()
    config = get_config(args.config)

    result_path = './test_result_inference/'

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
    inference_video(args, model, device, config.boundary_th, result_path)
    print("Done in {0}.".format(time.time() - start_time))

if __name__ == '__main__':
    main()

