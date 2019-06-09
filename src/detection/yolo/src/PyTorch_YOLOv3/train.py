from __future__ import division

from models import *
from utils.logger import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *
from test import evaluate

from terminaltables import AsciiTable

import os
import sys
import time
import datetime
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim

if __name__ == "__main__":
    
    # Get default parser
    parser = argparse.ArgumentParser()
    if 0:
        learning_rate = 1e-3
        parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
        parser.add_argument("--data_config", type=str, default="config/coco.data", help="path to data config file")
        parser.add_argument("--pretrained_weights", type=str, help="if specified starts from checkpoint model")
        parser.add_argument("--batch_size", type=int, default=4, help="size of each image batch")

        parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
        parser.add_argument("--gradient_accumulations", type=int, default=2, help="number of gradient accums before step")
        parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
        parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
        parser.add_argument("--checkpoint_interval", type=int, default=20, help="interval between saving model weights")
        parser.add_argument("--evaluation_interval", type=int, default=20, help="interval evaluations on validation set")
        parser.add_argument("--compute_map", default=False, help="if True computes mAP every tenth batch")
        parser.add_argument("--multiscale_training", default=True, help="allow for multi-scale training")
   
    else: # My: train on synthesized
        IF_PRINT_LOG_STR = True
        
        parser.add_argument("--model_def", type=str, 
            default="data/digits_generated/yolo.cfg", help="path to model definition file")
        
        if 0: # Train from imagenet
            learning_rate = 1e-4
            
            parser.add_argument("--pretrained_weights", type=str, help="if specified starts from checkpoint model",
                default='weights/darknet53.conv.74')
                
        else: # Train from my trained model
            learning_rate = 5e-6
            parser.add_argument("--pretrained_weights", type=str, help="if specified starts from checkpoint model",
                default='checkpoints/yolov3_ckpt_300.pth')
            
        parser.add_argument("--data_config", type=str, 
            default="data/digits_generated/yolo.data", help="path to data config file")
            
        parser.add_argument("--batch_size", type=int, help="size of each image batch",
            default=4)
            
        parser.add_argument("--epochs", type=int, default=21, help="number of epochs")
        parser.add_argument("--gradient_accumulations", type=int, default=2, help="number of gradient accums before step")
        parser.add_argument("--n_cpu", type=int, default=4, help="number of cpu threads to use during batch generation")
        parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
        parser.add_argument("--checkpoint_interval", type=int, default=5, help="interval between saving model weights")
        parser.add_argument("--evaluation_interval", type=int, default=5, help="interval evaluations on validation set")
        parser.add_argument("--compute_map", default=False, help="if True computes mAP every tenth batch")
        parser.add_argument("--multiscale_training", default=True, help="allow for multi-scale training")
    
    args = parser.parse_args()
    print("\nArgs:\n", args)

    logger = Logger("logs")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("output", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # Get data configuration
    data_config = parse_data_config(args.data_config)
    train_path = data_config["train"]
    valid_path = data_config["valid"]
    class_names = load_classes(data_config["names"])

    # Initiate model
    model = Darknet(args.model_def).to(device)
    model.apply(weights_init_normal)

    # If specified we start from checkpoint
    if args.pretrained_weights:
        if args.pretrained_weights.endswith(".pth"):
            model.load_state_dict(torch.load(args.pretrained_weights))
        else:
            model.load_darknet_weights(args.pretrained_weights)

    # Get dataloader
    dataset = ListDataset(train_path, augment=True, multiscale=args.multiscale_training)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.n_cpu,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    metrics = [
        "grid_size",
        "loss",
        "x",
        "y",
        "w",
        "h",
        "conf",
        "cls",
        "cls_acc",
        "recall50",
        "recall75",
        "precision",
        "conf_obj",
        "conf_noobj",
    ]

    for epoch in range(1, 1+args.epochs):
        model.train()
        start_time = time.time()
        for batch_i, (_, imgs, targets) in enumerate(dataloader):
            batch_i = batch_i + 1 # 1-indexed. (this won't affect the next loop)
            
            if batch_i % 50 == 0 or batch_i == len(dataloader):
                print(f"epoch = {epoch}/{args.epochs}, batch = {batch_i}/{len(dataloader)}")
            
            batches_done = len(dataloader) * (epoch - 1) + (batch_i - 1)

            imgs = Variable(imgs.to(device))
            targets = Variable(targets.to(device), requires_grad=False)

            loss, outputs = model(imgs, targets)
            loss.backward()

            if batches_done % args.gradient_accumulations == 0:
                # Accumulates gradient before each step
                optimizer.step()
                optimizer.zero_grad()

            # ----------------
            #   Log progress
            # ----------------

            log_str = "\n---- [Epoch %d/%d, Batch %d/%d] ----\n" % (epoch, args.epochs, batch_i, len(dataloader))

            metric_table = [["Metrics", *[f"YOLO Layer {i}" for i in range(len(model.yolo_layers))]]]

            # Log metrics at each YOLO layer
            for i, metric in enumerate(metrics):
                formats = {m: "%.6f" for m in metrics}
                formats["grid_size"] = "%2d"
                formats["cls_acc"] = "%.2f%%"
                row_metrics = [formats[metric] % yolo.metrics.get(metric, 0) for yolo in model.yolo_layers]
                metric_table += [[metric, *row_metrics]]

                # Tensorboard logging
                tensorboard_log = []
                for j, yolo in enumerate(model.yolo_layers):
                    for name, metric in yolo.metrics.items():
                        if name != "grid_size":
                            tensorboard_log += [(f"{name}_{j+1}", metric)]
                tensorboard_log += [("loss", loss.item())]
                logger.list_of_scalars_summary(tensorboard_log, batches_done)

            # Print log str
            if batch_i % 10 == 0 or batch_i == len(dataloader):

                log_str += AsciiTable(metric_table).table
                log_str += f"\nTotal loss {loss.item()}"

                # Determine approximate time left for epoch
                epoch_batches_left = len(dataloader) - (batch_i + 1)
                time_left = datetime.timedelta(seconds=epoch_batches_left * (time.time() - start_time) / (batch_i + 1))
                log_str += f"\n---- ETA {time_left}"
                
                if IF_PRINT_LOG_STR:
                    print(log_str)

            model.seen += imgs.size(0)

        if epoch % args.evaluation_interval == 0:
            print("\n---- Evaluating Model ----")
            # Evaluate the model on the validation set
            precision, recall, AP, f1, ap_class = evaluate(
                model,
                path=valid_path,
                iou_thres=0.5,
                conf_thres=0.5,
                nms_thres=0.5,
                img_size=args.img_size,
                batch_size=8,
            )
            evaluation_metrics = [
                ("val_precision", precision.mean()),
                ("val_recall", recall.mean()),
                ("val_mAP", AP.mean()),
                ("val_f1", f1.mean()),
            ]
            logger.list_of_scalars_summary(evaluation_metrics, epoch)

            # Print class APs and mAP
            ap_table = [["Index", "Class name", "AP"]]
            for i, c in enumerate(ap_class):
                ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
            print(AsciiTable(ap_table).table)
            print(f"---- mAP {AP.mean()}")

        if epoch % args.checkpoint_interval == 0:
            torch.save(model.state_dict(), f"checkpoints/yolov3_ckpt_%d.pth" % epoch)
 
 
 