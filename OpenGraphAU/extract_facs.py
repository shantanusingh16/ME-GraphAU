import os
import numpy as np
import torch
from model.ANFL import MEFARG
from utils import *
from conf import get_config,set_logger,set_outdir,set_env
import h5py
import cv2
from multiprocessing import Pool
import glob
import argparse
import traceback


def main(conf, vid_dir, keypts_dir, output_dir):
    dataset_info = hybrid_prediction_infolist

    # model
    net = MEFARG(num_main_classes=conf.num_main_classes, num_sub_classes=conf.num_sub_classes, backbone=conf.arc, neighbor_num=conf.neighbor_num, metric=conf.metric)

    weights = torch.load('checkpoints/OpenGprahAU-SwinB_first_stage.pth')['state_dict']
    weights = dict([(k.replace("module.", ""), v) for k, v in weights.items()])
    net.load_state_dict(weights)

    net.eval()
    
    if torch.cuda.is_available():
        net = net.cuda()

    img_transform = image_eval()

    # Load data
    filenames = os.listdir(vid_dir)
    os.makedirs(output_dir, exist_ok=True)

    for fn in filenames:
        try:
            basename = os.path.splitext(os.path.basename(fn))[0]
            vid_path = f'{vid_dir}/{basename}.mp4'
            keypts_path = f'{keypts_dir}/{basename}.h5'
            output_path = f'{output_dir}/{basename}.npy'

            # # For eval only
            # out_frames = f'{output_dir}/frames/{basename}'
            # os.makedirs(out_frames, exist_ok=True)

            if os.path.exists(output_path):
                continue

            keypts_path = keypts_path.encode('unicode_escape').decode().replace('\\u','#U')
            # keypts_path = "".join(["#U{:04x}".format(ord(c)) if c in "'\"" or c == "\\" or c == "/" else c for c in keypts_path])


            skels = np.array(h5py.File(keypts_path)[basename])
            vid = cv2.VideoCapture(vid_path)

            # Data pre-processing
            idx = 0
            final_scores = []
            while True:
                ret, frame = vid.read()
                if not ret or idx >= skels.shape[0]:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h,w,c = frame.shape

                #TODO : Fix to one format in h5!!!! Either (X,Y,Z,Confidence) or (X,Y,Confidence)
                if skels.shape[-1] == 252:
                    frame_keypts = skels[idx].reshape((-1, 4))
                elif skels.shape[-1] == 189:
                    frame_keypts = skels[idx].reshape((-1, 3))

                #left-shoulder idx 13, right_shoulder idx 14
                x_left, x_right = int(frame_keypts[13, 0] * w), int(frame_keypts[14, 0] * w)
                pad = min(int(((x_left - x_right) * 0.05)), 10)

                w1 = max(0, x_right-pad)
                w2 = min(w, x_left+pad)

                if (w1 >= w2) or (w2-w1 < w*0.1):
                    final_scores.append(np.zeros((41,), dtype=np.float32))
                    idx += 1
                    continue

                cropped_frame = frame[:, w1:w2, :]
                img = Image.fromarray(cropped_frame)
                img_ = img_transform(img).unsqueeze(0)

                if torch.cuda.is_available():
                    img_ = img_.cuda()

                with torch.no_grad():
                    pred = net(img_)
                    pred = pred.squeeze().cpu().numpy()

                score = pred * (pred > 0.5)
                final_scores.append(score)
                
                # infostr_probs, infostr_aus = dataset_info(pred, 0.5)
                # img = draw_text(conf.input, list(infostr_aus), pred, cropped_frame)
                # path = os.path.join(out_frames, '{:04}.png'.format(idx))
                # cv2.imwrite(path, img)

                idx += 1
            
            final_scores = np.array(final_scores)
            np.save(output_path, final_scores)
        except Exception as e:
            print(f'Error parsing file: {fn}')
            print(e)
            traceback.print_exc()


# ---------------------------------------------------------------------------------

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-j', '--jobs', help='Number of jobs to run in parallel', default=4, type=int)
    parser.add_argument('-vid', '--vid_dir', help='Source directory containing video files', required=True)
    parser.add_argument('-kpts', '--keypts_dir', help="Source directory containing keypoint files (h5)", required=True)
    parser.add_argument('-out', '--output_dir', help="Destination directory where AUs will be written (npy)", required=True)

    args = parser.parse_args()

    conf = get_config()
    conf.evaluate = True
    set_env(conf)
    # generate outdir name
    set_outdir(conf)
    # Set the logger
    set_logger(conf)

    vid_names = os.listdir(args.vid_dir)

    pool_args = [(conf,
                  os.path.join(args.vid_dir, name), 
                  os.path.join(args.keypts_dir, name), 
                  os.path.join(args.output_dir, name)) for name in vid_names]
    
    with Pool(processes=args.jobs) as pool:
        pool.starmap(main, pool_args)

    #for arg in pool_args:
    #    main(*arg)

