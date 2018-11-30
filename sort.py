"""
    SORT: A Simple, Online and Realtime Tracker
    Copyright (C) 2016 Alex Bewley alex@dynamicdetection.com

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
from __future__ import print_function

from numba import jit
import os.path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import io
from sklearn.utils.linear_assignment_ import linear_assignment
import glob
import time
import argparse
from filterpy.kalman import KalmanFilter

from sort_util import parse_args
from sort_util import Sort

  # all train
sequences = ['PETS09-S2L1','TUD-Campus','TUD-Stadtmitte','ETH-Bahnhof','ETH-Sunnyday','ETH-Pedcross2','KITTI-13','KITTI-17','ADL-Rundle-6','ADL-Rundle-8','Venice-2']
args = parse_args()
display = args.display
phase = 'train'
total_time = 0.0
total_frames = 0
colours = np.random.rand(32,3) #used only for display
if(display):
    if not os.path.exists('mot_benchmark'):
        print('\n\tERROR: mot_benchmark link not found!\n\n    Create a symbolic link to the MOT benchmark\n    (https://motchallenge.net/data/2D_MOT_2015/#download). E.g.:\n\n    $ ln -s /path/to/MOT2015_challenge/2DMOT2015 mot_benchmark\n\n')
        exit()
    plt.ion()
    fig = plt.figure()

if not os.path.exists('output'):
    os.makedirs('output')

for seq in sequences:
    mot_tracker = Sort() #create instance of the SORT tracker
    seq_dets = np.loadtxt('data/%s/det.txt'%(seq),delimiter=',') #load detections
    # print(seq_dets)
    with open('output/%s.txt'%(seq),'w') as out_file:
        print("Processing %s."%(seq))
        for frame in range(int(seq_dets[:,0].max())):
        # print(frame)
            frame += 1 #detection and frame numbers begin at 1
            dets = seq_dets[seq_dets[:,0]==frame,2:7]

            # print(dets)
            # time.sleep(3)
            dets[:,2:4] += dets[:,0:2] #convert to [x1,y1,w,h] to [x1,y1,x2,y2]
            # print(dets)
            # time.sleep(3)

            total_frames += 1

        if(display):
            ax1 = fig.add_subplot(111, aspect='equal')
            fn = 'mot_benchmark/%s/%s/img1/%06d.jpg'%(phase,seq,frame)
            im =io.imread(fn)
            ax1.imshow(im)
            plt.title(seq+' Tracked Targets')

        start_time = time.time()
        trackers = mot_tracker.update(dets)
        print(trackers)
        time.sleep(1)
        cycle_time = time.time() - start_time
        total_time += cycle_time

        for d in trackers:
            print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1'%(frame,d[4],d[0],d[1],d[2]-d[0],d[3]-d[1]),file=out_file)
            # time.sleep(1)
            if(display):
                d = d.astype(np.int32)
                ax1.add_patch(patches.Rectangle((d[0],d[1]),d[2]-d[0],d[3]-d[1],fill=False,lw=3,ec=colours[d[4]%32,:]))
                ax1.set_adjustable('box-forced')
        print(trackers)
        time.sleep(1)
        if(display):
            fig.canvas.flush_events()
            plt.draw()
            ax1.cla()

print("Total Tracking took: %.3f for %d frames or %.1f FPS"%(total_time,total_frames,total_frames/total_time))
if(display):
    print("Note: to get real runtime results run without the option: --display")
