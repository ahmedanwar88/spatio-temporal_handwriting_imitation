#!/usr/bin/env python3

import argparse
import os

from PIL import Image

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from pipeline.skeletonization import Skeletonizer
from pipeline.sampling import sample_to_penpositions
from pipeline.graves import GravesWriter
from pipeline.align import align
from pipeline.render_skeleton import render_skeleton
from pipeline.pen_style_transfer import PenStyleTransfer

from datastructures.PenPosition import plotPenPositions




def main():

    parser = argparse.ArgumentParser(description='The first working version (without pen style transfer). Modifies the content of an image.')
    parser.add_argument('--text-in', help='The input text', required=True)
    parser.add_argument('--text-out', help='The output text', required=True)
    parser.add_argument('input', help='The input file')
    args = parser.parse_args()
    print(args)

    inputImg = Image.open(args.input)

    with Skeletonizer() as skeletonizer:
        skeletonBlurImg = skeletonizer.skeletonize_blurred(inputImg)
        skeletonImg = skeletonizer.skeletonize_sharp(skeletonBlurImg)

    penPositions = sample_to_penpositions(skeletonImg)
    
    orig_pos_list = []

    print("Original pen positions:")
    for penPosition in penPositions:
      print(penPosition.__str__())
      orig_pos_list.append(penPosition.__str__())

    orig_df = pd.DataFrame(orig_pos_list)
    orig_df.to_csv('orig_points.csv')

    with GravesWriter() as writer:
        newPenPositions = writer.write(args.text_out, args.text_in, penPositions)

    

    newPenPositions = align(newPenPositions, penPositions)


    print("New pen positions:")
    print(newPenPositions)
    out_pos_list = []
    for penPosition in newPenPositions:
      print(penPosition.__str__())
      out_pos_list.append(penPosition.__str__())

    out_df = pd.DataFrame(out_pos_list)
    out_df.to_csv('out_points.csv')

    newSkeletonBlurImg, newSkeletonImg = render_skeleton(newPenPositions, inputImg.size)

    with PenStyleTransfer() as penStyleTransfer:
        outputImg = penStyleTransfer.transferStyle(newSkeletonBlurImg, inputImg)

    print("Done. Displaying results ...")

    plt.figure('Full Pipeline', figsize=(16, 9))
    plt.subplot(3, 2, 1)
    plt.imshow(inputImg)
    plt.subplot(3, 2, 3)
    plt.imshow(skeletonBlurImg)
    plt.subplot(3, 2, 5)
    plt.imshow(skeletonImg, cmap='binary', vmax=10)
    plotPenPositions(penPositions)
    plt.subplot(3, 2, 6)
    plt.imshow(newSkeletonImg, cmap='binary', vmax=256*10)
    plotPenPositions(newPenPositions)
    plt.subplot(3, 2, 4)
    plt.imshow(newSkeletonBlurImg)
    plt.subplot(3, 2, 2)
    plt.imshow(outputImg)
    #plt.show()
    plt.savefig('output.png')

if __name__ == "__main__":
    main()
