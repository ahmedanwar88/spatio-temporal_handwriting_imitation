#!/usr/bin/env python3

import time
from PIL import Image

from pipeline.skeletonization import Skeletonizer
from pipeline.sampling import sample_to_penpositions
from pipeline.graves import GravesWriter
from pipeline.align import align

class Mimicking():
    def __init__(self, inputImg, inputTxt):
        self.inputTxt = inputTxt
        with Skeletonizer() as skeletonizer:
            skeletonBlurImg = skeletonizer.skeletonize_blurred(inputImg)
            skeletonImg = skeletonizer.skeletonize_sharp(skeletonBlurImg)
        self.penPositions = sample_to_penpositions(skeletonImg)

    def filter_strokes(self, newPenPositions):
        strokes = []
        strokes_dict = {'x': [], 'y': []}
        for i, penPosition in enumerate(newPenPositions):
            strokes_dict['x'].append(penPosition.pos[0])
            strokes_dict['y'].append(penPosition.pos[1])
            if penPosition.penUp:
                strokes.append(strokes_dict)
                strokes_dict = {'x': [], 'y': []}
                continue
            # last element
            if i == len(newPenPositions) - 1:
                strokes.append(strokes_dict)
        return strokes

    def mimick(self, writer, outputTxt):
        newPenPositions = writer.write(outputTxt, self.inputTxt, self.penPositions)
        newPenPositions = align(newPenPositions, self.penPositions)
        strokes = self.filter_strokes(newPenPositions)
        return strokes
   
def main():
    inputImg = Image.open("input.png")
    inputTxt = "above or sinking below"
    outputTxt = "H"

    start_time = time.time()
    mimicking = Mimicking(inputImg, inputTxt)
    end_time = time.time()
    print(f"Model loading takes {end_time - start_time} s")
    
    with GravesWriter() as writer:
        start_time = time.time()
        strokes = mimicking.mimick(writer, outputTxt)
        end_time = time.time()
        print(f"Model first inference takes {end_time - start_time} s")

        start_time = time.time()
        strokes = mimicking.mimick(writer, outputTxt)
        end_time = time.time()
        print(f"Model second inference takes {end_time - start_time} s")

        start_time = time.time()
        strokes = mimicking.mimick(writer, outputTxt)
        end_time = time.time()
        print(f"Model third inference takes {end_time - start_time} s")
        
        times = []
        for _ in range(5):
            start_time = time.time()
            strokes = mimicking.mimick(writer, outputTxt)
            end_time = time.time()
            times.append(end_time - start_time)
        print(times)
        print(f"\nAverage inference time = {round(sum(times) / len(times) * 1000, 2)} ms")
    
    
if __name__ == "__main__":
    main()
