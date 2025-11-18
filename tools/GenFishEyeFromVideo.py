import argparse
import os
import cv2
from model.utils.FastFisheyeGen import FastFisheyeProjector
from model.utils.equirect_utils import DEFAULT_JITTER_CONFIG, NO_JITTER_CONFIG

def main():
    arg = argparse.ArgumentParser(description="Generate fisheye images from equirectangular video")
    arg.add_argument("-i", "--input_dir", required=True, help="Input directory containing equirectangular video")