# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

import os
import argparse
from torch.autograd import Variable
import torch

from models.PFLD import PFLD
from models.PFLD_Ghost import PFLD_Ghost
from models.PFLD_Ghost_Slim import PFLD_Ghost_Slim

parser = argparse.ArgumentParser(description='pytorch2onnx')
parser.add_argument('--model_type', default='PFLD_Ghost_Slim', type=str)
parser.add_argument('--input_size', default=96, type=int)
parser.add_argument('--width_factor', default=0.25, type=float)
parser.add_argument('--landmark_number', default=98, type=int)
parser.add_argument('--model_path', default="./checkpoint/models/1/PFLD_Ghost_Slim_0.25_96/pfld_ghost_slim_best.pth")
parser.add_argument('--onnx_model', default="./pfld.onnx")
parser.add_argument('--onnx_model_sim', help='Output ONNX model', default="./pfld-sim.onnx")
args = parser.parse_args()

print("=====> load pytorch checkpoint...")
checkpoint = torch.load(args.model_path, map_location=torch.device('cpu'))
MODEL_DICT = {'PFLD': PFLD,
              'PFLD_Ghost': PFLD_Ghost,
              'PFLD_Ghost_Slim': PFLD_Ghost_Slim,
              }
MODEL_TYPE = args.model_type
WIDTH_FACTOR = args.width_factor
INPUT_SIZE = args.input_size
LANDMARK_NUMBER = args.landmark_number
model = MODEL_DICT[MODEL_TYPE](WIDTH_FACTOR, INPUT_SIZE, LANDMARK_NUMBER)
model.load_state_dict(checkpoint)

print("=====> convert pytorch model to onnx...")
dummy_input = Variable(torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE))
input_names = ["input"]
output_names = ["output"]
torch.onnx.export(model, dummy_input, "{}_{}_{}.onnx".format(MODEL_TYPE, INPUT_SIZE, WIDTH_FACTOR), verbose=False, input_names=input_names, output_names=output_names)
