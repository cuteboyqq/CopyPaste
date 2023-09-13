import os
import shutil
import glob
import numpy as np
from task.stopsign import StopSignDataset
from task.lanemarking import LaneMarkingDataset
from task.pedestrain import PedestrainDataset
from utils.config import get_args_StopSign,get_args_LaneMarking,get_args_Pedestrain


if __name__=="__main__":

    STOPSIGN=False
    LANEMARKING=True
    SEQUENTIAL_COPYPASTE=True
    PEDESTRIAN=False

    if SEQUENTIAL_COPYPASTE:
        if STOPSIGN:
            args_stopsign = get_args_StopSign()
            stopsign = StopSignDataset(args_stopsign)
            stopsign.CopyPaste()
        if LANEMARKING:
            args_lanemarking = get_args_LaneMarking()
            lanemarking = LaneMarkingDataset(args_lanemarking)
            lanemarking.CopyPaste()
        if PEDESTRIAN:
            args_pedestrian = get_args_Pedestrain()
            pedestrain = PedestrainDataset(args_pedestrian)
            pedestrain.CopyPasteSimple()
    # if LANEMARKING:
    #         args_lanemarking = get_args_LaneMarking()
    #         lanemarking = LaneMarkingDataset(args_lanemarking)
    #         lanemarking.CopyPaste()
    
    # if PEDESTRIAN:
    #     args_pedestrian = get_args_Pedestrain()
    #     pedestrain = PedestrainDataset(args_pedestrian)
    #     pedestrain.CopyPasteSimple()
