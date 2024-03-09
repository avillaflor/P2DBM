import os
import sys

CARLA_PATH = os.environ.get("CARLA_9_10_PATH")
CARLA_PYTHONPATH = os.environ.get("CARLA_9_10_PYTHONPATH")
if CARLA_PATH == None:
    raise ValueError("Set $CARLA_9_10_PATH to directory that contains CarlaUE4.sh")
if CARLA_PYTHONPATH == None:
    raise ValueError("Set $CARLA_9_10_PYTHONPATH to directory that contains egg file")

try:
    sys.path.append(CARLA_PYTHONPATH)
    os.environ['CARLA_PATH'] = CARLA_PATH
except IndexError:
    print(".egg file not found! Kindly check for your Carla installation.")
    pass
