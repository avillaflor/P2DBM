import os
import argparse


from src.envs.carla.features.carla_map_features import CarlaMapFeatures


TOWNS = ['Town01', 'Town02', 'Town03', 'Town04', 'Town05', 'Town06', 'Town07', 'Town10HD']


def main(args):
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    if args.xodr_path != '':
        town = os.path.basename(args.xodr_path)[:-5]
        map_xodr_path = os.path.dirname(args.xodr_path)
        print('Generating features for {0}'.format(town))
        map_features = CarlaMapFeatures(town, map_xodr_path=map_xodr_path)
        map_features.save_data(args.save_dir)
        print('Features saved for {0}'.format(town))
    else:
        if args.carla_path == '':
            carla_path = os.environ['CARLA_PATH']
        else:
            carla_path = args.carla_path

        if not carla_path:
            print('Invalid carla path')

        for town in TOWNS:
            xodr_path = os.path.join(carla_path, 'CarlaUE4/Content/Carla/Maps/OpenDrive/')
            if os.path.exists(xodr_path):
                print('Generating features for {0}'.format(town))
                map_features = CarlaMapFeatures(town, map_xodr_path=xodr_path, precision=args.precision)
                map_features.save_data(args.save_dir)
                print('Features saved for {0}'.format(town))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--carla_path', type=str, default='')
    parser.add_argument('--xodr_path', type=str, default='')
    parser.add_argument('--save_dir', type=str, default='maps/')
    parser.add_argument('--precision', type=float, default=5.0)
    args = parser.parse_args()
    main(args)
