#!/usr/bin/env python3

# Copyright (c) 2017 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""Basic CARLA client example."""

from __future__ import print_function

import argparse
import logging
import random
import time

from carla.client import make_carla_client
from carla.sensor import Camera, Lidar
from carla.settings import CarlaSettings
from carla.tcp import TCPConnectionError
from carla.util import print_over_same_line
from carla.image_converter import *
from PIL import Image as PImage

import csv
import cv2
import h5py
from keras.models import load_model
import os

def run_carla_client(args):
    # Here we will run 3 episodes with 300 frames each.
    number_of_episodes = 3  #160
    frames_per_episode = 2000
    
    speed_model = load_model("./speed_model.h5")
    brake_model = load_model("./brake_model.h5")
    steer_model = load_model("./steer_model.h5")

    # We assume the CARLA server is already waiting for a client to connect at
    # host:port. To create a connection we can use the `make_carla_client`
    # context manager, it creates a CARLA client object and starts the
    # connection. It will throw an exception if something goes wrong. The
    # context manager makes sure the connection is always cleaned up on exit.
    with make_carla_client(args.host, args.port) as client:
        print('CarlaClient connected')

        for episode in range(0, number_of_episodes):
            # Start a new episode.

            if args.settings_filepath is None:

                # Create a CarlaSettings object. This object is a wrapper around
                # the CarlaSettings.ini file. Here we set the configuration we
                # want for the new episode.
                settings = CarlaSettings()
                settings.set(
                    SynchronousMode=True,
                    SendNonPlayerAgentsInfo=True,
                    NumberOfVehicles=40,
                    NumberOfPedestrians=40,
                    WeatherId=1,
                    QualityLevel=args.quality_level)
                settings.randomize_seeds()

                # Now we want to add a couple of cameras to the player vehicle.
                # We will collect the images produced by these cameras every
                # frame.

                # The default camera captures RGB images of the scene.
                camera0 = Camera('CameraRGB')
                # Set image resolution in pixels.
                camera0.set_image_size(800, 600)
                # Set its position relative to the car in meters.
                camera0.set_position(0.30, 0, 1.30)
                settings.add_sensor(camera0)

            else:

                # Alternatively, we can load these settings from a file.
                with open(args.settings_filepath, 'r') as fp:
                    settings = fp.read()

            # Now we load these settings into the server. The server replies
            # with a scene description containing the available start spots for
            # the player. Here we can provide a CarlaSettings object or a
            # CarlaSettings.ini file as string.
            scene = client.load_settings(settings)
            #world = client.load_world('Town02')
            #client.reload_world()

            # Choose one player start at random.
            number_of_player_starts = len(scene.player_start_spots)
            #player_start = random.randint(0, max(0, number_of_player_starts - 1))
            player_start = 36

            #def _poses_straight():
            #    return [[36, 40], [39, 35], [110, 114], [7, 3], [0, 4],
            #            [68, 50], [61, 59], [47, 64], [147, 90], [33, 87],
            #            [26, 19], [80, 76], [45, 49], [55, 44], [29, 107],
            #            [95, 104], [84, 34], [53, 67], [22, 17], [91, 148],
            #            [20, 107], [78, 70], [95, 102], [68, 44], [45, 69]]
#
#            #def _poses_one_curve():
#            #    return [[138, 17], [47, 16], [26, 9], [42, 49], [140, 124],
#            #            [85, 98], [65, 133], [137, 51], [76, 66], [46, 39],
#            #            [40, 60], [0, 29], [4, 129], [121, 140], [2, 129],
#            #            [78, 44], [68, 85], [41, 102], [95, 70], [68, 129],
            #            [84, 69], [47, 79], [110, 15], [130, 17], [0, 17]]

            #player_start = random.choice([109, 87, 100, 24, 73])
            print(player_start)
            print('------')
            # Notify the server that we want to start the episode at the
            # player_start index. This function blocks until the server is ready
            # to start the episode.
            print('Starting new episode at %r...' % scene.map_name)
            client.start_episode(player_start)

            # Iterate every frame in the episode.
            for frame in range(0, frames_per_episode):

                # Read the data produced by the server this frame.
                measurements, sensor_data = client.read_data()

                imageCV = to_rgb_array(sensor_data['CameraRGB'])
                imageCV = imageCV[200:400, 200:600]
                cv2.imshow("FC image", imageCV)
                cv2.waitKey(1)
                
                
                control = measurements.player_measurements.autopilot_control
                # Get speed
                speed = float(speed_model.predict(imageCV[None,:,:,:], batch_size=1))
                speed_network = speed
                if speed > 0.75:
                    speed = 0.9
                elif speed < 0.3:
                    speed = 0.0
                else:
                    speed = 0.5

                #brake = 0
                brake = float(brake_model.predict(imageCV[None,:,:,:], batch_size=1))
                brake_network = brake
                if brake > 0.4:
                    brake = 1.0
                else:
                    brake = 0.0

                #steer = 0
                steer = float(steer_model.predict(imageCV[None,:,:,:], batch_size=1))

                print(control.throttle, control.steer, control.brake)
                print(speed_network, steer, brake_network)
                #print(player_start)
                print('------')
                control.brake = brake
                control.throttle = speed
                control.steer = steer
                client.send_control(control)


def main():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='localhost',
        help='IP of the host server (default: localhost)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-a', '--autopilot',
        action='store_true',
        help='enable autopilot')
    argparser.add_argument(
        '-l', '--lidar',
        action='store_true',
        help='enable Lidar')
    argparser.add_argument(
        '-q', '--quality-level',
        choices=['Low', 'Epic'],
        type=lambda s: s.title(),
        default='Low',
        help='graphics quality level, a lower level makes the simulation run considerably faster.')
    argparser.add_argument(
        '-i', '--images-to-disk',
        action='store_true',
        dest='save_images_to_disk',
        help='save images (and Lidar data if active) to disk')
    argparser.add_argument(
        '-c', '--carla-settings',
        metavar='PATH',
        dest='settings_filepath',
        default=None,
        help='Path to a "CarlaSettings.ini" file')

    args = argparser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    args.out_filename_format = '_out/episode_{:0>4d}/{:s}/{:0>6d}'

    while True:
        try:

            run_carla_client(args)

            print('Done.')
            return

        except TCPConnectionError as error:
            logging.error(error)
            time.sleep(1)


if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
