import math
from logbook import Logger, StreamHandler
import sys
import torch
import cv2
import h5py
import numpy as np
import scipy.misc
from collections import deque
from utils.config import generate_configs, save_config, resume
from utils import utils
from agents import get_agent

# TODO must modify other stuff just to report evaluation

NAMESPACE = 'test_network_activations'
log = Logger(NAMESPACE)

tag_names = [
    'Steer', 'Gas', 'Brake', 'Hand Brake', 'Reverse Gear', 'Steer Noise',
    'Gas Noise', 'Brake Noise', 'Position X', 'Position Y', 'Speed',
    'Collision Other', 'Collision Pedestrian', 'Collision Car',
    'Opposite Lane Inter', 'Sidewalk Intersect', 'Acceleration X',
    'Acceleration Y', 'Acceleration Z', 'Platform time', 'Game Time',
    'Orientation X', 'Orientation Y', 'Orientation Z', 'Control signal',
    'Noise', 'Camera', 'Angle'
]

X_dim = [-1, -1, 0, 1, 1, 1, 0, -1]
Y_dim = [0, 1, 1, 1, 0, -1, -1, -1]


def get_cluster(start_pos, activation_map, pixel_value=None):

    cluster_map = np.zeros_like(activation_map)
    cluster_map[start_pos[0], start_pos[1]] = activation_map[start_pos[0],
                                                             start_pos[1]]

    if pixel_value is None:
        activation_map[start_pos[0], start_pos[1]] = 0
    else:
        activation_map[start_pos[0], start_pos[1]] = pixel_value
    nr_pixels = 1
    Q = deque()
    Q.appendleft(start_pos)
    while len(Q) != 0:
        pos_x, pos_y = Q.pop()
        nr_pixels += 1

        for x, y in zip(X_dim, Y_dim):
            if pos_x + x < 0:
                continue
            elif pos_y + y < 0:
                continue
            elif pos_x + x >= cluster_map.shape[0]:
                continue
            elif pos_y + y >= cluster_map.shape[1]:
                continue
            else:
                if (pixel_value is
                        None) and activation_map[pos_x + x, pos_y + y] != 0:
                    cluster_map[pos_x + x, pos_y + y] = activation_map[
                        pos_x + x, pos_y + y]
                    activation_map[pos_x + x, pos_y + y] = 0
                    Q.appendleft((pos_x + x, pos_y + y))
                elif (pixel_value is
                      not None) and activation_map[pos_x + x, pos_y + y] == 0:
                    cluster_map[pos_x + x, pos_y + y] = pixel_value
                    activation_map[pos_x + x, pos_y + y] = pixel_value
                    Q.appendleft((pos_x + x, pos_y + y))
                    if nr_pixels > 1500:
                        return cluster_map, nr_pixels

    return cluster_map.astype(np.uint8), nr_pixels


def get_cluster_no_activation(activation_map, value):

    for i in range(40, activation_map.shape[0]):
        for j in range(90, activation_map.shape[1]):
            if activation_map[i, j] == 0:
                cluster_map, nr_pixels = get_cluster((i, j), activation_map,
                                                     value)
                if nr_pixels > 1500:
                    return cluster_map


def cluster_activation(activation_map):
    nr_clusters = 0
    clusters = []

    activation_map[activation_map < 40] = 0
    for i in range(activation_map.shape[0]):
        for j in range(activation_map.shape[1]):
            if activation_map[i, j] != 0:
                nr_clusters += 1
                cluster_map, nr_pixels = get_cluster((i, j), activation_map)
                if nr_pixels > 100:
                    clusters.append(cluster_map)

    return clusters


def get_activation_image(raw_activation, image_activation):

    activation_map = raw_activation.data[0, 0].cpu().numpy()
    activation_map = (activation_map - np.min(activation_map)
                      ) / np.max(activation_map) - np.min(activation_map)

    activation_map = (activation_map * 255.0)

    if image_activation.shape[0] != activation_map.shape[0]:
        activation_map = scipy.misc.imresize(
            activation_map,
            [image_activation.shape[0], image_activation.shape[1]])

    kernel = np.ones((3, 3), np.uint8)
    activation_map = cv2.dilate(activation_map, kernel, iterations=1)

    clusters = cluster_activation(np.copy(activation_map).astype(np.uint8))
    no_activation_cluster = get_cluster_no_activation(
        np.copy(activation_map).astype(np.uint8), 111)

    image_activation[:, :, 1] += activation_map.astype(np.uint8)
    heat_map = cv2.applyColorMap(
        activation_map.astype(np.uint8), cv2.COLORMAP_JET)

    #image_activation = cv2.resize(image_activation, (740, 460), cv2.INTER_AREA)
    image_activation = cv2.resize(image_activation, (260, 180), cv2.INTER_AREA)
    image_activation = cv2.cvtColor(image_activation, cv2.COLOR_RGB2BGR)

    #heat_map = cv2.resize(heat_map, (740, 460), cv2.INTER_AREA)
    heat_map = cv2.resize(heat_map, (260, 180), cv2.INTER_AREA)


    return image_activation, activation_map, heat_map, clusters, no_activation_cluster


def obstruct_image(image, cluster, image_mean):
    image[cluster > 0] = image_mean.astype(np.uint8)
    return image


def eval_image(agent, image, speed, control_cmd, gt_steer, gt_gas, gt_brake,
               gt_speed):

    pred_steer, pred_gas, pred_brake, pred_speed, activ_map = agent.run_image(
        image, speed, control_cmd)

    log.info("Printing network predictions .....\n\n")
    log.info("Steer: predicted {},  ground_truth {}".format(
        pred_steer, gt_steer))
    log.info("Break: predicted {},  ground_truth {}".format(
        pred_brake, gt_brake))
    log.info("Gas: predicted {}, ground_truth {}".format(pred_gas, gt_gas))

    log.info("Speed: predicted {}, ground_truth {}\n\n".format(
        pred_speed, gt_speed))

    return pred_steer, pred_gas, pred_brake, pred_speed, activ_map


def eval_network(agent, cfg):

    openedFile = h5py.File(cfg.path_to_image, 'r')

    image = openedFile["rgb"][cfg.image_number]
    ground_truth = openedFile["targets"][cfg.image_number]
    image_mean = np.mean(np.mean(image, axis=0), axis=0)

    if cfg.image_number > 0:
        previous_speed = openedFile["targets"][cfg.image_number
                                               - 1][tag_names.index("Speed")]
    else:
        previous_speed = ground_truth[tag_names.index("Speed")]

    gt_steer = ground_truth[tag_names.index('Steer')]
    gt_brake = ground_truth[tag_names.index('Brake')]
    gt_gas = ground_truth[tag_names.index('Gas')]
    gt_speed = ground_truth[tag_names.index('Speed')]

    #evaluate original image
    pred_steer, pred_gas, pred_brake, pred_speed, activ_map = eval_image(
        agent, image, previous_speed,
        ground_truth[tag_names.index('Control signal')], gt_steer, gt_gas,
        gt_brake, gt_speed)

    #get clusters and heatmap for activation
    image_activ, activ_map, heat_map, clusters, no_activ_cluster = get_activation_image(
        activ_map, np.copy(image))

    activ_map[activ_map < 50] = 0
    #check a random cluster from image that has activations equal with 0
    '''
    obstructed_image = obstruct_image(
        np.copy(image), no_activ_cluster, image_mean)
    eval_image(agent, obstructed_image, previous_speed,
               ground_truth[tag_names.index('Control signal')], pred_steer,
               pred_gas, pred_brake, pred_speed)

    obstructed_image = cv2.resize(obstructed_image, (720, 460), cv2.INTER_AREA)
    obstructed_image = cv2.cvtColor(obstructed_image, cv2.COLOR_RGB2BGR)
    cv2.imshow("Activations",
               np.concatenate(
                   (image_activ, heat_map, obstructed_image), axis=1))
    cv2.waitKey(0)
    '''
    '''
    #for each cluster obstruct the cluster in the image and evaluate again
    for cluster in clusters:
        obstructed_image = obstruct_image(np.copy(image), cluster, image_mean)
        eval_image(agent, obstructed_image, previous_speed,
                   ground_truth[tag_names.index('Control signal')], pred_steer,
                   pred_gas, pred_brake, pred_speed)

        obstructed_image = cv2.resize(obstructed_image, (720, 460),
                                      cv2.INTER_AREA)
        obstructed_image = cv2.cvtColor(obstructed_image, cv2.COLOR_RGB2BGR)
        cv2.imshow("Activations",
                   np.concatenate(
                       (image_activ, heat_map, obstructed_image), axis=1))

        cv2.waitKey(0)
    '''
    '''
    for i in range(-22, 22, 2):

        transformation_matrix = np.float32([[1, 0, i], [0, 1, 0]])
        image_translated = cv2.warpAffine(
            image.astype(np.uint8), transformation_matrix, image.shape[-2::-1])
        eval_image(agent, image_translated, previous_speed,
                   ground_truth[tag_names.index('Control signal')], pred_steer,
                   pred_gas, pred_brake, pred_speed)

        image_translated = cv2.resize(image_translated, (720, 460),
                                      cv2.INTER_AREA)
        image_translated = cv2.cvtColor(image_translated, cv2.COLOR_RGB2BGR)
        cv2.imshow(cfg.save_image,
                   np.concatenate(
                       (image_activ, heat_map, image_translated), axis=1))
        cv2.waitKey(0)

    for i in range(-22, 22, 2):

        cluster_mask = np.zeros_like(no_activ_cluster)
        cluster_mask[no_activ_cluster > 0] = 1
        translated_image = translate_image_patch(
            np.copy(image), cluster_mask, i, 1)

        _, _, _, _, activ_map_trans = eval_image(
            agent, translated_image, previous_speed,
            ground_truth[tag_names.index('Control signal')], pred_steer,
            pred_gas, pred_brake, pred_speed)

        _, _, heat_map_trans, _, _ = get_activation_image(
            activ_map_trans, np.copy(image))

        translated_image = cv2.resize(translated_image, (720, 460),
                                      cv2.INTER_AREA)
        translated_image = cv2.cvtColor(translated_image, cv2.COLOR_RGB2BGR)
        cv2.imshow(
            cfg.save_image,
            np.concatenate(
                (image_activ, heat_map, translated_image, heat_map_trans),
                axis=1))

        cv2.waitKey(0)

    for i in range(-22, 22, 2):

        cluster_mask = np.zeros_like(activ_map)
        cluster_mask[activ_map > 0] = 1
        translated_image = translate_image_patch(
            np.copy(image), cluster_mask, i, 1)

        _, _, _, _, activ_map_trans = eval_image(
            agent, translated_image, previous_speed,
            ground_truth[tag_names.index('Control signal')], pred_steer,
            pred_gas, pred_brake, pred_speed)

        _, _, heat_map_trans, _, _ = get_activation_image(
            activ_map_trans, np.copy(image))

        translated_image = cv2.resize(translated_image, (720, 460),
                                      cv2.INTER_AREA)
        translated_image = cv2.cvtColor(translated_image, cv2.COLOR_RGB2BGR)
        cv2.imshow(
            cfg.save_image,
            np.concatenate(
                (image_activ, heat_map, translated_image, heat_map_trans),
                axis=1))

        cv2.waitKey(0)
    '''
    
    cv2.imshow(cfg.save_image, np.concatenate((image_activ, heat_map), axis=1))
    cv2.waitKey(0)
    if cfg.save_image != "NetworkActivation":
        cv2.imwrite(
            cfg.save_image, np.concatenate((image_activ, heat_map), axis=1))

    cv2.destroyAllWindows()


def translate_image_patch(image, patch_mask, nr_pixels, direction=1):

    #image_mean = np.array([255, 255, 255]).astype(np.uint8)
    image_mean = np.mean(np.mean(image, axis=0), axis=0)

    #get the patch from image
    patch_image = np.zeros_like(image)
    grey_image = np.zeros_like(image)
    grey_image[:, :] = image_mean

    #kernel = np.ones((abs(nr_pixels) , abs(nr_pixels) ), np.uint8)
    #patch_mask = cv2.dilate(patch_mask, kernel, iterations=1)
    patch_image[patch_mask != 0] = image[patch_mask != 0]

    #in the original image, where was the patch, replace with mean
    #image[patch_mask == 1] = image_mean

    #translate the patch
    transformation_matrix = np.float32([[1, 0, direction * nr_pixels],
                                        [0, 1, 0]])
    patch_image = cv2.warpAffine(patch_image, transformation_matrix,
                                 patch_image.shape[-2::-1])

    #reapply patch on original image
    image[patch_image != 0] = patch_image[patch_image != 0]

    #grey_image[patch_image != 0] = patch_image[patch_image != 0]

    return image


def run_once(args):
    cfg, run_id, path = args

    # -- Set seed
    cfg.general.seed = utils.set_seed(cfg.general.seed)

    # -- Resume agent and metrics if checkpoints are available
    # TODO Resume
    resume_path = path + "/" + cfg.checkpoint
    if resume_path:
        log.info("Network_activation ...")
        cfg.agent.resume = resume_path

    # -- Get agent
    agent = get_agent(cfg.agent)

    if cfg.eval_model is False:
        log.info("Not in eval mode")
        return

    if cfg.image_number != -1:
        eval_network(agent, cfg)
    else:
        pass


if __name__ == '__main__':
    # Initialize logger properties
    StreamHandler(sys.stdout).push_application()
    log.info("[MODE]  Network activations importance")
    log.warn('Logbook is too awesome for most applications')

    # -- Parse config file & generate
    procs_no, arg_list = generate_configs()
    log.info("Starting...")

    run_once(arg_list[0])
