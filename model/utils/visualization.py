import os
from PIL import Image
import numpy as np
import cv2
import torch
import copy
import matplotlib.pyplot as plt
from time import sleep
try:
    from .box_utils import cxcywh_to_x1y1x2y2
except:
    from box_utils import cxcywh_to_x1y1x2y2

# COCO 17-keypoint skeleton edges (indices 0-16)
edge_pairs = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16)
]


def extract_image_from_video(image_path: str):
    '''use ffmpeg to extract image from video
    Args:
        image_path: path to the image, example: 'data/JAAD/images/video_0001/00000.jpg', which is the 0th frame of 'data/JAAD/JAAD_clips/video_0001.mp4'
    '''
    jaad_root, rel_path = image_path.split('images')
    rel_path = rel_path.replace("\\", "/").strip("/").split("/")
    # extract frame name from image_path
    frame_id = int(rel_path[-1].split('.')[0])
    # extract video name from image_path
    video_name = rel_path[-2]
    video_path = os.path.join(jaad_root, "JAAD_clips", video_name + ".mp4")
    assert os.path.exists(video_path), "video path {} does not exist".format(video_path)
    # extract image from video
    os.makedirs(os.path.dirname(image_path), exist_ok=True)
    cmd = "ffmpeg -i {} -vf \"select=eq(n\\,{})\" -vframes 1 {}".format(video_path, frame_id, image_path)
    print(cmd)
    os.system(cmd)
    sleep(1)
    return


def draw_single_pose(img, pose, color=None):
    '''
    Draw a single pedestrian pose in COCO 17-keypoint format.
    pose: np.ndarray or torch.Tensor of shape (17, 2) or (17, 3)
    '''
    if color is None:
        color = (np.random.rand(3) * 255).tolist()
    elif isinstance(color, np.ndarray):
        color = color.tolist()

    if isinstance(pose, torch.Tensor):
        pose = pose.detach().cpu().numpy()
    elif not isinstance(pose, np.ndarray):
        raise TypeError('Unknown pose type {}'.format(type(pose)))

    if pose.ndim == 1:
        if pose.size % 3 == 0 and pose.size // 3 == 17:
            pose = pose.reshape(17, 3)
        else:
            pose = pose.reshape(-1, 2)

    coords = pose[:, :2].astype(np.int32)
    if pose.shape[-1] >= 3:
        valid = (coords.max(axis=-1) > 0) & (pose[:, 2] > 0)
    else:
        valid = coords.max(axis=-1) > 0

    # plot points
    for i, point in enumerate(coords):
        if valid[i]:
            cv2.circle(img, tuple(point.tolist()), radius=3, color=color, thickness=-1)

    for edge in edge_pairs:
        if not valid[edge[0]] or not valid[edge[1]]:
            continue
        cv2.line(img, tuple(coords[edge[0]].tolist()), tuple(coords[edge[1]].tolist()), color=color, thickness=2)
    return img


def vis_pose_on_img(img, poses, color=None):
    '''poses: (T, 17, 2) or (T, 17, 3)'''
    for pose in poses:
        if pose.ndim == 1:
            if pose.size % 3 == 0 and pose.size // 3 == 17:
                pose = pose.reshape(17, 3)
            else:
                pose = pose.reshape(-1, 2)
        img = draw_single_pose(img, pose, color)

    return img


def viz_pose_trajectories(poses, img_root, vid_traj_id, frame_id, img=None, color=None):
    '''
    draw the temporal sequence of poses
    poses: (T, 17, 2) or (T, 17, 3)
    img: np.array
    '''
    frame_id = int(frame_id[-1]) if isinstance(frame_id, (list, tuple, np.ndarray, torch.Tensor)) else int(frame_id)
    # NOTE this only works for JAAD
    vid = vid_traj_id[:10]
    traj_id = vid_traj_id[11:]
    frames_path = os.path.join(img_root, vid)
    if img is None:
        # when img is not provided, extract it from the video using ffmpeg
        image_path = os.path.join(frames_path, str(frame_id).zfill(5) + '.png')
        if not os.path.exists(image_path):
            extract_image_from_video(image_path)
        img = Image.open(image_path)
        img = np.array(img)

    img = vis_pose_on_img(img, poses, color=color)

    return img


class Visualizer():
    def __init__(self, mode='image'):
        self.mode = mode
        self.H, self.W, self.CH = 1080, 1920, 3
        if self.mode == 'image':
            self.img = None
        elif self.mode == 'plot':
            self.fig, self.ax = None, None
        else:
            raise NameError(mode)

    def initialize(self, img_path=None):
        if self.mode == 'image':
            if not os.path.exists(img_path):
                extract_image_from_video(img_path)
            self.img = np.array(Image.open(img_path))
            self.H, self.W, self.CH = self.img.shape
        elif self.mode == 'plot':
            self.fig, self.ax = plt.subplots()

    def visualize(self,
                  inputs,
                  id_to_show=0,
                  normalized=False,
                  bbox_type='x1y1x2y2',
                  color=(255, 0, 0),
                  thickness=4,
                  radius=5,
                  label=None,
                  viz_type='point',
                  viz_time_step=None):
        if viz_type == 'bbox':
            self.viz_bbox_trajectories(inputs, normalized=normalized, bbox_type=bbox_type, color=color, viz_time_step=viz_time_step, thickness=thickness, radius=radius)
        elif viz_type == 'point':
            self.viz_point_trajectories(inputs, color=color, label=label, thickness=thickness, radius=radius)

    def clear(self):
        plt.close()
        self.fig.clear()
        self.ax.clear()
        del self.fig, self.ax

    def save_plot(self, fig_path, clear=True):
        self.ax.set_xlabel('x [px]', fontsize=12)
        self.ax.set_ylabel('y [px]', fontsize=12)
        if self.ax.get_legend_handles_labels()[1]:
            self.ax.legend(fontsize=12)
        plt.savefig(fig_path)
        if clear:
            self.clear()

    def plot_to_image(self, clear=False):
        """
        @brief Convert a Matplotlib figure to a 3D numpy array with RGB channels and return it
        @param fig a matplotlib figure
        @return a numpy 3D array of RGB values
        """
        if self.ax.get_legend_handles_labels()[1]:
            self.ax.legend()
        # draw the renderer
        self.fig.canvas.draw()
        # Get the RGB buffer from the figure
        if hasattr(self.fig.canvas, 'buffer_rgba'):
            buf = np.asarray(self.fig.canvas.buffer_rgba(), dtype=np.uint8)[..., :3].copy()
        else:
            w, h = self.fig.canvas.get_width_height()
            buf = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
        if clear:
            self.clear()
        return buf

    def viz_point_trajectories(self, points, color=(255, 0, 0), label=None, thickness=4, radius=5):
        '''
        points: (T, 2) or (T, K, 2)
        '''
        if isinstance(points, torch.Tensor):
            points = points.detach().cpu().numpy()
        else:
            points = np.asarray(points)
        if color is None:
            color = (255, 0, 0)
        elif isinstance(color, np.ndarray):
            color = color.tolist()
        if self.mode == 'image':
            # plot traj on image
            if len(points.shape) == 2:
                points = points[:, None, :]
            T, K, _ = points.shape
            points = points.astype(np.int32)
            for k in range(K):
                cv2.polylines(self.img, [np.ascontiguousarray(points[:, k, :])], isClosed=False, color=color, thickness=thickness)

                for t in range(T):
                    cv2.circle(self.img, tuple(points[t, k, :].tolist()), color=color, radius=radius, thickness=-1)
        elif self.mode == 'plot':
            # plot traj in matplotlib
            if isinstance(color, (tuple, list, np.ndarray)) and max(color) > 1:
                color = tuple(c / 255.0 for c in color)
            if len(points.shape) == 2:
                self.ax.plot(points[:, 0], points[:, 1], '-o', color=color, label=label)
            elif len(points.shape) == 3:
                # multiple traj as (T, K, 2)
                for k in range(points.shape[1]):
                    label = label if k == 0 else None
                    self.ax.plot(points[:, k, 0], points[:, k, 1], '-', color=color, label=label)
            else:
                raise ValueError('points shape wrong:', points.shape)
            self.ax.axis('equal')

    def draw_single_bbox(self, bbox, color=None):
        '''
        img: a numpy array
        bbox: a list or 1d array or tensor with size 4, in x1y1x2y2 format
        '''
        if color is None:
            color = (np.random.rand(3) * 255).tolist()
        if self.mode == 'image':
            if isinstance(color, np.ndarray):
                color = color.tolist()
            cv2.rectangle(self.img, (int(bbox[0]), int(bbox[1])),
                          (int(bbox[2]), int(bbox[3])), color, 2)
        elif self.mode == 'plot':
            plot_color = color
            if isinstance(plot_color, (tuple, list, np.ndarray)) and max(plot_color) > 1:
                plot_color = tuple(c / 255.0 for c in plot_color)
            rect = plt.Rectangle((float(bbox[0]), float(bbox[1])),
                                 float(bbox[2] - bbox[0]), float(bbox[3] - bbox[1]),
                                 fill=False, edgecolor=plot_color, linewidth=2)
            self.ax.add_patch(rect)

    def viz_bbox_trajectories(self, bboxes, normalized=False, bbox_type='x1y1x2y2', color=None, thickness=4, radius=5, viz_time_step=None):
        '''
        bboxes: (T, 4) or (T, K, 4)
        '''
        if isinstance(bboxes, torch.Tensor):
            bboxes = bboxes.detach().cpu().numpy().astype(np.float32)
        else:
            bboxes = copy.deepcopy(np.asarray(bboxes)).astype(np.float32)
        if len(bboxes.shape) == 2:
            bboxes = bboxes[:, None, :]

        if color is None:
            color = (np.random.rand(3) * 255).tolist()
        elif isinstance(color, np.ndarray):
            color = color.tolist()

        if normalized:
            bboxes[..., [0, 2]] *= self.W
            bboxes[..., [1, 3]] *= self.H
        if bbox_type == 'cxcywh':
            bboxes = cxcywh_to_x1y1x2y2(bboxes)
        elif bbox_type == 'x1y1x2y2':
            pass
        else:
            raise ValueError(bbox_type)
        bboxes = bboxes.astype(np.int32)
        T, K, _ = bboxes.shape

        # also draw the center points
        center_points = (bboxes[..., [0, 1]] + bboxes[..., [2, 3]]) / 2  # (T, K, 2)
        self.viz_point_trajectories(center_points, color=color, thickness=thickness, radius=radius)

        # draw way point every several frames, just to make it more visible
        if viz_time_step is not None:
            bboxes = bboxes[viz_time_step, :]
            if bboxes.ndim == 2:
                bboxes = bboxes[None, :, :]
            T = bboxes.shape[0]
            for t in range(T):
                for k in range(K):
                    self.draw_single_bbox(bboxes[t, k, :], color=color)


if __name__ == "__main__":
    img_path = "data/JAAD/images/video_0007/00006.jpg"
    # test Visualizer
    viz = Visualizer(mode='image')
    viz.initialize(img_path)
    viz.viz_bbox_trajectories(np.array([[100, 100, 200, 200], [200, 200, 300, 300]]), normalized=False, bbox_type='x1y1x2y2', color=(255, 0, 0), thickness=4, radius=5)
    img = viz.img
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
