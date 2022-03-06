import hmac
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from climbing_thing.climbnet import Instances
import climbing_thing.utils.image as imutils
import numpy as np
import cv2
import sklearn.cluster

class HistogramClustering:
    def __init__(self, target_hold) -> None:
        """
        :param target_hold 
        """
        self.target_hold = target_hold
        self.clusterer = None

    def segment_route(self, image, holds, cluster_idx) -> Instances:
        """Return Instances of holds that are of specified route color"""
        if self.clusterer is not None:
            return Instances(holds.instances[[idx for idx in range(len(holds)) if (self.clusterer.labels_[idx] == cluster_idx)]])
       
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        samples = []
        for idx, mask in enumerate(holds.masks):
            mask = mask.to('cpu')
            mask = np.array(mask.long()).astype(np.uint8)
            hold_bbox = holds.boxes[idx].tensor.int()

            masked_image = hsv_image[mask > 0]
            h_mean = self.circular_mean(masked_image[..., 0]*2) / 360
            sv_mean = masked_image[..., 1:].mean(axis=0) / 255
            feature_vector = np.concatenate([[h_mean], sv_mean], 0) 
            samples.append(feature_vector)

            # h_hist = cv2.calcHist([hsv_image[...,0]], [0], mask, [180], [0,180])
            # s_hist = cv2.calcHist([hsv_image[...,1]], [0], mask, [180], [0,256])
            # v_hist = cv2.calcHist([hsv_image[...,2]], [0], mask, [180], [0,256])
            # feature_vector = np.concatenate([h_hist, s_hist, v_hist], 0) / mask.sum()
            # samples.append(h_hist[..., 0] / mask.sum())

            # fig, axs = plt.subplots(1,2)
            # axs[0].imshow(image[..., ::-1][hold_bbox[0,1]:hold_bbox[0,3], hold_bbox[0,0]:hold_bbox[0,2]])
            # axs[1].plot(h_hist, color="b")
            # axs[1].plot(s_hist, color="g")
            # axs[1].plot(v_hist, color="r")
            # plt.xlim((0,180))
            # plt.show(block=True)

        clusterer = sklearn.cluster.KMeans(n_clusters=7)
        self.clusterer = clusterer.fit(samples)

        return Instances(holds.instances[[idx for idx in range(len(holds)) if (self.clusterer.labels_[idx] == cluster_idx)]])
           

    @staticmethod
    def circular_mean(angles):
        return np.arctan2(np.sin(angles).mean(), np.cos(angles).mean())

    def get_masked_route(self):
        binary_mask = self.holds.combine_masks()
        masked_route = imutils.mask(self.route_image, binary_mask)
        return masked_route
