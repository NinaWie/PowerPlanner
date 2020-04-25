import numpy as np


class SplitUtils():

    @staticmethod
    def construct_patches(
        instance, instance_corr, pix_per_part, margin, padding
    ):
        """
        Split instance into two parts
        Arguments:
            instance: 3D numpy array, cost layers
            instance_corr: 2D np array, hard constraints layer
            pix_per_part: length along split axis of each part
            margin: overlapping size
            padding: how much to pad around the instance
        Returns:
            list of two instances, list of both corridors of corresponding
            size
        """
        two_insts = [
            (instance[:, :pix_per_part + margin + padding]).copy(),
            (instance[:, pix_per_part - margin - padding:]).copy()
        ]
        two_corrs = [
            (instance_corr[:pix_per_part + margin]).copy(),
            (instance_corr[pix_per_part - margin:]).copy()
        ]
        pad_zeros = np.zeros((padding, instance_corr.shape[1]))
        two_corrs[0] = np.concatenate((two_corrs[0], pad_zeros), axis=0)
        two_corrs[1] = np.concatenate((pad_zeros, two_corrs[1]), axis=0)
        return two_insts, two_corrs
