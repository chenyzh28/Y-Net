from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral, \
    create_pairwise_gaussian, softmax_to_unary
import pydensecrf.densecrf as dcrf
import torch.nn as nn
import numpy as np

class dense_crf(nn.Module):
    def __init__(self, num_class):
        super(dense_crf, self).__init__()
        self.num_class = num_class

    def forward(self, img, output_probs):
        h = img.shape[1]
        w = img.shape[0]

        d = dcrf.DenseCRF(h * w, self.num_class)
        U = unary_from_labels(output_probs, self.num_class, gt_prob=0.8, zero_unsure=False)
        U = np.ascontiguousarray(U)
        d.setUnaryEnergy(U)


        # for VOC
        feats = create_pairwise_gaussian(sdims=(5, 5), shape=img.shape[:2])
        d.addPairwiseEnergy(feats, compat=3, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
        feats = create_pairwise_bilateral(sdims=(20, 20), schan=(10, 10, 10), img=img, chdim=2)
        d.addPairwiseEnergy(feats, compat=10, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)

        # for Person-Part
        # feats = create_pairwise_gaussian(sdims=(5, 5), shape=img.shape[:2])
        # d.addPairwiseEnergy(feats, compat=3, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
        # feats = create_pairwise_bilateral(sdims=(10, 10), schan=(5, 5, 5), img=img, chdim=2)
        # d.addPairwiseEnergy(feats, compat=10, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)

        Q = d.inference(10)
        Q = np.argmax(np.array(Q), axis=0).reshape((w, h))

        return Q