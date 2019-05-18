import numpy as np
import torch


class Normalize(object):


    def __call__(self, xy_real):

    	# choose the central hip as central point
    	central_point = xy_real[8]

    	xy_real -= central_point

    	scale_array = numpy.linalg.norm(xy_real - central_point).reshape(1, -1)

    	scale_factor = np.mean(scale_array, axis=0)

    	xy_normalized = xy_real / scale_factor

    	return xy_normalized


class ToTensor(object):

    def __call__(self, xy_real):
        return torch.from_numpy(xy_real)



        


























