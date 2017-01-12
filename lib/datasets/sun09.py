# coding: utf-8
import os
import h5py
import numpy as np
import cv2
import cPickle
import scipy.sparse
from datasets.imdb import imdb

class SUN09(imdb):
    def __init__(self, image_set, year):
        super(SUN09, self).__init__(name='SUN09_{}'.format(image_set))
        
        self._year = year
        self._image_set = image_set
        self._data_path = "/home/danilo/workspace/datasets/Images/static_sun09_database"
        
        with open('/home/danilo/workspace/caffe-libs/train-DeepLab/exper/sun09/list/names.txt', 'r') as fp:
        	classes = {int(line.strip().split()[0]): line.strip().split()[1] for line in fp.readlines()}

        self._classes = classes.values()
        self._class_to_ind = classes.keys()
        self.inv_index = {v: k for k, v in classes.iteritems()}

        self.hdf = h5py.File('/home/danilo/workspace/datasets/sun09_{}.hdf5'.format(image_set), 'r')
        self._image_index = [k for k in self.hdf.keys() \
        if set(self.hdf[k].keys()).intersection(set(self._classes))]
        
        # Default to roidb handler
        self.set_proposal_method('gt')
        self.competition_mode(False)

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        image_path = os.path.join(self._data_path, index)
        assert os.path.exists(image_path), 'Path does not exist: {}'.format(image_path)
        return image_path


    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.
        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            print 'Reading ROI db from cache {}'.format(cache_file)
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        gt_roidb = [self._load_sun09_annotation(index) for index in self._image_index]

        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)

        print 'wrote gt roidb to {}'.format(cache_file)
        return gt_roidb

    def _load_sun09_annotation(self, index):
    	objs = [key for key in self.hdf[index] if key in self.classes]
    	num_objs = len(objs)
    	
        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        seg_areas = np.zeros((num_objs), dtype=np.float32)
      	
      	im = cv2.imread(self.image_path_from_index(index))
      	if im is None:
      		return {}

        for ix, obj in enumerate(objs):
        	cls = self.inv_index[obj]
        	for id_ in self.hdf[index][obj]:
        		cnt = np.array(self.hdf[index][obj][id_])
	        	x1, y1, w, h = cv2.boundingRect(cnt[np.newaxis])		
	        	x2 = x1 + w 
	        	y2 = y1 + h 

	        	xmin = np.min([x1, x2]) - 2 
	        	xmax = np.max([x1, x2]) - 2
	        	ymin = np.min([y1, y2]) - 2
	        	ymax = np.max([y1, y2]) - 2

	        	# weird annotation scheme tool of SUN09
	        	xmin = xmin if xmin >= 0 else 0
	        	xmax = xmax if xmax >= 0 else 0
	        	ymin = ymin if ymin >= 0 else 0
	        	ymax = ymax if ymax >= 0 else 0

	        	xmax = xmax if xmax < im.shape[1] else im.shape[1] - 1
	        	ymax = ymax if ymax < im.shape[0] else im.shape[0] - 1

	        	# make 0-indexed
	        	boxes[ix, :] = np.array([xmin, ymin, xmax, ymax])
	        	gt_classes[ix] = cls
	        	seg_areas[ix] = (xmax - xmin + 1) * (ymax - ymin + 1)
	        	overlaps[ix, cls] = 1.0
    	overlaps = scipy.sparse.csr_matrix(overlaps)

        # ds_utils.validate_boxes(boxes, width=width, height=height)
        # overlaps = scipy.sparse.csr_matrix(overlaps)
        return {'boxes' : boxes,
                'gt_classes': gt_classes,
                'gt_overlaps' : overlaps,
                'flipped' : False,
                'seg_areas' : seg_areas}

    def evaluate_detections(self, all_boxes, output_dir):
        self._write_results_file(all_boxes)
        self._do_python_eval(output_dir)

    def _write_results_file(self, all_boxes):
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            print 'Writing {} VOC results file'.format(cls)
            filename = self._get_voc_results_file_template().format(cls)
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self.image_index):
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    # the VOCdevkit expects 1-based indices
                    for k in xrange(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index, dets[k, -1],
                                       dets[k, 0] + 1, dets[k, 1] + 1,
                                       dets[k, 2] + 1, dets[k, 3] + 1))