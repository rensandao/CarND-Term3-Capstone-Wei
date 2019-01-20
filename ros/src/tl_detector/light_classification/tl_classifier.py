from styx_msgs.msg import TrafficLight
import numpy as np
import tensorflow as tf
import rospy

class TLClassifier(object):
    def __init__(self):
	#TODO load classifier
        self.detection_graph = tf.Graph()
        graph_path = './light_classification/model/sim_model/frozen_inference_graph.pb'

        with self.detection_graph.as_default():
            ObjDet_graph_def = tf.GraphDef()
            with tf.gfile.GFile(graph_path, 'rb') as fid:
                serialized_graph = fid.read()
                ObjDet_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(ObjDet_graph_def, name='')

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        with self.detection_graph.as_default():
            with tf.Session(graph=self.detection_graph) as sess:
  		
                image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
                detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
                detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
                detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

                image_np_expanded = np.expand_dims(image, axis=0)

                (boxes, scores, classes, num) = sess.run([detection_boxes, detection_scores, detection_classes, 
							num_detections], 
							feed_dict={image_tensor: image_np_expanded})

                boxes = np.squeeze(boxes)
                scores = np.squeeze(scores)
                classes = np.squeeze(classes).astype(np.int32)

                green_score = 0  ## initialize scores
                red_score = 0  
                yellow_score = 0  
                is_detected = False
		
		## set vaule for three light colors: green, red, yellow 
                for color, score in zip(classes, scores):
                    ## should higher than a certain score 
		    if score > 0.5:
                        is_detected = True
		
                        if color == 1:
                            green_score = green_score + score
                        if color == 2:
                            red_score = red_score + score
                        if color == 3:
                            yellow_score = yellow_score + score
		## once been dectected, calculate the corresponding score and show log information.
                if is_detected:
                    sums = np.array([green_score, red_score, yellow_score])
                    idx = np.argsort(sums)[-1]
                    rospy.loginfo('The Front Light Index: {}'.format(idx))

                    if idx == 0:
                        return TrafficLight.GREEN
                    if idx == 1:
                        return TrafficLight.RED
                    if idx == 2:
                        return TrafficLight.YELLOW

        return TrafficLight.UNKNOWN
