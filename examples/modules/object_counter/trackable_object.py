class TrackableObject:
    def __init__(self, object_id, centroid):
        # store the object ID, then initialize a list of centroids
        # using the current centroid
        self.object_id = object_id
        self.centroids = [centroid]

        # initialize a boolean used to indicate if the object has
        # already been counted or not
        self.counted = False
