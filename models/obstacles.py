'''

    Model to detect obstacles on the road, such as 
    pedestrians, vehicles, and other 

'''

class ObstacleDetector():

    def __init__(self):
        self.__data = ""

    def train(self, images, labels):
        pass

    def test(self, original, grayscale):
        return grayscale