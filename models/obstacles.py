'''

    Model to detect obstacles on the road, such as 
    pedestrians, vehicles, and other 

'''

class ObstacleDetector():

    def __init__(self):
        self.__data = ""

    def train(self, grayscale, rgb, labels):
        print('Starting training')
        grayscale[0].show()

    def test(self, grayscale, rgb, labels):
        pass

    def new(self, original, grayscale):
        return grayscale