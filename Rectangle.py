class Rectangle:
    
    """ Class for implementation of all attributes of a rectangle """

    def __init__(self, bottom_left, top_right):
        self.bottom_left = bottom_left
        self.top_right = top_right
    def intersects(self, other):
        """Checks if two Rectangle objects insects or overlaps"""
        return not (self.top_right.x < other.bottom_left.x or self.bottom_left.x > other.top_right.x or self.top_right.y < other.bottom_left.y or self.bottom_left.y > other.top_right.y)