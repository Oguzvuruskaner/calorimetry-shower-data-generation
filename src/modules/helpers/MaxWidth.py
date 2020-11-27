class MaxWidth:

    def __init__(self,max_width):
        self.max_width = max_width

    def get_width(self,value):
        return min(self.max_width,value)