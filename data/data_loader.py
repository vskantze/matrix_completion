import numpy as np
import matplotlib.pyplot as plt

class DataHandler:
    def __init__(self,
                 data_path):
        self.data = None
        self.data_path = data_path

    def load_data(self):
        # Load the image
        self.data = plt.imread(elf.data_path)
        return self.data