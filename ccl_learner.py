import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

def visualize_labels(image, title = "Labeled Image"):
    """
    Visualizes a labeled image where 0 is the background and other numbers are labels.
    """
    if image.max() == 0:
        print("No labels found in the image.")
        cmap = colors.ListedColormap(['black'])
    else:
        # create random colors for each label and ensure there are enough for numbers of lebel
        unique_labels = np.unique(image)
        num_labels = len(unique_labels) - 1
        
        # start black as background
        cmap_colors = ['black']
        
        for _ in range(num_labels):
            cmap_colors.append(np.random.rand(3,))
        cmap = colors.ListedColormap(cmap_colors)
        
        # create a normalizer to map label values to colormap indices
        
        bounds = np.concatenate(([image.min()-0.5], np.sort(unique_labels) + 0.5))
        norm = colors.BoundaryNorm(bounds, cmap.N)
        
    plt.imshow(image, cmap=cmap, norm=norm if image.max() > 0 else None, interpolation='nearest')
    plt.title(title)
    plt.colorbar(ticks=np.unique(image))
    plt.show()         
    
class TwoPassCCL:
    def __init__(self, connectivity=8):
        if connectivity not in [4,8]:
            raise ValueError("Connectivity must be 4 or 8.")
        self.connectivity = connectivity
        self.labels = None
        self.equivalences = {}
        self.next_label = 1
        print(f"CCL Processor initialized with {self.connectivity}-connectivity.")
    def _reset(self):
        self.labels = None
        self.equivalences = {}
        self.next_label = 1
        print("CCL state reset.")
    
    # def find_representative(self, label):
    #     """
    #     Find the root representative for a given label using path compression.
    #     """
    #     if label not in self.equivalences:
    #         return label
        
    #     #path to find the root
    #     path = [label]
    #     root = self.equivalences[label]
    #     while root != self.equivalences[root]:
    #         path.append(root)
    #         root = self.equivalences[root] 
    def first_pass(self, image):
        """
        Performs the first pass of the CCL alorithm.
        Assigns provisional lebels and records equivalences.
        """
        rows, cols = image.shape
        #init the label image
        self.labels = np.zeros_like(image, dtype = int)      
        
        for r in range(rows):
            for c in range(cols):
                if image[r, c] == 1: # foreground pixel
                    neighbors_labels = []
                    
                    # for 4 connectivity
                    # top (r-1, c) this is the 4 connectivith
                    if r > 0 and self.labels[r-1, c] > 0:
                        neighbors_labels.append(self.labels[r-1, c])
                    
                    # left (r, c-1) this is the 4 connectivith
                    if c > 0 and self.labels[r, c-1] > 0:
                        neighbors_labels.append(self.labels[r, c-1])