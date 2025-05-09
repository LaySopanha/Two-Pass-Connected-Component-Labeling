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
    
    def find_representative(self, label):
        """
        Find the root representative for a given label using path compression.
        """
        if label not in self.equivalences:
            return label
        
        #path to find the root
        path = [label]
        root = self.equivalences[label]
        while root != self.equivalences[root]:
            path.append(root)
            root = self.equivalences[root] 
        # path compression" make all node in the path point to the root
        for l_in_path in path:
            self.equivalences[l_in_path] = root
            return root
    def union(self, label1, label2):
        """
        Merges the sets containing label1 and label2.
        The Representative of the merged set will be the minimum of their current representatives.
        """
        
        rep1 = self.find_representative(label1)
        rep2 = self.find_representative(label2)
        
        if rep1 != rep2:
            # make smaller representative parent the larger one
            if rep1 < rep2:
                self.equivalences[rep2] = rep1
            else:
                self.equivalences[rep1] = rep2
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
                    # Neighbor 1: top (r-1, c) this is the 4 connectivith
                    if r > 0 and self.labels[r-1, c] > 0:
                        neighbors_labels.append(self.labels[r-1, c])
                    
                    # Neighbor 2: left (r, c-1) this is the 4 connectivith
                    if c > 0 and self.labels[r, c-1] > 0:
                        neighbors_labels.append(self.labels[r, c-1])
                    
                    # for 8 connectivity
                    if self.connectivity == 8:
                        # Neighbor 3: top-left (r-1, c-1)
                        if r > 0 and c > 0 and self.labels[r-1, c-1] > 0:
                            neighbors_labels.append(self.labels[r-1, c-1])
                        
                        # Neighbor 4: top-right (r-1, c+1)
                        if r > 0 and c < cols - 1 and self.labels[r-1, c+1] > 0:
                            neighbors_labels.append(self.labels[r-1, c+1]) 
                    if not neighbors_labels:
                        # case 1: no foreground neighbors
                        self.labels[r, c] = self.next_label
                        self.equivalences[self.next_label] = self.next_label #point to itself
                        self.next_label += 1
                    else:
                        # case 2, 3 ,4
                        min_neighbor_label = min(neighbors_labels)
                        self.labels[r, c] = min_neighbor_label
                        
                        for l in neighbors_labels:
                            if l != min_neighbor_label:
                                self.union(min_neighbor_label, l)
    def second_pass(self, image):
        """
        Performs the second pass. Replaces provisional labels with their
        final representative labels.
        """
        rows, cols = image.shape
        final_labels = np.zeros_like(self.labels)

        for r in range(rows):
            for c in range(cols):
                if self.labels[r, c] > 0: # If it was a foreground pixel
                    final_labels[r, c] = self.find_representative(self.labels[r, c])
        
        self.labels = final_labels
        # This makes visualization nicer and is common practice.
        unique_final_labels = np.unique(self.labels)
        # print("Unique final labels:", unique_final_labels)
        
        current_new_label = 1
        label_map = {} # old_label -> new_consecutive_label
        
        for old_label in unique_final_labels:
            if old_label == 0: # Skip background
                label_map[0] = 0
                continue
            label_map[old_label] = current_new_label
            current_new_label += 1

        consecutive_labels_image = np.zeros_like(self.labels)
        for r in range(rows):
            for c in range(cols):
                consecutive_labels_image[r,c] = label_map[self.labels[r,c]]
        
        self.labels = consecutive_labels_image

    def label_components(self, image):
        """
        Main function to perform CCL on the binary image.
        Image should be a 2D Numpy array with 0 for background, 1 for foreground.
        Return the labeled image.
        """
        self._reset() # reset state for new image
        if image.ndim != 2:
            raise ValueError("Input image must be 2D.")
        if not ((image == 0) | (image == 1)).all():
            raise ValueError("Input image must be binary (0s and 1s.)")
        
        self.first_pass(image)
        if not self.equivalences: # no foreground pixels found
            self.labels = np.zeros_like(image, dtype=int)
            return self.labels
        
        self.second_pass(image)
        
        return self.labels