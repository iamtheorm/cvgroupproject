import os
import glob
import cv2
import numpy as np

class DataLoader:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.dic_images_dir = os.path.join(data_dir, 'dic_images')
        self.fem_maps_dir = os.path.join(data_dir, 'fem_strain_maps')

    def load_dic_sequence(self):
        '''Loads a sequence of DIC images from the dic_images directory.'''
        if not os.path.exists(self.dic_images_dir):
            return self._generate_mock_dic_sequence()

        image_files = sorted(glob.glob(os.path.join(self.dic_images_dir, '*.png')))
        if not image_files:
            return self._generate_mock_dic_sequence()

        sequence = []
        for file in image_files:
            img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                sequence.append(img)
        return sequence

    def load_fem_strain_maps(self):
        '''Loads synthetic FEM strain maps.'''
        if not os.path.exists(self.fem_maps_dir):
            return self._generate_mock_fem_maps()

        map_files = sorted(glob.glob(os.path.join(self.fem_maps_dir, '*.npy')))
        if not map_files:
            return self._generate_mock_fem_maps()

        maps = []
        for file in map_files:
            maps.append(np.load(file))
        return maps

    def _generate_mock_dic_sequence(self):
        '''Generates a mock DIC sequence (random speckle pattern with slight shifts).'''
        print("Warning: Real DIC images not found. Using mock data.")
        base_pattern = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
        sequence = [base_pattern]
        
        # Create subtle shifts to simulate micro-deformation
        for i in range(1, 5):
            shifted = np.roll(base_pattern, shift=i, axis=1) # Shift horizontally
            # Add some slight structural variation
            shifted = cv2.GaussianBlur(shifted, (3, 3), 0)
            sequence.append(shifted)
            
        return sequence

    def _generate_mock_fem_maps(self):
        '''Generates mock FEM strain maps for training/validation.'''
        print("Warning: Real FEM maps not found. Using mock data.")
        maps = []
        for _ in range(4): # 1 less than sequences, representing strain between frames
            # Generate a 2D gaussian distribution as a mock strain concentration
            x, y = np.meshgrid(np.linspace(-1,1,256), np.linspace(-1,1,256))
            d = np.sqrt(x*x+y*y)
            sigma, mu = 0.5, 0.0
            strain = np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) )
            maps.append(strain)
        return maps
