import unittest
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loader import DataLoader
import numpy as np

class TestDataLoader(unittest.TestCase):
    def setUp(self):
        self.loader = DataLoader('./non_existent_data_dir')

    def test_mock_dic_sequence(self):
        seq = self.loader.load_dic_sequence()
        self.assertEqual(len(seq), 5)
        self.assertEqual(seq[0].shape, (256, 256))

    def test_mock_fem_maps(self):
        maps = self.loader.load_fem_strain_maps()
        self.assertEqual(len(maps), 4)
        self.assertEqual(maps[0].shape, (256, 256))

if __name__ == '__main__':
    unittest.main()
