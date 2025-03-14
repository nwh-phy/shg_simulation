import unittest
import numpy as np
from src.tensor_utils import create_tensor, str_to_indices

class TestTensorUtils(unittest.TestCase):

    def test_create_tensor(self):
        components = ['xyz', 'xxy', 'zzz']
        expected_tensor = np.zeros((3, 3, 3))
        expected_tensor[0, 1, 2] = 1.0  # xyz
        expected_tensor[0, 0, 1] = 1.0  # xxy
        expected_tensor[2, 2, 2] = 1.0  # zzz
        
        tensor = create_tensor(components)
        np.testing.assert_array_equal(tensor, expected_tensor)

    def test_str_to_indices(self):
        self.assertEqual(str_to_indices('xyz'), (0, 1, 2))
        self.assertEqual(str_to_indices('xxy'), (0, 0, 1))
        self.assertEqual(str_to_indices('zzz'), (2, 2, 2))
        self.assertEqual(str_to_indices('yzy'), (1, 2, 1))

if __name__ == '__main__':
    unittest.main()