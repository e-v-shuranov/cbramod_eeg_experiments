import unittest

import numpy as np

from experiments.channel_permutation import apply_joint_permutation


class ChannelPermutationTest(unittest.TestCase):
    def test_joint_permutation_applies_same_order_to_signal_and_metadata(self):
        x = np.array(
            [
                [
                    [10, 11],
                    [20, 21],
                    [30, 31],
                ]
            ]
        )
        meta = {
            "channel_names": ["Fz", "Cz", "Pz"],
            "coordinates": np.array(
                [
                    [0.0, 1.0],
                    [1.0, 0.0],
                    [0.0, -1.0],
                ]
            ),
            "montage": "toy",
        }
        perm = [2, 0, 1]

        x_perm, meta_perm = apply_joint_permutation(
            x,
            meta,
            perm=perm,
            channel_axis=1,
        )

        np.testing.assert_array_equal(x_perm[0, :, 0], np.array([30, 10, 20]))
        self.assertEqual(meta_perm["channel_names"], ["Pz", "Fz", "Cz"])
        np.testing.assert_array_equal(
            meta_perm["coordinates"],
            np.array(
                [
                    [0.0, -1.0],
                    [0.0, 1.0],
                    [1.0, 0.0],
                ]
            ),
        )
        self.assertEqual(meta_perm["montage"], "toy")

    def test_invalid_permutation_raises(self):
        x = np.zeros((1, 3, 2))

        with self.assertRaises(ValueError):
            apply_joint_permutation(x, ["Fz", "Cz", "Pz"], perm=[0, 0, 1])


if __name__ == "__main__":
    unittest.main()
