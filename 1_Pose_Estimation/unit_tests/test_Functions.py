import unittest
import Functions
import numpy as np
import pandas as pd

class TestContinuity(unittest.TestCase):
    sample_player_name = 'Batter'
    sample_player_key = '%s_player' % (sample_player_name)
    num_frames = 4
    num_joints = 3

    def setUp(self):
        pass

    def test_allzeros(self):
        raise unittest.SkipTest()
        frames = []
        for i in range(self.num_frames):
            frames.append(np.zeros((self.num_joints, 2)))

        df_res = pd.DataFrame(pd.Series(frames), columns=[self.sample_player_key])

        interpolated = Functions.continuity(df_res,
            self.sample_player_name,
            num_joints=self.num_joints)

        # print interpolated

        # Unwrap[0] once more at the end because pandas returns iloc elem wrapped in array.
        center = interpolated.iloc[1].tolist()[0]
        # print center
        self.assertEqual(center[0][0], 0)
        self.assertEqual(center[0][1], 0)
        self.assertEqual(center[1][0], 0)
        self.assertEqual(center[1][1], 0)
        self.assertEqual(center[2][0], 0)
        self.assertEqual(center[2][1], 0)

    def test_simple_input(self):
        frames = []
        for i in range(self.num_frames):
            frames.append(np.zeros((self.num_joints, 2)))

        # 1,2   ...
        # 0,0   ...
        # 3,4   ...

        first_frame = frames[0]
        first_frame[0][0] = 1
        first_frame[0][1] = 2

        last_frame = frames[2]
        last_frame[0][0] = 3
        last_frame[0][1] = 4

        df_res = pd.DataFrame(pd.Series(frames), columns=[self.sample_player_key])

        df_res = Functions.continuity(df_res,
            self.sample_player_name,
            num_joints=self.num_joints)

        # Unwrap[0] once more at the end because pandas returns iloc elem wrapped in array.
        second_frame = df_res[self.sample_player_key].iloc[1]

        self.assertEqual(second_frame[0][0], 2)
        self.assertEqual(second_frame[0][1], 3)

class TestMixLeftRight(unittest.TestCase):

    def test_simple(self):
        raise unittest.SkipTest()
        self.assertEqual(True, True)
        print '\nWARN: Not implemented.'

if __name__ == '__main__':
    unittest.main()