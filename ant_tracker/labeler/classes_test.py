from classes import *
import numpy as np
import numpy.testing as npt

import unittest

class TestClasses(unittest.TestCase):
    def setUp(self):
        pass
    def testSizeShape(self):
        a = np.array([[0,1,0],
                    [0,1,0],
                    [0,1,0]])
        b = np.array([[0,0,0],
                    [0,1,1],
                    [0,0,0]],dtype="uint8")

        ants = AntCollection(a)

        self.assertEqual(ants.videoSize,b.size)
        self.assertEqual(ants.videoShape,b.shape)
    def testEncodeDecode(self):
        a = np.array([[0,1,0],
                    [0,1,0],
                    [0,1,0]])
        b = np.array([[0,0,0],
                    [0,1,1],
                    [0,0,0]])
        c = np.array([[0,0,1],
                    [0,0,1],
                    [1,0,0]])

        ants = AntCollection(a)

        for _ in range(4):
            ants.newAnt()

        ants.addUnlabeledFrame(1,a)
        ants.addUnlabeledFrame(2,b)
        ants.addUnlabeledFrame(3,c)

        a = np.array([[0,1,0],
                    [0,3,0],
                    [2,2,0]])
        b = np.array([[0,1,1],
                    [0,3,3],
                    [2,0,0]])
        c = np.array([[4,0,1],
                    [2,0,3],
                    [2,0,0]])

        ants.updateAreas(1,a)
        ants.updateAreas(2,b)
        ants.updateAreas(3,c)

        jsonstring = ants.serialize()
        print(jsonstring)
        ants2 = ants.deserialize(jsonstring=jsonstring)

        self.assertEqual(jsonstring,ants2.serialize())
    def testFillNextFrames(self):
        a = np.array([[0,1,0],
                    [0,1,0],
                    [0,0,0]])
        b = np.array([[0,0,0],
                    [0,1,1],
                    [0,0,0]])
        c = np.array([[0,0,1],
                    [0,0,1],
                    [1,0,0]])
        empty = np.array(
            [[0,0,0],
             [0,0,0],
             [0,0,0]])
        first = np.array(
            [[0,3,0],
             [0,3,0],
             [0,0,0]])

        ants = AntCollection(a)

        for _ in range(4):
            ants.newAnt()

        ants.updateAreas(1,first)
        ants.updateAreas(2,empty.copy())
        ants.updateAreas(3,empty.copy())

        # ants.addUnlabeledFrame(1,a)
        ants.addUnlabeledFrame(2,b)
        ants.addUnlabeledFrame(3,c)

        ants.labelFollowingFrames(1,3,conflict_radius=0)

        c_mask = ants.getMask(3)
        expected = np.array([
                    [0,0,3],
                    [0,0,3],
                    [-1,0,0]])
        npt.assert_array_equal(c_mask,expected)


# abyframe = AreasByFrame()
# abyframe.updateArea(1,a)
# abyframe.updateArea(2,b)
# abyframe.updateArea(3,c)
# print(abyframe.encode())

# ant1 = Ant()
# ant1.updateArea(1,a)
# ant1.updateArea(2,b)
# ant1.updateArea(3,c)

# print(ant1.encode())

# ant2 = Ant()
# ant2.updateArea(1,a)
# ant2.updateArea(2,b)
# ant2.updateArea(3,c)

# ant3 = Ant()
# ant3.updateArea(1,a)
# ant3.updateArea(2,b)
# ant3.updateArea(3,c)


# ants = [ant1,ant2,ant3]

# with open("test.txt","w") as fd:
#     print(json.dump({"ants": ants},fd,cls=AntsEncoder,indent=2))

# frame = 1
# mask = np.zeros(ants2.videoShape,dtype='int16')
# for (id, area) in ((ant.id,ant.getArea(frame)) for ant in ants2.ants if ant.getArea(frame) != None):
#     antMask = area.getMask()
#     print("id: %d" % id)
#     print(antMask)
#     mask = mask + antMask*id

# print(mask)




# colored_mask = np.array([
#     [0,1,2],
#     [3,0,0],
#     [0,0,0],
# ])
