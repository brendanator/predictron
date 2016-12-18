import tensorflow as tf
from predictron.maze import Maze

class MazeTest(tf.test.TestCase):
  def test_random_maze(self):
    for height in range(4,21):
      for width in range(4,21):
        for density in [0.1, 0.3, 0.7]:
          filled = int(
            (height*width - 2) # Corners are never filled
            * density)

          maze = Maze(height=height, width=width, density=density)
          self.assertEqual(maze.count(1), filled)

  def test_connected_squares(self):
    maze1 = Maze(binary='0100100000000000', height=4)
    connected1 = maze1.connected_squares()
    expected1 = Maze(binary='0011011111111111', height=4)
    self.assertEqual(connected1, expected1)

    maze2 = Maze(binary='0000000011110000', height=4)
    connected2 = maze2.connected_squares()
    expected2 = Maze(binary='0000000000001111', height=4)
    self.assertEqual(connected2, expected2)

    maze3 = Maze(binary='0001010001110000', height=4)
    connected3 = maze3.connected_squares()
    expected3 = Maze(binary='1110101110001111', height=4)
    self.assertEqual(connected3, expected3)

  def test_connected_diagonals(self):
    maze1 = Maze(binary='0100100000000000', height=4)
    diagonal1 = maze1.connected_diagonals()
    expected1 = [False, True, True, True]
    self.assertEqual(diagonal1, expected1)

    maze2 = Maze(binary='0000000011110000', height=4)
    diagonal2 = maze2.connected_diagonals()
    expected2 = [False, False, False, True]
    self.assertEqual(diagonal2, expected2)

    maze3 = Maze(binary='0001010001110000', height=4)
    diagonal3 = maze3.connected_diagonals()
    expected3 = [True, False, False, True]
    self.assertEqual(diagonal3, expected3)


if __name__ == "__main__":
  tf.test.main()
