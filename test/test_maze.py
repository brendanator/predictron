import tensorflow as tf
from predictron.maze import MazeGenerator

class MazeTest(tf.test.TestCase):
  def test_random_maze(self):
    for height in range(4,21):
      for width in range(4,21):
        for density in [0.1, 0.3, 0.7]:
          filled = int(
            (height*width - 2) # Corners are never filled
            * density)

          generator = MazeGenerator(height=height, width=width, density=density)
          maze = generator.generate()
          self.assertEqual(bin(maze).count('1'), filled)

  def test_performance(self):
    generator = MazeGenerator(20, 20, 0.3)
    for _ in range(1000):
      maze = generator.generate()
      generator.connected_squares(maze)

  def test_connected_squares(self):
    generator = MazeGenerator(height=4)

    maze1 = int('0100100000000000', base=2)
    connected1 = generator.connected_squares(maze1)
    expected1 = int('0011011111111111', base=2)
    self.assertEqual(connected1, expected1)

    maze2 = int('0000000011110000', base=2)
    connected2 = generator.connected_squares(maze2)
    expected2 = int('0000000000001111', base=2)
    self.assertEqual(connected2, expected2)

    maze3 = int('0001010001110000', base=2)
    connected3 = generator.connected_squares(maze3)
    expected3 = int('1110101110001111', base=2)
    self.assertEqual(connected3, expected3)

  def test_connected_diagonals(self):
    generator = MazeGenerator(height=4)

    maze1 = int('0100100000000000', base=2)
    diagonal1 = generator.connected_diagonals(maze1)
    expected1 = [0, 1, 1, 1]
    self.assertEqual(diagonal1, expected1)

    maze2 = int('0000000011110000', base=2)
    diagonal2 = generator.connected_diagonals(maze2)
    expected2 = [0, 0, 0, 1]
    self.assertEqual(diagonal2, expected2)

    maze3 = int('0001010001110000', base=2)
    diagonal3 = generator.connected_diagonals(maze3)
    expected3 = [1, 0, 0, 1]
    self.assertEqual(diagonal3, expected3)


if __name__ == "__main__":
  tf.test.main()
