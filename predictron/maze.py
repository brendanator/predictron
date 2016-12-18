from bitstring import Bits
import random
import math

class Maze(Bits):
  def __new__(cls, height=20, width=None, density=0.3, binary=None, **kwargs):
    if not width:
      width = height

    if binary:
      maze = super(Maze, cls).__new__(cls, bin='0b'+binary)
    elif kwargs:
      maze = super(Maze, cls).__new__(cls, **kwargs)
    else:
      binary = Maze._random_maze(height, width, density)
      maze = super(Maze, cls).__new__(cls, bin='0b'+binary)

    maze.height = height
    maze.width = width
    return maze

  @classmethod
  def _random_maze(cls, height, width, density):
    # Ensure start and end squares are empty
    non_corner_size = height * width - 2
    count = int(non_corner_size * density)
    walls = ['1'] * count + ['0'] * (non_corner_size - count)
    random.shuffle(walls)
    return '0b0' + ''.join(walls) + '0'

  def __str__(self):
    format = '{:0' + str(self.width) + 'b}'
    rows = [format.format(self[i:i+self.width].uint)
            for i in range(0, self.len, self.width)]
    return '\n'.join(rows)

  def _copy(self):
    copy = super()._copy()
    copy.height = self.height
    copy.width = self.width
    return copy

  def connected_squares(self):
    """Find squares connected to the end square in the maze"""
    empty_squares = ~self
    not_left_edge, not_right_edge, \
      not_top_edge, not_bottom_edge = self._edges()

    current = None
    next = Maze(int=1, length=self.height*self.width)
    while current != next:
      current = next

      left = current << 1 & not_right_edge
      right = current >> 1 & not_left_edge
      up = current << self.width & not_bottom_edge
      down = current >> self.width & not_top_edge

      next = (current | left | right | up | down) & empty_squares

    return current

  def connected_diagonals(self):
    assert self.height == self.width
    connected = self.connected_squares()
    return [connected[(self.height+1) * i] for i in range(self.height)]

  def _edges(self):
    full_columns = '1' * (self.width-1)
    not_left = Maze(binary=('0' + full_columns) * self.height,
                    height=self.height, width=self.width)
    not_right = Maze(binary=(full_columns + '0') * self.height,
                     height=self.height, width=self.width)

    empty_row = '0' * self.width
    full_row = '1' * self.width
    full_rows = full_row * (self.height-1)
    not_top = Maze(binary=empty_row + full_rows,
                   height=self.height, width=self.width)
    not_bottom = Maze(binary=full_rows + empty_row,
                      height=self.height, width=self.width)

    return not_left, not_right, not_top, not_bottom
