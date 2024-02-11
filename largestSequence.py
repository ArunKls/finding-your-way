from prevapproaches import Approaches
from dronefinder import DroneFinder
import random
from copy import deepcopy
import numpy as np

# Class to generate the maze with largest sequence possible
class LargestSequence(Approaches):

    # Reading from input file
    def __init__(self, mazeFile):
        self.maze = []
        self.mazeFile = mazeFile
        with open(mazeFile) as f:
            for line in f.readlines():
                l = list(line.strip())
                for i in range(len(l)):
                    if l[i] == "_":
                        l[i] = 0
                    else:
                        l[i] = 1
                self.maze.append(l)
        self.rows = len(self.maze)
        self.cols = len(self.maze[0])

    # Method to count all the cells across the maze bounded(either by closed cell or wall) on 2 sides and 3 sides
    def boundCellsCount(self):
        twoBounded = []
        threeBounded = []
        for row in range(self.rows):
            for col in range(self.cols):
                count = 0
                # looking across the current cell neighbours
                for k, (nrow, ncol) in enumerate(zip([row-1, row+1, row, row],[col, col, col-1, col+1])):
                    if nrow<0 or nrow>=self.rows:
                        count += 1
                    if ncol<0 or ncol>=self.cols:
                        count += 1
                    if 0 <= nrow < self.rows and 0 <= ncol < self.cols:
                        if self.maze[nrow][ncol] == 1:
                            count += 1
                if count == 2:
                    twoBounded.append((row, col))
                elif count == 3:
                    threeBounded.append((row, col))
        return twoBounded, threeBounded

    # Method to remove cells which are 3 bounded while checking which of those can give highest sequence length
    def removeCells(self, threeBounded):
        # Shuffling the three bounded cells list so we don't end up remove cells completely emptying the topleft area
        random.shuffle(threeBounded)
        df = DroneFinder(self.mazeFile)
        sequence, finalBelief = df.converge(df.belief)
        sequenceLength = len(sequence)
        LargestSequenceMaze = deepcopy(self.maze)
        print(f"Initial maze sequence: {sequence}, size: {sequenceLength}")
        while threeBounded:
            # Removing 1 3 bounded cell from this list
            remove = threeBounded.pop()
            row = remove[0]
            col = remove[1]
            # Iterating across its neighbours to remove one blocked cell
            for k, (nrow, ncol) in enumerate(zip([row-1, row+1, row, row],[col, col, col-1, col+1])):
                    if 0 <= nrow < self.rows and 0 <= ncol < self.cols:
                        if self.maze[nrow][ncol] == 1:
                            self.maze[nrow][ncol] = 0
                            break
            mazeText = []
            for i in range(self.rows):
                mazeRows = ""
                for j in range(self.cols):
                    if self.maze[i][j] == 0:
                        mazeRows += "_"
                    else:
                        mazeRows += "X"
                mazeText.append(mazeRows)
            with open("largestSequence.txt", "w") as f:
                for line in mazeText:
                    f.write(line)
                    f.write("\n")
            df = DroneFinder("largestSequence.txt")
            sequence, finalBelief = df.converge(df.belief)
            if len(sequence) > sequenceLength:
                sequenceLength = len(sequence)
                LargestSequenceMaze = deepcopy(self.maze)
            print(f"Largest sequence till now: {sequence}, length: {sequenceLength}")
        return LargestSequenceMaze



    # Working on existing maze for now
    def generateRandomMaze(self):
        pass


if __name__ == "__main__":
    ls = LargestSequence("Thor23-SA74-VERW-Schematic (Classified).txt")
    twoBounded, threeBounded = ls.boundCellsCount()
    largestSequenceMaze = ls.removeCells(threeBounded)
    print(np.matrix(largestSequenceMaze))