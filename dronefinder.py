from collections import deque
import matplotlib.pyplot as plt
from copy import deepcopy
import heapq
import numpy as np
from prevapproaches import Approaches
import sys

# Class to find the drone
class DroneFinder(Approaches):
    def __init__(self, mazeFile):
        self.maze = []
        self.belief = []
        self.open = 0
        with open(mazeFile) as f:
            for line in f.readlines():
                l = list(line.strip())
                b = []
                for i in range(len(l)):
                    if l[i] == "_":
                        l[i] = 0
                        # Keeping a track of number of open cells since I am initializing probabilities as 1 instead of 1/open cells
                        self.open += 1
                        # Tracking probabilities with 1
                        b.append(1)
                    else:
                        l[i] = 1
                        b.append(0)
                self.maze.append(l)
                self.belief.append(b)
        self.rows = len(self.maze)
        self.cols = len(self.maze[0])

        # Initializing maze read from file, initial belief state, dimensions as class variables
    
    # Method to plot the maze -- buggy with cmap issues. prints worked for me for debugging
    def plotMaze(self, belief):
        # for i in range(self.rows):
        #     for j in range(self.cols):
        #         if grid[i][j] == 1:
        #             belief[i][j] = 10**-4
        # cmap = plt.get_cmap('gray')
        # cmap.set_under('red')
        plt.imshow(belief)
        plt.ginput()

    # Method to make a move by displacing probability belief values
    # Takes either single letter input or batch of direction commands
    def move(self, belief, dir="", dirs=""):
        if dir=="":
            for i in dirs:
                # Executing move recursively for every direction in batch
                belief = self.move(belief, dir=i)
            return belief
        # Initializing next probability state as nextBelief
        nextBelief = [[0 for i in range(self.cols)] for j in range(self.rows)]
        if dir == "U":
            # If direction is up, iterating over rows from bottom up, displacing probabiity values upward through every column
            for i in range(self.rows-2, -1, -1):
                for j in range(self.cols):
                    if self.maze[i][j] == 0:
                        if(self.maze[i+1][j] == 0):
                            nextBelief[i][j] = belief[i+1][j]
                    else:
                        nextBelief[i+1][j] +=  belief[i+1][j]
            # for the first row, adding previously first row probability values as drone cannot move out of grid upwards
            for i in range(self.cols):
                if self.maze[0][i] == 0:
                    nextBelief[0][i] += belief[0][i]
        elif dir == "D":
            # If direction is down, iterating over rows, displacing probabiity values downward through every column
            for i in range(1, self.rows):
                for j in range(self.cols):
                    if self.maze[i][j] == 0:
                        if(self.maze[i-1][j] == 0):
                            nextBelief[i][j] = belief[i-1][j]
                    else:
                        nextBelief[i-1][j] +=  belief[i-1][j] 
            # for the last row, adding previously last row probability values as drone cannot move out of grid downwards
            for i in range(self.cols):
                if self.maze[self.rows-1][i] == 0:
                    nextBelief[self.rows-1][i] += belief[self.rows-1][i]
        elif dir == "L":
            # If direction is left, iterating over columns, displacing probabiity leftward through every row
            for j in range(self.cols-2, -1, -1):
                for i in range(self.rows):
                    if self.maze[i][j] == 0:
                        if(self.maze[i][j+1] == 0):
                            nextBelief[i][j] = belief[i][j+1]
                    else:
                        nextBelief[i][j+1] +=  belief[i][j+1]
            # for the first column, adding previously first column probability values as drone cannot move out of grid leftwards
            for i in range(self.rows):
                if self.maze[i][0] == 0:
                    nextBelief[i][0] += belief[i][0]
        elif dir == "R":
            # If direction is right, iterating over column, displacing probabiity values rightward through every row
            for j in range(1, self.cols):
                for i in range(self.rows):
                    if self.maze[i][j] == 0:
                        if(self.maze[i][j-1] == 0):
                            nextBelief[i][j] = belief[i][j-1]
                    else:
                        nextBelief[i][j-1] +=  belief[i][j-1] 
            # for the last column, adding previously last column probability values as drone cannot move out of grid rightwards
            for i in range(self.rows):
                if self.maze[i][self.cols-1] == 0:
                    nextBelief[i][self.cols-1] += belief[i][self.cols-1]
        return nextBelief

    # Method to check if we reached the desired goal state by iterating through belief for number of open cells value    
    def goalState(self, belief):
        for i in range(self.rows):
            for j in range(self.cols):
                # Checking for number of open cells value
                if belief[i][j] == self.open:
                    return True
        return False

    # Method to define heuristic as the tuple: 
    # (maximum distance between 2 non 0 belief values + 2*number of non 0 belief values, 
    # maximum distance between 2 non 0 belief values, 
    # length of "fence" over all non zero belief values
    # number of non 0 belief values)
    # Detailed in report
    def heuristic(self, belief):
        # Compiling all non zero belief value indices
        beliefList = []
        for i in range(self.rows):
            for j in range(self.cols):
                if belief[i][j] != 0:
                    beliefList.append((i, j))
        # beliefList.sort(key=lambda x: [x[0], x[1]])
        if not beliefList:
            return 0
        # Method to return positive if q is more anti-clockwise to p than r, else negative. We are doing this with vector product
        def orientation(p, q, r):
            return (q[1]-p[1])*(r[0]-q[0])-(q[0]-p[0])*(r[1]-q[1])
        boundary = []
        # Iterating over non zero belief points and adding all less relatively counterclockwise points to stack
        for i in range(len(beliefList)):
            while len(boundary) >= 2 and orientation(boundary[-2], boundary[-1], beliefList[i]) > 0:
                boundary.pop()
            boundary.append(beliefList[i])
        boundary.pop()
        # Repeating the reverse direction
        for i in range(len(beliefList)-1, -1, -1):
            while len(boundary) >= 2 and orientation(boundary[-2], boundary[-1], beliefList[i]) > 0:
                boundary.pop()
            boundary.append(beliefList[i])
        # Boundary now represents the points through which the "fence" surrounding these points runs
        # This indicates the amount of convergence of non zero belief values
        hlen = 0
        # Finding the perimeter length for this fence
        for i in range(1, len(boundary)):
            hlen += abs(boundary[i][1]-boundary[i-1][1]) + abs(boundary[i][0]-boundary[i-1][0])
        maxDist = 1
        # Iterating over this boundary to find the maximum length "diameter" for any two points
        for i in range(len(boundary)):
            for j in range(i, len(boundary)):
                maxDist = max(maxDist, abs(boundary[i][0]-boundary[j][0])+abs(boundary[i][1]-boundary[j][1]))
        maxl = 0
        # Using BFS to find the maximum length between any two points in above boundary. It gets us distance mindful of blocked cells
        for i in range(len(beliefList)):
            distances = self.distanceMatrix(beliefList[i])
            for j in range(i, len(beliefList)):
                if distances[beliefList[j][0]][beliefList[j][1]] and maxl < distances[beliefList[j][0]][beliefList[j][1]]:
                    maxl = distances[beliefList[j][0]][beliefList[j][1]]
        # Returning heuristic tuple
        return maxl+2*len(beliefList), maxDist, hlen, len(beliefList)

    # Method to convert belief 2d list to column as list is unhashable. This is done to add current belief in visited states during Best First Search
    def convertTuple(self, belief):
        tup = []
        for i in belief:
            tup.append(tuple(i))
        return tuple(tup)

    # Method to return a distance matrix that contains distances from a cell to all other cells for given cell
    def distanceMatrix(self, cell):
        q = deque()

        q.append((cell, 0))

        visited = set()
        visited.add(cell)

        distances = [[0 for i in range(self.cols)] for j in range(self.rows)]
        while q:
            (row, col), dist = q.popleft()
            for i, (nrow, ncol) in enumerate(zip([row-1, row+1, row, row],[col, col, col-1, col+1])):

                if 0 <= nrow < self.rows and 0 <= ncol < self.cols and (nrow, ncol) not in visited and self.maze[nrow][ncol] != 1:
                    visited.add((nrow, ncol))
                    # Uppdating distance for every cell level by level
                    distances[nrow][ncol] = dist+1
                    q.append(((nrow, ncol), dist+1))

        return distances

    # Method to apply Best First Search algorithm by taking above computed heuristic
    def converge(self, belief):
        heuristic = self.heuristic(belief)
        h = []
        # Initializing priority queue
        heapq.heapify(h)
        heapq.heappush(h, (self.heuristic(belief), "", belief))
        # Initializing visited set
        vis = set()
        # Adding initial belief state to visited set
        vis.add(self.convertTuple(belief))
        elem = [0, "", None]
        # Doing Best First Search until current popped sequence length is 1000. Increase for bigger inputs
        while len(elem[1])<1000:
            elem = heapq.heappop(h)
            # If we managed to converge all beliefs, exit
            if self.goalState(elem[2]):
                return elem[1], elem[2]

            # Printing progress bar for convergence. Very rudimentary. Does not work properly for states converging in more than 200-250 sequence
            sys.stdout.write('\r')
            sys.stdout.write("Finding drone ... [%-20s] %d%%" % ('#'*(len(elem[1])//10), len(elem[1])//2))
            sys.stdout.flush()



            # If current sequence is empty, add all 4 dirs movements to the heap queue
            # if elem[1] == "":
            for i in ["L", "D", "R", "U"]:
                newbelief = self.move(deepcopy(elem[2]), i)
                # Adding only to queue if it is not already visited
                if self.convertTuple(newbelief) not in vis: 
                    vis.add(self.convertTuple(newbelief))
                    heapq.heappush(h, [self.heuristic(newbelief), elem[1] + i, newbelief])
            # # else, check if previous move is left, try not to move to right in that case.
            # # We are doing this to decrease redundant moves. LR next to each other seems equivalent to wiggling in place
            
            
            # Never mind. Adding all moves to queue is giving me better convergence
            
            
            # else:
            #     if elem[1][-1] != "L":
            #         newbelief = self.move(deepcopy(elem[2]), "R")

            #         if self.convertTuple(newbelief) not in vis: 
            #             vis.add(self.convertTuple(newbelief))
            #             heapq.heappush(h, [self.heuristic(newbelief), elem[1] + "R", newbelief])
            #     if elem[1][-1] != "D":
            #         newbelief = self.move(deepcopy(elem[2]), "U")

            #         if self.convertTuple(newbelief) not in vis: 
            #             vis.add(self.convertTuple(newbelief))
            #             heapq.heappush(h, [self.heuristic(newbelief), elem[1] + "U", newbelief])
            #     if elem[1][-1] != "R":
            #         newbelief = self.move(deepcopy(elem[2]), "L")

            #         if self.convertTuple(newbelief) not in vis: 
            #             vis.add(self.convertTuple(newbelief))
            #             heapq.heappush(h, [self.heuristic(newbelief), elem[1] + "L", newbelief])
            #     if elem[1][-1] != "U":
            #         newbelief = self.move(deepcopy(elem[2]), "D")

            #         if self.convertTuple(newbelief) not in vis: 
            #             vis.add(self.convertTuple(newbelief))
            #             heapq.heappush(h, [self.heuristic(newbelief), elem[1] + "D", newbelief])
            # heuristic, moveDir, belief = heapq.heappop(h)
            # seq += moveDirif e
        print("\n")
        self.heuristic(elem[2]) 
        return elem[1], elem[2]


if __name__ == "__main__":
    # File location input to class during class object creation
    
    df = DroneFinder("Thor23-SA74-VERW-Schematic (Classified).txt")
    # df = DroneFinder("test.txt")
    print(df.open)
    print(df.belief)
    
    # Running Best First Search here
    seq, belief = df.converge(df.belief)
    print(seq, len(seq))
    
    print("--------")
    s = []
    # Printing converged belief with walls for better debugging
    prbelief = deepcopy(belief)
    for i in range(df.rows):
        for j in range(df.cols):
            if belief[i][j] != 0:
                s.append(belief[i][j])
            if df.maze[i][j] == 1:
                prbelief[i][j] = -1

    
    print(np.matrix(prbelief))
    print(len(s), s, sum(s))
    if len(s) == 1:
        for i in range(df.rows):
            for j in range(df.cols):
                if belief[i][j] == df.open:
                    print(f"Drone found at [{i},{j}] in {len(seq)} steps with command sequence: {seq}")
                    break
    print("========")
    