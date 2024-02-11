import heapq
from copy import deepcopy
from collections import deque

class Approaches():

    def maxDistHorizontal(self, belief):
        maxDist = 0
        for i in range(self.rows):
            start = 0
            count = 0
            for j in range(self.cols):
                if belief[i][j] > 0:
                    count += 1
                if self.maze[i][j] == 1:
                    if count > 0:
                        # print(i, start, j, j-start)
                        maxDist = max(maxDist, j-start)
                    start = j+1
                    count = 0
            for j in range(start, self.cols):
                if belief[i][j] > 0:
                    count += 1
            if count > 0:
                # print(i, start, self.cols, self.cols-start)
                maxDist = max(maxDist, self.cols-start)
        return maxDist

    def maxDistLeft(self, belief):
        maxDist = float("-inf")
        for i in range(self.rows):
            start = self.cols-1
            count = 0
            first = None
            for j in range(self.cols-1, -1, -1):
                if belief[i][j] > 0:
                    count += 1
                    if first is None:
                        first = j
                if self.maze[i][j] == 1:
                    if first is not None and count>1:
                        # print(i, start, j, j-start)
                        maxDist = max(maxDist, first-j-1)
                    start = j-1
                    count = 0
                    first = None
            # first = None
            count = 0
            for j in range(start, -1, -1):
                if belief[i][j] > 0:
                    if not first:
                        first = j
                    count += 1
            if first is not None and count>1:
                # print(i, start, self.cols, self.cols-start)
                maxDist = max(maxDist, first)
            # print(i, maxDist)
        return max(maxDist, 0)

    def maxDistRight(self, belief):
        maxDist = float("-inf")
        for i in range(self.rows):
            start = 0
            count = 0
            first = None
            for j in range(self.cols):
                if belief[i][j] > 0:
                    count += 1
                    if first is None:
                        first = j
                if self.maze[i][j] == 1:
                    if first is not None and count>1:
                        maxDist = max(maxDist, j-first-1)
                    start = j+1
                    count = 0
                    first = None
            # print(i, maxDist)
            # first = None 
            count = 0
            for j in range(start, self.cols):
                if belief[i][j] > 0:
                    count += 1
                    if first is None:
                        first = j
            
            if first is not None and count>1:
                # print(i, start, self.cols, self.cols-start)
                maxDist = max(maxDist, self.cols-first-1)
        return max(maxDist, 0)

    def maxDistLeft1(self, belief):
        maxDist = 0
        for i in range(self.rows):
            start = self.cols-1
            count = 0
            first = None
            for j in range(self.cols-1, -1, -1):
                if belief[i][j] > 0:
                    count += 1
                    if first is None:
                        first = j
                if self.maze[i][j] == 1:
                    if first is not None:
                        # print(i, start, j, j-start)
                        maxDist = max(maxDist, first-j-1)
                    start = j-1
                    count = 0
                    first = None
            # first = None
            count = 0
            for j in range(start, -1, -1):
                if belief[i][j] > 0:
                    if not first:
                        first = j
                    count += 1
            if first is not None:
                # print(i, start, self.cols, self.cols-start)
                maxDist = max(maxDist, first)
            # print(i, maxDist)
        mm = maxDist
        if maxDist>0:
            mc = 0
            for k in range(1, maxDist+1):
                simBelief = deepcopy(belief)
                simBelief = self.move(simBelief, dirs="L"*k)
                
                maxCount = 0
                for j in range(self.cols):
                    start = 0
                    count = 0
                    for i in range(self.rows):
                        if simBelief[i][j] > 0:
                            count += 1
                        if self.maze[i][j] == 1:
                            if count > 1:
                                maxCount += 1
                            start = i+1
                            count = 0
                    count = 0
                    for i in range(start, self.rows):
                        if simBelief[i][j] > 0:
                            count += 1
                    if count > 1:
                        maxCount += 1
                if maxCount >= mc:
                    mc = maxCount
                    mm = k
                # print("k ", maxCount, k)
        return mm

    def maxDistRight1(self, belief):
        maxDist = 0
        for i in range(self.rows):
            start = 0
            count = 0
            first = None
            for j in range(self.cols):
                if belief[i][j] > 0:
                    count += 1
                    if first is None:
                        first = j
                if self.maze[i][j] == 1:
                    if first is not None:
                        maxDist = max(maxDist, j-first-1)
                    start = j+1
                    count = 0
                    first = None
            # print(i, maxDist)
            # first = None 
            count = 0
            for j in range(start, self.cols):
                if belief[i][j] > 0:
                    count += 1
                    if first is None:
                        first = j
            
            if first is not None:
                # print(i, start, self.cols, self.cols-start)
                maxDist = max(maxDist, self.cols-first-1)
        mm = maxDist
        if maxDist>0:
            mc = 0
            for k in range(1, maxDist+1):
                simBelief = deepcopy(belief)
                simBelief = self.move(simBelief, dirs="L"*k)
                
                maxCount = 0
                for j in range(self.cols):
                    start = 0
                    count = 0
                    for i in range(self.rows):
                        if simBelief[i][j] > 0:
                            count += 1
                        if self.maze[i][j] == 1:
                            if count > 1:
                                maxCount += 1
                            start = i+1
                            count = 0
                    count = 0
                    for i in range(start, self.rows):
                        if simBelief[i][j] > 0:
                            count += 1
                    if count > 1:
                        maxCount += 1
                if maxCount >= mc:
                    mc = maxCount
                    mm = k
                # print("k ", maxCount, k)
        return mm

    def maxDistVertical(self, belief):
        maxDist = 0
        for j in range(self.cols):
            start = 0
            count = 0
            for i in range(self.rows):
                if belief[i][j] > 0:
                    count += 1
                if self.maze[i][j] == 1:
                    if count > 0:
                        # print(j, start, i, i-start)
                        maxDist = max(maxDist, i-start)
                    start = i+1
                    count = 0
            for i in range(start, self.rows):
                if belief[i][j] > 0:
                    count += 1
            if count > 0:
                # print(j, start, self.rows, self.rows-start)
                maxDist = max(maxDist, self.rows-start)
        return maxDist

    def maxDistUp(self, belief):
        maxDist = float("-inf")
        for j in range(self.cols):
            start = self.rows-1
            count = 0
            first = None
            for i in range(self.rows-1, -1, -1):
                if belief[i][j] > 0:
                    count += 1
                    if first is None:
                        first = i
                if self.maze[i][j] == 1:
                    if first is not None and count>1:
                        # print(j, start, i, i-start)
                        maxDist = max(maxDist, i-first-1)
                    start = i-1
                    count = 0
                    first = None
            # first = None
            count = 0
            for i in range(start, -1, -1):
                if belief[i][j] > 0:
                    count += 1
                    if first is None:
                        first = i
            if first is not None and count>1:
                # print(j, start, self.rows, self.rows-start)
                maxDist = max(maxDist, first)
        return max(maxDist, 0)

    def maxDistDown(self, belief):
        maxDist = float("-inf")
        for j in range(self.cols):
            start = 0
            count = 0
            first = None
            for i in range(self.rows):
                if belief[i][j] > 0:
                    count += 1
                    if first is None:
                        first = i
                if self.maze[i][j] == 1:
                    if first is not None and count>1:
                        # print(j, start, i, i-start)
                        maxDist = max(maxDist, i-first-1)
                    start = i+1
                    count = 0
                    first = None
            # first = None
            count = 0
            for i in range(start, self.rows):
                if belief[i][j] > 0:
                    count += 1
                    if first is None:
                        first = i
            if first is not None and count>1:
                # print(j, start, self.rows, self.rows-start)
                maxDist = max(maxDist, self.rows-first-1)
        return max(maxDist, 0)

    def maxDistUp1(self, belief):
        maxDist = 0
        for j in range(self.cols):
            start = self.rows-1
            count = 0
            first = None
            for i in range(self.rows-1, -1, -1):
                if belief[i][j] > 0:
                    count += 1
                    if first is None:
                        first = i
                if self.maze[i][j] == 1:
                    if first is not None:
                        # print(j, start, i, i-start)
                        maxDist = max(maxDist, i-first-1)
                    start = i-1
                    count = 0
                    first = None
            # first = None
            count = 0
            for i in range(start, -1, -1):
                if belief[i][j] > 0:
                    count += 1
                    if first is None:
                        first = i
            if first is not None:
                # print(j, start, self.rows, self.rows-start)
                maxDist = max(maxDist, first)
        mm = maxDist
        if maxDist>0:
            mc = 0
            for k in range(1, maxDist+1):
                simBelief = deepcopy(belief)
                simBelief = self.move(simBelief, dirs="U"*k)
                
                maxCount = 0
                for i in range(self.rows):
                    start = 0
                    count = 0
                    for j in range(self.cols):
                        if simBelief[i][j] > 0:
                            count += 1
                        if self.maze[i][j] == 1:
                            if count > 1:
                                maxCount += 1
                            start = j+1
                            count = 0
                    count = 0
                    for j in range(start, self.cols):
                        if simBelief[i][j] > 0:
                            count += 1
                    if count > 1:
                        maxCount += 1
                if maxCount >= mc:
                    mc = maxCount
                    mm = k
                # print("k ", maxCount, k)
        return mm

    def maxDistDown1(self, belief):
        maxDist = 0
        for j in range(self.cols):
            start = 0
            count = 0
            first = None
            for i in range(self.rows):
                if belief[i][j] > 0:
                    count += 1
                    if first is None:
                        first = i
                if self.maze[i][j] == 1:
                    if first is not None and count > 0:
                        # print(j, start, i, i-start)
                        maxDist = max(maxDist, i-first-1)
                    start = i+1
                    count = 0
                    first = None
            # first = None
            count = 0
            for i in range(start, self.rows):
                if belief[i][j] > 0:
                    count += 1
                    if first is None:
                        first = i
            if first is not None and count > 0:
                # print(j, start, self.rows, self.rows-start)
                maxDist = max(maxDist, self.rows-first-1)
        mm = maxDist
        if maxDist>0:
            mc = 0
            for k in range(1, maxDist+1):
                simBelief = deepcopy(belief)
                simBelief = self.move(simBelief, dirs="D"*k)
                
                maxCount = 0
                for i in range(self.rows):
                    start = 0
                    count = 0
                    for j in range(self.cols):
                        if simBelief[i][j] > 0:
                            count += 1
                        if self.maze[i][j] == 1:
                            if count > 1:
                                maxCount += 1
                            start = j+1
                            count = 0
                    count = 0
                    for j in range(start, self.cols):
                        if simBelief[i][j] > 0:
                            count += 1
                    if count > 1:
                        maxCount += 1
                if maxCount >= mc:
                    mc = maxCount
                    mm = k
                # print("k ", maxCount, k)
        return mm
    
    def play(self, belief):
        seq = ""
        while len(seq)<500:
            if self.goalState(belief):
                return seq, belief
            mdh = self.maxDistHorizontal(belief)
            belief = self.move(belief, dirs="L"*mdh)
            seq += "L"*mdh
            if self.goalState(belief):
                return seq, belief
            mdv = self.maxDistVertical(belief)
            belief = self.move(belief, dirs="D"*mdv)
            seq += "D"*mdv
            if self.goalState(belief):
                return seq, belief
            mdh = self.maxDistHorizontal(belief)
            belief = self.move(belief, dirs="R"*mdh)
            seq += "R"*mdh
            if self.goalState(belief):
                return seq, belief
            mdv = self.maxDistVertical(belief)
            belief = self.move(belief, dirs="U"*mdv)
            seq += "U"*mdv
        return seq, belief

    def play0(self, belief):
        if self.goalState(belief):
            return ""
        q = deque()
        q.append(("U", self.move(deepcopy(belief), "U")))
        q.append(("D", self.move(deepcopy(belief), "D")))
        q.append(("L", self.move(deepcopy(belief), "L")))
        q.append(("R", self.move(deepcopy(belief), "R")))
        while q:
            qlen = len(q)
            for i in range(qlen):
                sequence, state = q.popleft()
                print("SEQUENCE", sequence)
                if self.goalState(state):
                    return sequence, state
                if sequence[-1]!="U":
                    q.append((sequence+"D", self.move(deepcopy(state), "D")))
                if sequence[-1]!="D":
                    q.append((sequence+"U", self.move(deepcopy(state), "U")))
                if sequence[-1]!="L":
                    q.append((sequence+"R", self.move(deepcopy(state), "R")))
                if sequence[-1]!="R":
                    q.append((sequence+"L", self.move(deepcopy(state), "L")))
        return "", belief

    def play1(self, belief, display=False):
        seq = ""
        currBelief = deepcopy(belief)
        while len(seq)<100:
            currseq = ""
            # print(seq, belief)
            if self.goalState(belief):
                return seq, belief
            mdl = self.maxDistLeft(belief)
            belief = self.move(belief, dirs="L"*mdl)
            if display:
                self.plotMaze(self.maze, belief)
            # print(seq, belief)
            currseq += "L"*mdl
            
            if self.goalState(belief):
                return seq+currseq, belief

            mdd = self.maxDistDown(belief)
            belief = self.move(belief, dirs="D"*mdd)
            if display:
                self.plotMaze(self.maze, belief)
            # print(seq, belief)
            currseq += "D"*mdd
            
            if self.goalState(belief):
                return seq+currseq, belief

            mdr = self.maxDistRight(belief)
            belief = self.move(belief, dirs="R"*mdr)
            if display:
                self.plotMaze(self.maze, belief)
            # print(seq, belief)
            currseq += "R"*mdr
            
            if self.goalState(belief):
                return seq+currseq, belief
        
            mdu = self.maxDistUp(belief)
            belief = self.move(belief, dirs="U"*mdu)
            if display:
                self.plotMaze(self.maze, belief)
            # print(seq, belief)

            currseq += "U"*mdu

            if self.goalState(belief):
                return seq+currseq, belief
            
            if belief == currBelief:
                break
            currBelief = deepcopy(belief)
            seq += currseq
            
        # print("here", self.maxDistDown1(belief), self.maxDistLeft1(belief), self.maxDistRight1(belief), self.maxDistUp1(belief))
        return seq, belief

    def play2(self, belief, display=False):
        seq = ""
        currBelief = deepcopy(belief)
        print(self.maxDistLeft1(belief), self.maxDistDown1(belief), self.maxDistUp1(belief), self.maxDistDown1(belief))
        while len(seq)<10000000:
            currseq = ""
            # print(seq, belief)
            if self.goalState(belief):
                return seq, belief
            

            mdr = self.maxDistRight1(belief)
            belief = self.move(belief, dirs="R"*mdr)
            if display:
                self.plotMaze(self.maze, belief)
            # print(seq, belief)
            currseq += "R"*mdr
            
            if self.goalState(belief):
                return seq+currseq, belief

            mdd = self.maxDistDown1(belief)
            belief = self.move(belief, dirs="D"*mdd)
            if display:
                self.plotMaze(self.maze, belief)
            # print(seq, belief)
            currseq += "D"*mdd
            
            if self.goalState(belief):
                return seq+currseq, belief

            mdl = self.maxDistLeft1(belief)
            belief = self.move(belief, dirs="L"*mdl)
            if display:
                self.plotMaze(self.maze, belief)
            # print(seq, belief)
            currseq += "L"*mdl
            
            if self.goalState(belief):
                return seq+currseq, belief
        
            mdu = self.maxDistUp1(belief)
            belief = self.move(belief, dirs="U"*mdu)
            if display:
                self.plotMaze(self.maze, belief)
            # print(seq, belief)

            currseq += "U"*mdu

            if self.goalState(belief):
                return seq+currseq, belief
            

            if belief == currBelief:
                break
            currBelief = deepcopy(belief)
            seq += currseq
            
        # print("here", self.maxDistDown1(belief), self.maxDistLeft1(belief), self.maxDistRight1(belief), self.maxDistUp1(belief))
        return seq, belief

    def play3(self, belief):
        seq = ""
        currBelief = deepcopy(belief)
        while True:
            h = []
            heapq.heapify(h)
            mdl = self.maxDistLeft(belief)
            if mdl > 1:
                heapq.heappush(h, [mdl, "L"])
            mdr = self.maxDistRight(belief)
            if mdr>1:
                heapq.heappush(h, [mdr, "R"])
            mdu = self.maxDistUp(belief)
            if mdu > 1:
                heapq.heappush(h, [mdu, "U"])
            mdd = self.maxDistDown(belief)
            if mdd > 1:
                heapq.heappush(h, [mdd, "D"])
            # print(h)
            if not h:
                return seq, belief
            if len(seq) == 0:
                val, moveDir = heapq.heappop(h)
            else:
                val, moveDir = heapq.heappop(h)
                if seq[-1] == "U":
                    if h and moveDir == "D":
                        val, moveDir = heapq.heappop(h)
                elif seq[-1] == "D":
                    if h and moveDir == "U":
                        val, moveDir = heapq.heappop(h)
                elif seq[-1] == "L":
                    if h and moveDir == "R":
                        val, moveDir = heapq.heappop(h)
                elif seq[-1] == "R":
                    if h and moveDir == "L":
                        val, moveDir = heapq.heappop(h)
            belief = self.move(belief, dirs = moveDir*val)
            # if belief == currBelief:
            #     break
            seq += moveDir*val
            print(seq)
            if self.goalState(belief):
                return seq, belief
            currBelief = deepcopy(belief)
        return seq, belief

    def distanceMatrix(self, end):
        q = deque()

        q.append((end, 0))

        visited = set()
        visited.add(end)

        distances = [[(None, None) for i in range(self.cols)] for j in range(self.rows)]
        directions = ["D", "U", "R", "L"]

        while q:
            (row, col), dist = q.popleft()
            for i, (nrow, ncol) in enumerate(zip([row-1, row+1, row, row],[col, col, col-1, col+1])):

                if 0 <= nrow < self.rows and 0 <= ncol < self.cols and (nrow, ncol) not in visited and self.maze[nrow][ncol] != 1:
                    visited.add((nrow, ncol))
                    distances[nrow][ncol] = (directions[i], dist+1)
                    q.append(((nrow, ncol), dist+1))

        return distances

    def play4(self, belief):
        seq = ""
        currBelief = deepcopy(belief)
        while True:
            maxl = float("-inf")
            start, end = None, None
            pathDict = {}
            for i in range(self.rows):
                for j in range(self.cols):
                    if belief[i][j] != 0:
                        path = self.distanceMatrix((i, j))
                        pathDict[(i,j)] = path
                        for ni in range(self.rows):
                            for nj in range(self.cols):
                                if belief[ni][nj] != 0 and path[ni][nj][1] and maxl < path[ni][nj][1]:
                                    maxl = path[ni][nj][1]
                                    start, end = (ni, nj), (i, j)
            if start is None and end is None:
                break
            # print(pathDict.keys(), maxl, start, end)
            startsum, endsum = 0, 0
            for i in range(self.rows):
                for j in range(self.cols):
                    if pathDict[start][i][j][1] is not None and (i, j) != end:
                        startsum += pathDict[start][i][j][1]
                    if pathDict[end][i][j][1] is not None and (i, j) != start:
                        endsum += pathDict[end][i][j][1]
            if startsum > endsum:
                moveDir = pathDict[start][end[0]][end[1]][0]
            else:
                moveDir = pathDict[end][start[0]][start[1]][0]
            belief = self.move(belief, moveDir)
            if belief == currBelief:
                break
            seq += moveDir
            if self.goalState(belief):
                return seq, belief
        return seq, belief