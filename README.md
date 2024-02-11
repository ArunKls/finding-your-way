# Finding Your Way

---

AI Final Exam Assignmant

---

### Brief:

In this scenario, we have a maze-like grid, where we must determine the position of a drone with 100% confidence. We can issue directional commands to this drone.
 
Hence the objective can be better put as:
- Issue probability of drone being present in a cell (which is open) as (1/number of open cells). In the case of my code, for better readability, I am assigning 1 and checking for the number-of-open-cells value to stop my search.
- Issue a minimal number of commands such that all these probabilities add up and converge to 1 (or, in my case, number-of-open-cells), so we can locate the drone with 100% probability.
