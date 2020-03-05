import numpy as np

# height profile constraints:
## simply exclude edges which cannot be placed --> only works when iterating over edges


# find pixels on line between
# https://stackoverflow.com/questions/50995499/generating-pixel-values-of-line-connecting-2-points
def bresenham_line(x0, y0, x1, y1):
    steep = abs(y1 - y0) > abs(x1 - x0)
    if steep:
        x0, y0 = y0, x0
        x1, y1 = y1, x1

    switched = False
    if x0 > x1:
        switched = True
        x0, x1 = x1, x0
        y0, y1 = y1, y0

    if y0 < y1:
        ystep = 1
    else:
        ystep = -1

    deltax = x1 - x0
    deltay = abs(y1 - y0)
    error = -deltax / 2
    y = y0

    line = []
    for x in range(x0, x1 + 1):
        if steep:
            line.append((y, x))
        else:
            line.append((x, y))

        error = error + deltay
        if error > 0:
            y = y + ystep
            error = error - deltax
    if switched:
        line.reverse()
    return line


## Angle constraints:
# * line graph
# * path straighening toolbox

path = np.asarray(path)
for p, (i, j) in enumerate(path[:-2]):
    v1 = path[p + 1] - path[p]
    v2 = path[p + 1] - path[p + 2]
    v1_norm = np.linalg.norm(v1)
    v2_norm = np.linalg.norm(v2)
    angle = np.arccos(np.dot(v1, v2))
    if angle < 

def dotproduct(v1, v2):
    return sum((a * b) for a, b in zip(v1, v2))


def length(v):
    return math.sqrt(dotproduct(v, v))


def angle(v1, v2):
    return math.acos(dotproduct(v1, v2) / (length(v1) * length(v2)))


# Questions:
## altitude leads to more costs for pylons? because they are higher?