"""Coding exercise

You have a list of Cartesian coordinates of locations: List[ (Double, Double) ]. You have a second list of
approximate locations, of the same type. For example:

List 1:
[ (2.5, 2.1),
  (0.9, 1.8),
  (-2.5, 2.1), …]

List 2:
[ (0.0, 0.0),
   (0.2, -0.1),
   (2.4, 2.2),
   (1.0, 2.0), …]

   The coordinates represent distance in kilometres North and East of London.
 	Both lists are large but can fit in the memory of a single computer.

You want to match each item in the second list to the closest location in the first list. You do not know the error
on the estimated location in the second list. As the lists are large, you need the search to be efficient.

"""

import numpy
import timeit

debug = False
verbose = True

experiment_size = 10
float_precision = 2
dimensions = 2

# Assumption: London is 40x40 KMs
# Known locations list
list1 = list(zip(numpy.round(numpy.random.uniform(-20, 20, experiment_size), float_precision),
                 numpy.round(numpy.random.uniform(-20, 20, experiment_size), float_precision)))
# Approximate locations list
list2 = list(zip(numpy.round(numpy.random.uniform(-20, 20, experiment_size // 2), float_precision),
                 numpy.round(numpy.random.uniform(-20, 20, experiment_size // 2), float_precision)))

"""
A few ways to approach the ask.
First, the bullish way: for each point in list2 calculate the distance from each point in list1 and pick the smallest,
in loops! Just helps get a feel of things.
"""


# We need a function that calculates the Euclidean distance between two 2d coordinates
def dist(a, b):
    x1 = a[0]
    x2 = a[1]
    y1 = b[0]
    y2 = b[1]
    # Good old Pythagoras!
    return ((y2 - x2) ** 2 + (y1 - x1) ** 2) ** .5


def find_nearest_v1(p, known_list):
    # Assume it's the first point as a starting comparison reference
    nearest_point = known_list[0]
    nearest_distance = dist(p, nearest_point)

    # Now loop through the rest of the points
    for q in known_list[1:]:
        d = dist(p, q)
        if d < nearest_distance:
            nearest_distance = d
            nearest_point = q

    return nearest_point


# Here goes the matching -- complexity is quadratic
matched_list = {}


def method1():
    for point in list2:
        matched_list[point] = find_nearest_v1(point, list1)
    pass


print('Method 1')
method1()
if verbose:
    print(matched_list)

"""
Let's move on to something slightly more efficient.
One way could be to stray from lists and traditional functions/loops to array manipulation
"""


# Switch from lists to arrays
def find_nearest_v2(p, known_list):
    knwon_array = numpy.asarray(known_list)
    # Distance calculation from point to each known location
    distance = numpy.sum(((knwon_array - p) ** 2), axis=1)
    # Return the point with smallest distance
    return known_list[distance.argmin()]


# Let's calculate again for all approximate locations
matched_list = {}


def method2():
    for point in list2:
        matched_list[point] = find_nearest_v2(point, list1)
    pass


print('Method 2')
method2()
if verbose:
    print(matched_list)

"""
We could push this further. Instead of a function call for each individual approximate point let's try a method where
we drop the remaining loop and have one that works directly on both lists (arrays)
"""


def find_nearest_v3(approximate_list, known_list):
    # Lists to arrays for quick manipulation
    approximate_array = numpy.asarray(approximate_list)
    known_array = numpy.asarray(known_list)

    # Duplicate the known location points array as many times as there are approximate points to search for
    known_array_dup = numpy.tile(known_array, len(approximate_list))
    # Flatten approximate locations array for broadcasting to work
    approximate_array_flat = approximate_array.reshape(1, approximate_array.size)
    # Calculate the distance: subtracting, squaring and then summing along the tiled matrix two columns at a time
    distance = numpy.add.reduceat((known_array_dup - approximate_array_flat) ** 2,
                                  range(0, approximate_array.size, dimensions),
                                  axis=1)
    # Pick the smallest distance and return a result array as index of matched location
    return distance.argmin(axis=0)


# Let's match!

def method3():
    return find_nearest_v3(list2, list1)


print('Method 3')
matched_list = dict(zip(list2, [list1[key] for key in method3()]))
if verbose:
    print(matched_list)

# Who did best?!
"""
This causes the approximation code to rerun --
If you play with experiment size it will be evident that as n->∞ method3 becomes the most efficient
Tested on my laptop (pushed it):

For experiment size 50000
Method 1 execution time: 0.02309236899964162
Method 2 execution time: 0.02571395000040866
Method 3 execution time: 0.018711035999785963

"""
if debug:
    print("For experiment size {}".format(experiment_size))
    print("Method 1 execution time: {}".format(timeit.timeit('method1', globals=globals())))
    print("Method 2 execution time: {}".format(timeit.timeit('method2', globals=globals())))
    print("Method 3 execution time: {}".format(timeit.timeit('method3', globals=globals())))
