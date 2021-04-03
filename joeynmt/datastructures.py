## Taken from https://github.com/rycolab/bfbs/blob/0f2a4de58407499692000a44e55a59d56e3a3eae/datastructures/min_max_queue.py

class MinMaxHeap(object):
    """
    Implementation of a Min-max heap following Atkinson, Sack, Santoro, and
    Strothotte (1986): https://doi.org/10.1145/6617.6621
    """
    def __init__(self, reserve=0):
        self.a = [None] * reserve
        self.size = 0

    def __len__(self):
        return self.size

    def __iter__(self):
        return self

    def __list__(self):
        return self.a

    def __next__(self):
        try:
            return self.popmin()
        except AssertionError:
            raise StopIteration

    def insert(self, key):
        """
        Insert key into heap. Complexity: O(log(n))
        """
        if len(self.a) < self.size + 1:
            self.a.append(key)
        insert(self.a, key, self.size)
        self.size += 1

    def peekmin(self):
        """
        Get minimum element. Complexity: O(1)
        """
        return peekmin(self.a, self.size)

    def peekmax(self):
        """
        Get maximum element. Complexity: O(1)
        """
        return peekmax(self.a, self.size)

    def popmin(self):
        """
        Remove and return minimum element. Complexity: O(log(n))
        """
        m, self.size = removemin(self.a, self.size)
        self.a.pop(-1)
        return m

    def popmax(self):
        """
        Remove and return maximum element. Complexity: O(log(n))
        """
        m, self.size = removemax(self.a, self.size)
        self.a.pop(-1)
        return m

    def replacemax(self, val):
        """
        Remove and return maximum element. Complexity: O(log(n))
        """
        replacemax(self.a, self.size, val)

    def replacemin(self, val):
        """
        Remove and return maximum element. Complexity: O(log(n))
        """
        replacemin(self.a, self.size, val)



def level(i):
    return (i+1).bit_length() - 1


def trickledown(array, i, size):
    if level(i) % 2 == 0:  # min level
        trickledownmin(array, i, size)
    else:
        trickledownmax(array, i, size)


def trickledownmin(array, i, size):
    if size > i * 2 + 1:  # i has children
        m = i * 2 + 1
        if i * 2 + 2 < size and array[i*2+2] < array[m]:
            m = i*2+2
        child = True
        for j in range(i*4+3, min(i*4+7, size)):
            if array[j] < array[m]:
                m = j
                child = False

        if child:
            if array[m] < array[i]:
                array[i], array[m] = array[m], array[i]
        else:
            if array[m] < array[i]:
                if array[m] < array[i]:
                    array[m], array[i] = array[i], array[m]
                if array[m] > array[(m-1) // 2]:
                    array[m], array[(m-1)//2] = array[(m-1)//2], array[m]
                trickledownmin(array, m, size)


def trickledownmax(array, i, size):
    if size > i * 2 + 1:  # i has children
        m = i * 2 + 1
        if i * 2 + 2 < size and array[i*2+2] > array[m]:
            m = i*2+2
        child = True
        for j in range(i*4+3, min(i*4+7, size)):
            if array[j] > array[m]:
                m = j
                child = False

        if child:
            if array[m] > array[i]:
                array[i], array[m] = array[m], array[i]
        else:
            if array[m] > array[i]:
                if array[m] > array[i]:
                    array[m], array[i] = array[i], array[m]
                if array[m] < array[(m-1) // 2]:
                    array[m], array[(m-1)//2] = array[(m-1)//2], array[m]
                trickledownmax(array, m, size)


def bubbleup(array, i):
    if level(i) % 2 == 0:  # min level
        if i > 0 and array[i] > array[(i-1) // 2]:
            array[i], array[(i-1) // 2] = array[(i-1)//2], array[i]
            bubbleupmax(array, (i-1)//2)
        else:
            bubbleupmin(array, i)
    else:  # max level
        if i > 0 and array[i] < array[(i-1) // 2]:
            array[i], array[(i-1) // 2] = array[(i-1) // 2], array[i]
            bubbleupmin(array, (i-1)//2)
        else:
            bubbleupmax(array, i)


def bubbleupmin(array, i):
    while i > 2:
        if array[i] < array[(i-3) // 4]:
            array[i], array[(i-3) // 4] = array[(i-3) // 4], array[i]
            i = (i-3) // 4
        else:
            return


def bubbleupmax(array, i):
    while i > 2:
        if array[i] > array[(i-3) // 4]:
            array[i], array[(i-3) // 4] = array[(i-3) // 4], array[i]
            i = (i-3) // 4
        else:
            return


def peekmin(array, size):
    assert size > 0
    return array[0]


def peekmax(array, size):
    assert size > 0
    if size == 1:
        return array[0]
    elif size == 2:
        return array[1]
    else:
        return max(array[1], array[2])


def removemin(array, size):
    assert size > 0
    elem = array[0]
    array[0] = array[size-1]
    # array = array[:-1]
    trickledown(array, 0, size - 1)
    return elem, size-1


def removemax(array, size):
    assert size > 0
    if size == 1:
        return array[0], size - 1
    elif size == 2:
        return array[1], size - 1
    else:
        i = 1 if array[1] > array[2] else 2
        elem = array[i]
        array[i] = array[size-1]
        # array = array[:-1]
        trickledown(array, i, size - 1)
        return elem, size-1

def replacemax(array, size, val):
    assert size > 0
    if size == 1:
        array[0] = val
    elif size == 2:
        array[1] = val
        bubbleup(array, 1)
    else:
        i = 1 if array[1] > array[2] else 2
        array[i] = array[size-1]
        trickledown(array, i, size)
        array[size-1] = val
        bubbleup(array, size-1)

def replacemin(array, size, val):
    assert size > 0
    array[0] = val
    trickledown(array, 0, size)
    assert minmaxheapproperty(array, len(array))

def insert(array, k, size):
    array[size] = k
    bubbleup(array, size)


def minmaxheapproperty(array, size):
    for i, k in enumerate(array[:size]):
        if level(i) % 2 == 0:  # min level
            # check children to be larger
            for j in range(2 * i + 1, min(2 * i + 3, size)):
                if array[j] < k:
                    print(array, j, i, array[j], array[i], level(i))
                    return False
            # check grand children to be larger
            for j in range(4 * i + 3, min(4 * i + 7, size)):
                if array[j] < k:
                    print(array, j, i, array[j], array[i], level(i))
                    return False
        else:
            # check children to be smaller
            for j in range(2 * i + 1, min(2 * i + 3, size)):
                if array[j] > k:
                    print(array, j, i, array[j], array[i], level(i))
                    return False
            # check grand children to be smaller
            for j in range(4 * i + 3, min(4 * i + 7, size)):
                if array[j] > k:
                    print(array, j, i, array[j], array[i], level(i))
                    return False

    return True
