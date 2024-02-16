import numpy as np
from ..utils.fitting import exponential, linear


def test_exponential():
    # Test case 1: x = 0
    x = 0
    a = 2
    b = 0.5
    c = 1
    expected_result = 3
    assert np.isclose(exponential(x, a, b, c), expected_result)

    # Test case 2: x = 1
    x = 1
    a = 2
    b = 0.5
    c = 1
    expected_result = 2.213061319425267
    assert np.isclose(exponential(x, a, b, c), expected_result)

    # Test case 3: x = -1
    x = -1
    a = 2
    b = 0.5
    c = 1
    expected_result = 4.297442541400256
    assert np.isclose(exponential(x, a, b, c), expected_result)

    # Test case 4: x = 2
    x = 2
    a = 2
    b = 0.5
    c = 1
    expected_result = 1.7357588823428847
    assert np.isclose(exponential(x, a, b, c), expected_result)

    # Test case 5: x = -2
    x = -2
    a = 2
    b = 0.5
    c = 1
    expected_result = 6.43656365691809
    assert np.isclose(exponential(x, a, b, c), expected_result)

    print("All test cases pass")

def test_linear():
    # Test case 1: x = [1, 2, 3], m = 2, b = 1
    x = np.array([1, 2, 3])
    m = 2
    b = 1
    expected_result = np.array([3, 5, 7])
    assert np.array_equal(linear(x, m, b), expected_result)

    # Test case 2: x = [0, 0, 0], m = -1, b = 0
    x = np.array([0, 0, 0])
    m = -1
    b = 0
    expected_result = np.array([0, 0, 0])
    assert np.array_equal(linear(x, m, b), expected_result)

    # Test case 3: x = [-1, 0, 1], m = 0.5, b = -1
    x = np.array([-1, 0, 1])
    m = 0.5
    b = -1
    expected_result = np.array([-1.5, -1, -0.5])
    assert np.allclose(linear(x, m, b), expected_result)

    print("All test cases pass")

test_exponential()
test_linear()
