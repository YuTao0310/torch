package com.tao;

import org.junit.Test;
import static org.junit.Assert.*;

import ai.djl.ndarray.*;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;

public class ChenTensorTest {
    @Test
    public void test1() {
        NDManager m = Tensor.manager;
        NDArray x = m.create(new double[] {
                0.237, 0.5625, 0.9193, 0.4668,
                0.2721, 0.8366, 0.4005, 0.5825,
                0.9025, 0.4926, 0.0213, 0.3763,
                0.4924, 0.1453, 0.8526, 0.1448,

                0.7581, 0.7629, 0.7482, 0.9231,
                0.533, 0.1731, 0.6851, 0.9389,
                0.2265, 0.6634, 0.3424, 0.6444,
                0.5905, 0.4717, 0.7549, 0.6502 }, new Shape(2, 4, 4));
        NDArray mask = m.create(new boolean[] { true, true, false, false }, new Shape(4, 1));
        NDArray updates = m.create(new double[] {
                0.3826, 0.6515,
                0.3317, 0.8853,
                0.6396, 0.6491,
                0.0572, 0.2227,

                0.0635, 0.1951,
                0.3229, 0.2428,
                0.8302, 0.6933,
                0.3371, 0.6881 }, new Shape(2, 4, 2));
        NDArray y_actual = m.zeros(new Shape(2, 4, 4), DataType.FLOAT64);
        NDArray y_expect = m.create(new double[] {
                0.3826, 0.6515, 0.3317, 0.8853,
                0.6396, 0.6491, 0.0572, 0.2227,
                0.9025, 0.4926, 0.0213, 0.3763,
                0.4924, 0.1453, 0.8526, 0.1448,

                0.0635, 0.1951, 0.3229, 0.2428,
                0.8302, 0.6933, 0.3371, 0.6881,
                0.2265, 0.6634, 0.3424, 0.6444,
                0.5905, 0.4717, 0.7549, 0.6502 }, new Shape(2, 4, 4));
        Tensor.maskedScatter(x, mask, updates, y_actual);
        assertEquals(y_expect, y_actual);
    }

    @Test
    public void test2() {
        NDManager m = Tensor.manager;
        NDArray x = m.create(new double[] {
                0.8178, 0.4918, 0.0954, 0.5229,
                0.5902, 0.1383, 0.0709, 0.3609,
                0.6846, 0.4097, 0.8979, 0.6637,
                0.2668, 0.4597, 0.1929, 0.2226 }, new Shape(4, 4));
        NDArray mask = m.create(new boolean[] { true, true, false, false }, new Shape(4, 1));
        NDArray updates = m.create(new double[] {
                0.0171, 0.99, 0.3842, 0.6995,
                0.5386, 0.1438, 0.1282, 0.9491,
                0.5934, 0.0572, 0.293, 0.5344,
                0.5652, 0.2698, 0.5364, 0.2687 }, new Shape(4, 4));
        NDArray y_actual = m.zeros(new Shape(4, 4), DataType.FLOAT64);
        NDArray y_expect = m.create(new double[] {
                0.0171, 0.99, 0.3842, 0.6995,
                0.5386, 0.1438, 0.1282, 0.9491,
                0.6846, 0.4097, 0.8979, 0.6637,
                0.2668, 0.4597, 0.1929, 0.2226 }, new Shape(4, 4));
        Tensor.maskedScatter(x, mask, updates, y_actual);
        assertEquals(y_expect, y_actual);
    }

    @Test
    public void test3() {
        NDManager m = Tensor.manager;
        NDArray x = m.create(new double[] {
                0.929602, 0.072217, 0.249901, 0.527799,
                0.610949, 0.269106, 0.088661, 0.141707,
                0.147212, 0.659903, 0.610025, 0.313186,
                0.662238, 0.646008, 0.827745, 0.661658 }, new Shape(4, 4));
        NDArray mask = m.create(new boolean[] {
                false, true, false, true,
                false, true, true, true,
                false, true, true, false,
                false, false, false, true }, new Shape(4, 4));
        NDArray updates = m.create(new double[] {
                0.345455, 0.938078, 0.621754, 0.714848,
                0.422705, 0.968609, 0.749588, 0.026307,
                0.283278, 0.656372, 0.82967, 0.336543,
                0.438996, 0.107025, 0.4234, 0.455608 }, new Shape(4, 4));
        NDArray y_actual = m.zeros(new Shape(4, 4), DataType.FLOAT64);
        NDArray y_expect = m.create(new double[] {
                0.929602, 0.345455, 0.249901, 0.938078,
                0.610949, 0.621754, 0.714848, 0.422705,
                0.147212, 0.968609, 0.749588, 0.313186,
                0.662238, 0.646008, 0.827745, 0.026307 }, new Shape(4, 4));
        Tensor.maskedScatter(x, mask, updates, y_actual);
        assertEquals(y_expect, y_actual);
    }

    @Test
    public void test4() {
        NDManager m = Tensor.manager;
        NDArray x = m.create(new double[] {
                0.067487, 0.520008, 0.592678, 0.121186,
                0.704295, 0.254583, 0.537615, 0.941369,
                0.600341, 0.818687, 0.146106, 0.130324,
                0.594028, 0.514504, 0.61037, 0.461913,

                0.506862, 0.439021, 0.270833, 0.08705,
                0.564196, 0.430587, 0.11346, 0.210679,
                0.006547, 0.589734, 0.605281, 0.106689,
                0.708217, 0.840064, 0.790911, 0.005919 }, new Shape(2, 4, 4));
        NDArray mask = m.create(new boolean[] {
                false, true, true, true,
                true, true, false, true,
                false, true, false, true,
                true, true, true, false,

                true, false, false, true,
                true, false, false, false,
                true, true, false, false,
                true, false, true, true }, new Shape(2, 4, 4));
        NDArray updates = m.create(new double[] {
                0.297794, 0.741571, 0.288436, 0.449174,
                0.390588, 0.149118, 0.42289, 0.432216,
                0.181794, 0.353971, 0.325786, 0.671605,
                0.858458, 0.848439, 0.57454, 0.429427,

                0.417952, 0.307038, 0.482919, 0.450703,
                0.325863, 0.214852, 0.58665, 0.233112,
                0.194518, 0.652836, 0.405681, 0.186812,
                0.703447, 0.170602, 0.289051, 0.02187 }, new Shape(2, 4, 4));
        NDArray y_actual = m.zeros(new Shape(2, 4, 4), DataType.FLOAT64);
        NDArray y_expect = m.create(new double[] {
                0.067487, 0.297794, 0.741571, 0.288436,
                0.449174, 0.390588, 0.537615, 0.149118,
                0.600341, 0.42289, 0.146106, 0.432216,
                0.181794, 0.353971, 0.325786, 0.461913,

                0.671605, 0.439021, 0.270833, 0.858458,
                0.848439, 0.430587, 0.11346, 0.210679,
                0.57454, 0.429427, 0.605281, 0.106689,
                0.417952, 0.840064, 0.307038, 0.482919 }, new Shape(2, 4, 4));
        Tensor.maskedScatter(x, mask, updates, y_actual);
        assertEquals(y_expect, y_actual);
    }

    @Test
    public void test5() {
        @SuppressWarnings("unused")
        NDArray x = null;
        @SuppressWarnings("unused")
        NDArray mask = null;
        @SuppressWarnings("unused")
        NDArray updates = null;
        NDArray y_actual = null;
        NDArray y_expect = null;
        assertEquals(y_expect, y_actual);
    }

    @Test
    public void test6() {
        NDManager m = Tensor.manager;
        NDArray x = m.create(new double[]{}, new Shape(0));
        NDArray mask = m.create(new boolean[] { false, false }, new Shape(2));
        NDArray updates = m.create(new double[] {
                0.82205, 0.503393,
                0.678416, 0.607124,
                0.444116, 0.112699,
                0.301088, 0.15907,

                0.683432, 0.317917,
                0.971433, 0.101182,
                0.740625, 0.244003,
                0.601303, 0.260228
        }, new Shape(2, 4, 2));
        NDArray y = m.zeros(new Shape(2, 4, 2), DataType.FLOAT64);
        boolean y_actual = Tensor.maskedScatter(x, mask, updates, y);
        boolean y_expect = false;
        assertEquals(y_expect, y_actual);
    }

    @Test
    public void test7() {
        NDManager m = Tensor.manager;
        NDArray x = m.create(new double[] {
                0.595454, 0.72622, 0.963517, 0.109894,
                0.243732, 0.628344, 0.169528, 0.427846,
                0.241419, 0.093672, 0.664569, 0.226081,
                0.558668, 0.632903, 0.124359, 0.981156 }, new Shape(4, 4));
        NDArray mask = m.create(new boolean[] { false, true, true, false }, new Shape(4, 1));
        NDArray updates = m.create(new double[] {
                0.120078, 0.029578, 0.17691, 0.567436,
                0.404321, 0.343146, 0.966219, 0.584363,
                0.529343, 0.70835, 0.193494, 0.633365,
                0.181736, 0.261316, 0.722338, 0.914812 }, new Shape(4, 4));
        NDArray y_actual = m.zeros(new Shape(4, 4), DataType.FLOAT64);
        NDArray y_expect = m.create(new double[] {
                0.595454, 0.72622, 0.963517, 0.109894,
                0.120078, 0.029578, 0.17691, 0.567436,
                0.404321, 0.343146, 0.966219, 0.584363,
                0.558668, 0.632903, 0.124359, 0.981156 }, new Shape(4, 4));
        Tensor.maskedScatter(x, mask, updates, y_actual);
        assertEquals(y_expect, y_actual);
    }

    @Test
    public void test8() {
        NDManager m = Tensor.manager;
        NDArray x = m.create(new double[] {
                0.950084, 0.284479, 0.541106, 0.409327,
                0.140376, 0.74404, 0.882456, 0.469504,
                0.362858, 0.50707, 0.382469, 0.476461,
                0.898087, 0.138938, 0.006282, 0.048815,

                0.088017, 0.186602, 0.792576, 0.023824,
                0.85368, 0.954485, 0.422903, 0.153391,
                0.204843, 0.824639, 0.862368, 0.158169,
                0.786181, 0.558003, 0.48729, 0.378208 }, new Shape(2, 4, 4));
        NDArray mask = m.create(new boolean[] { false, true, false, true }, new Shape(4, 1));
        NDArray updates = m.create(new double[] {
                0.566516, 0.776724, 0.570652, 0.481755,
                0.950134, 0.3962, 0.719146, 0.749683,
                0.42591, 0.731526, 0.360486, 0.749502,
                0.059771, 0.417891, 0.87278, 0.468465,

                0.511148, 0.449972, 0.050667, 0.262224,
                0.060786, 0.608981, 0.27633, 0.879873,
                0.716769, 0.545286, 0.004457, 0.599634,
                0.225264, 0.133855, 0.083683, 0.530688 }, new Shape(2, 4, 4));
        NDArray y_actual = m.zeros(new Shape(2, 4, 4), DataType.FLOAT64);
        NDArray y_expect = m.create(new double[] {
                0.950084, 0.284479, 0.541106, 0.409327,
                0.566516, 0.776724, 0.570652, 0.481755,
                0.362858, 0.50707, 0.382469, 0.476461,
                0.950134, 0.3962, 0.719146, 0.749683,

                0.088017, 0.186602, 0.792576, 0.023824,
                0.42591, 0.731526, 0.360486, 0.749502,
                0.204843, 0.824639, 0.862368, 0.158169,
                0.059771, 0.417891, 0.87278, 0.468465 }, new Shape(2, 4, 4));
        Tensor.maskedScatter(x, mask, updates, y_actual);
        assertEquals(y_expect, y_actual);
    }

    @Test
    public void test9() {
        NDManager m = Tensor.manager;
        NDArray x = m.create(new double[] {
                0.081664, 0.359249, 0.086218, 0.737273,
                0.257317, 0.608719, 0.287797, 0.058301 }, new Shape(4, 2));
        NDArray mask = m.create(new double[] {}, new Shape(0));
        NDArray updates = m.create(new double[] {
                0.82205, 0.503393,
                0.678416, 0.607124,
                0.444116, 0.112699,
                0.301088, 0.15907,

                0.683432, 0.317917,
                0.971433, 0.101182,
                0.740625, 0.244003,
                0.601303, 0.260228 }, new Shape(2, 4, 2));
        NDArray y = m.zeros(new Shape(4, 2), DataType.FLOAT64);
        boolean y_actual = Tensor.maskedScatter(x, mask, updates, y);
        boolean y_expect = false;
        assertEquals(y_expect, y_actual);
    }

    @Test
    public void test10() {
        NDManager m = Tensor.manager;
        NDArray x = m.create(new double[] {
                0.2495, 0.9488, 0.2217, 0.6993,
                0.2374, 0.743, 0.9903, 0.1816,
                0.9535, 0.9805, 0.7517, 0.246,
                0.5079, 0.219, 0.9746, 0.1524 }, new Shape(4, 4));
        NDArray mask = m.create(new boolean[] {
                true, false,
                false, true,
                true, true,
                true, true }, new Shape(4, 2));
        NDArray updates = m.create(new double[] {
                0.1081, 0.3112, 0.3262, 0.4502,
                0.832, 0.7325, 0.4333, 0.5152,
                0.5923, 0.5506, 0.5903, 0.2408,
                0.0586, 0.8283, 0.1369, 0.0971 }, new Shape(4, 4));
        NDArray y = m.zeros(new Shape(4, 2), DataType.FLOAT64);
        boolean y_actual = Tensor.maskedScatter(x, mask, updates, y);
        boolean y_expect = false;
        assertEquals(y_expect, y_actual);
    }

    @Test
    public void test11() {
        NDManager m = Tensor.manager;
        NDArray x = m.create(new double[] {
                0.859363, 0.318005, 0.621846, 0.349388,
                0.903266, 0.617065, 0.419287, 0.482189,
                0.158396, 0.817671, 0.988992, 0.037841,
                0.65801, 0.746904, 0.819382, 0.845713 }, new Shape(4, 4));
        NDArray mask = m.create(new boolean[] {
                true, false, false, true,
                false, true, false, true,
                false, false, true, false,
                false, false, false, true }, new Shape(4, 4));
        NDArray updates = m.create(new double[] {
                0.66697, 0.973092, 0.232692, 0.502006, 0.222684, 0.459294,
                0.995914, 0.848406, 0.859697, 0.710582, 0.734682, 0.888684,
                0.4365, 0.105199, 0.652645, 0.215087, 0.035374, 0.794563,
                0.18346, 0.999772, 0.09852, 0.652984, 0.429157, 0.464318 }, new Shape(4, 6));
        NDArray y_actual = m.zeros(new Shape(4, 4), DataType.FLOAT64);
        NDArray y_expect = m.create(new double[] {
                0.66697, 0.318005, 0.621846, 0.973092,
                0.903266, 0.232692, 0.419287, 0.502006,
                0.158396, 0.817671, 0.222684, 0.037841,
                0.65801, 0.746904, 0.819382, 0.459294 }, new Shape(4, 4));
        Tensor.maskedScatter(x, mask, updates, y_actual);
        assertEquals(y_expect, y_actual);
    }

    @Test
    public void test12() {
        NDManager m = Tensor.manager;
        NDArray x = m.create(new double[] {
                0.579653, 0.96366, 0.588367, 0.660249,
                0.663035, 0.425077, 0.31224, 0.376898,
                0.273773, 0.655856, 0.40289, 0.617129,
                0.555981, 0.138962, 0.857442, 0.425162,

                0.964898, 0.204174, 0.374252, 0.330872,
                0.024397, 0.653358, 0.266778, 0.467423,
                0.865655, 0.296849, 0.880151, 0.626156,
                0.457795, 0.884763, 0.49729, 0.869059 }, new Shape(2, 4, 4));
        NDArray mask = m.create(new boolean[] {
                true, true, true, true,
                false, true, false, true,
                false, true, false, false,
                false, true, false, true,

                true, false, true, true,
                true, true, true, false,
                false, true, true, false,
                true, false, false, true }, new Shape(2, 4, 4));
        NDArray updates = m.create(new double[] {
                0.569555, 0.173924, 0.889675, 0.068642, 0.596887, 0.146293,
                0.750077, 0.611804, 0.112642, 0.915179, 0.790348, 0.310971,
                0.478967, 0.686268, 0.016044, 0.357691, 0.842107, 0.30703,
                0.972965, 0.091465, 0.244955, 0.100442, 0.4101, 0.013227 }, new Shape(4, 6));
        NDArray y_actual = m.zeros(new Shape(2, 4, 4), DataType.FLOAT64);
        NDArray y_expect = m.create(new double[] {
                0.569555, 0.173924, 0.889675, 0.068642,
                0.663035, 0.596887, 0.31224, 0.146293,
                0.273773, 0.750077, 0.40289, 0.617129,
                0.555981, 0.611804, 0.857442, 0.112642,

                0.915179, 0.204174, 0.790348, 0.310971,
                0.478967, 0.686268, 0.016044, 0.467423,
                0.865655, 0.357691, 0.842107, 0.626156,
                0.30703, 0.884763, 0.49729, 0.972965 }, new Shape(2, 4, 4));
        Tensor.maskedScatter(x, mask, updates, y_actual);
        assertEquals(y_expect, y_actual);
    }

    @Test
    public void test13() {
        NDManager m = Tensor.manager;
        NDArray x = m.create(new double[] {
                0.220563, 0.220562, 0.874502, 0.65702,
                0.744828, 0.634401, 0.643221, 0.114343,
                0.851807, 0.29013, 0.064638, 0.756552,
                0.505506, 0.081819, 0.907541, 0.225031,

                0.65549, 0.668774, 0.120161, 0.267262,
                0.08368, 0.535413, 0.094103, 0.656065,
                0.998803, 0.600117, 0.990946, 0.931947,
                0.615415, 0.660363, 0.968039, 0.557831 }, new Shape(2, 4, 4));
        NDArray mask = m.create(new boolean[] {
                true, false, true, true,
                true, true, true, true,
                true, true, true, true,
                false, true, false, true,

                false, false, true, true,
                false, true, false, true,
                false, true, true, true,
                false, false, true, false }, new Shape(2, 4, 4));
        NDArray updates = m.create(new double[] {}, new Shape(0));
        NDArray y = m.zeros(new Shape(4, 2), DataType.FLOAT64);
        boolean y_actual = Tensor.maskedScatter(x, mask, updates, y);
        boolean y_expect = false;
        assertEquals(y_expect, y_actual);
    }

    @Test
    public void test14() {
        NDManager m = Tensor.manager;
        NDArray x = m.create(new double[] {
                0.000592, 0.276188, 0.3237, 0.036467,
                0.805756, 0.771859, 0.241923, 0.068973,
                0.514108, 0.935482, 0.128346, 0.887272,
                0.377234, 0.203972, 0.915916, 0.967531,

                0.081138, 0.628782, 0.435806, 0.80246,
                0.122536, 0.957648, 0.710682, 0.333485,
                0.596873, 0.252986, 0.123289, 0.359879,
                0.796889, 0.356141, 0.213105, 0.663922 }, new Shape(2, 4, 4));
        NDArray mask = m.create(new boolean[] {
                false, false, false, false,
                true, true, true, true,
                true, true, true, true,
                true, true, true, true,

                false, false, false, false,
                true, true, true, true,
                true, true, true, true,
                true, true, true, true }, new Shape(2, 4, 4));
        NDArray updates = m.create(new double[] { 0.049149 }, new Shape(1));
        NDArray y = m.zeros(new Shape(2, 4, 4), DataType.FLOAT64);
        boolean y_actual = Tensor.maskedScatter(x, mask, updates, y);
        boolean y_expect = false;
        assertEquals(y_expect, y_actual);
    }

    @Test
    public void test15() {
        NDManager m = Tensor.manager;
        NDArray x = m.create(new double[] {
                0.305353, 0.224532, 0.625014, 0.425538, 0.645451,
                0.966436, 0.254806, 0.340366, 0.17633, 0.925344,
                0.980594, 0.555812, 0.503716, 0.491934, 0.529449,
                0.145938, 0.817549, 0.299602, 0.776627, 0.227554,
                0.782175, 0.843763, 0.43636, 0.998611, 0.371221,

                0.976383, 0.264757, 0.731821, 0.731995, 0.304554,
                0.156271, 0.065114, 0.432395, 0.013488, 0.055342,
                0.803082, 0.851918, 0.734394, 0.39124, 0.934414,
                0.71086, 0.784264, 0.019514, 0.192347, 0.613955,
                0.582508, 0.405992, 0.68334, 0.058832, 0.644838,

                0.165988, 0.730874, 0.166256, 0.309037, 0.190109,
                0.069069, 0.800693, 0.075851, 0.840386, 0.690784,
                0.592592, 0.281155, 0.568213, 0.706446, 0.619048,
                0.919462, 0.106465, 0.13771, 0.111847, 0.320837,
                0.677675, 0.640901, 0.688543, 0.319433, 0.179025,

                0.752486, 0.499727, 0.859264, 0.022924, 0.816812,
                0.456758, 0.380398, 0.109697, 0.386512, 0.53043,
                0.060409, 0.208129, 0.141951, 0.50794, 0.679799,
                0.650115, 0.908909, 0.921203, 0.286748, 0.621787,
                0.460813, 0.722314, 0.81259, 0.385458, 0.276701,

                0.348429, 0.774396, 0.349907, 0.976895, 0.088227,
                0.608515, 0.384606, 0.96788, 0.920019, 0.803995,
                0.647775, 0.379848, 0.759616, 0.962004, 0.744329,
                0.495934, 0.29906, 0.047789, 0.048551, 0.389006,
                0.870474, 0.369421, 0.774538, 0.400295, 0.236194 }, new Shape(5, 5, 5));
        NDArray mask = m.create(new boolean[] {
                true, false, true, true, false,
                true, true, false, true, false,
                true, false, false, true, true,
                false, true, true, true, true,
                true, true, false, false, true,

                false, false, true, false, false,
                false, false, false, false, true,
                true, false, true, true, false,
                true, false, false, false, true,
                false, false, false, false, false,

                true, true, true, false, false,
                false, false, true, false, true,
                true, false, true, true, false,
                false, true, true, false, false,
                true, true, false, false, false,

                true, false, false, false, true,
                true, true, true, true, true,
                false, true, false, false, false,
                false, false, false, false, false,
                true, true, false, true, true,

                false, true, true, true, false,
                false, true, true, true, true,
                false, false, false, true, false,
                false, false, false, true, true,
                false, false, true, true, true }, new Shape(5, 5, 5));
        NDArray updates = m.create(new double[] {
                0.910447, 0.069544, 0.447657, 0.441921, 0.595295, 0.631804, 0.511761, 0.278704, 0.160596,
                0.596251, 0.17016, 0.178881, 0.039432, 0.346015, 0.005686, 0.697137, 0.014381, 0.841652,
                0.292824, 0.012456, 0.792399, 0.934044, 0.018374, 0.396851, 0.710005, 0.747471, 0.283306,
                0.61622, 0.632477, 0.955724, 0.340466, 0.741465, 0.957577, 0.171479, 0.146362, 0.21251,
                0.16314, 0.033396, 0.849875, 0.805518, 0.713358, 0.332295, 0.515625, 0.639229, 0.789744,

                0.270197, 0.498939, 0.761286, 0.835185, 0.167911, 0.028714, 0.855166, 0.111335, 0.540889,
                0.825574, 0.37372, 0.524661, 0.224345, 0.023464, 0.773385, 0.935651, 0.050413, 0.914178,
                0.860973, 0.860467, 0.25383, 0.795551, 0.959421, 0.137983, 0.340443, 0.224125, 0.148896,
                0.818196, 0.49857, 0.744712, 0.322585, 0.566556, 0.913262, 0.761802, 0.989209, 0.430145,
                0.009905, 0.067435, 0.78822, 0.92418, 0.741782, 0.516445, 0.601218, 0.298493, 0.762601,

                0.579777, 0.784137, 0.718753, 0.234138, 0.414149, 0.614561, 0.388877, 0.080962, 0.142621,
                0.941496, 0.91619, 0.68376, 0.721325, 0.728231, 0.035125, 0.541888, 0.715555, 0.384256,
                0.658378, 0.427182, 0.796401, 0.255833, 0.223985, 0.866767, 0.308655, 0.849221, 0.013954,
                0.747467, 0.300295, 0.467088, 0.747044, 0.5525, 0.849201, 0.358539, 0.737197, 0.164941,
                0.608574, 0.452258, 0.906071, 0.077521, 0.604657, 0.419083, 0.50125, 0.632859, 0.080324,

                0.560946, 0.873192, 0.958345, 0.789748, 0.904019, 0.704594, 0.622331, 0.691103, 0.217111,
                0.023833, 0.354904, 0.339499, 0.053788, 0.113468, 0.306667, 0.134097, 0.407959, 0.836159,
                0.775878, 0.64557, 0.842261, 0.435812, 0.306166, 0.157695, 0.362695, 0.404028, 0.708571,
                0.545841, 0.154218, 0.235718, 0.679978, 0.024192, 0.1212, 0.919973, 0.550621, 0.434135,
                0.124306, 0.076962, 0.03349, 0.90841, 0.564723, 0.735693, 0.698075, 0.014295, 0.435206,

                0.974708, 0.855862, 0.749027, 0.238152, 0.88178, 0.103517, 0.365809, 0.366792, 0.292712,
                0.261022, 0.556999, 0.809029, 0.85448, 0.769568, 0.440014, 0.91812, 0.93698, 0.95991,
                0.269176, 0.956004, 0.733661, 0.902605, 0.698041, 0.516465, 0.521006, 0.527662, 0.033861,
                0.393495, 0.567389, 0.107058, 0.516356, 0.888049, 0.486064, 0.236288, 0.932129, 0.597397,
                0.849001, 0.477183, 0.897428, 0.42956, 0.567854, 0.158922, 0.700447, 0.813353, 0.832457 },
                new Shape(5, 5, 9));
        NDArray y_actual = m.zeros(new Shape(5, 5, 5), DataType.FLOAT64);
        NDArray y_expect = m.create(new double[] {
                0.910447, 0.224532, 0.069544, 0.447657, 0.645451,
                0.441921, 0.595295, 0.340366, 0.631804, 0.925344,
                0.511761, 0.555812, 0.503716, 0.278704, 0.160596,
                0.145938, 0.596251, 0.17016, 0.178881, 0.039432,
                0.346015, 0.005686, 0.43636, 0.998611, 0.697137,

                0.976383, 0.264757, 0.014381, 0.731995, 0.304554,
                0.156271, 0.065114, 0.432395, 0.013488, 0.841652,
                0.292824, 0.851918, 0.012456, 0.792399, 0.934414,
                0.934044, 0.784264, 0.019514, 0.192347, 0.018374,
                0.582508, 0.405992, 0.68334, 0.058832, 0.644838,

                0.396851, 0.710005, 0.747471, 0.309037, 0.190109,
                0.069069, 0.800693, 0.283306, 0.840386, 0.61622,
                0.632477, 0.281155, 0.955724, 0.340466, 0.619048,
                0.919462, 0.741465, 0.957577, 0.111847, 0.320837,
                0.171479, 0.146362, 0.688543, 0.319433, 0.179025,

                0.21251, 0.499727, 0.859264, 0.022924, 0.16314,
                0.033396, 0.849875, 0.805518, 0.713358, 0.332295,
                0.060409, 0.515625, 0.141951, 0.50794, 0.679799,
                0.650115, 0.908909, 0.921203, 0.286748, 0.621787,
                0.639229, 0.789744, 0.81259, 0.270197, 0.498939,

                0.348429, 0.761286, 0.835185, 0.167911, 0.088227,
                0.608515, 0.028714, 0.855166, 0.111335, 0.540889,
                0.647775, 0.379848, 0.759616, 0.825574, 0.744329,
                0.495934, 0.29906, 0.047789, 0.37372, 0.524661,
                0.870474, 0.369421, 0.224345, 0.023464, 0.773385 }, new Shape(5, 5, 5));
        Tensor.maskedScatter(x, mask, updates, y_actual);
        assertEquals(y_expect, y_actual);
    }
}
