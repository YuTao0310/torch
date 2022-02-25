package com.tao;

import org.junit.Test;
import static org.junit.Assert.*;

import ai.djl.ndarray.*;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;

public class TaoTensorTest {
    @Test
    public void test1() {
        NDManager m = Tensor.manager;
        NDArray x = m.create(new double[] {
                0.6554, 0.6587, 0.8480,
                0.5713, 0.5169, 0.4044,
                0.1661, 0.0549, 0.2353
        }, new Shape(3, 3));
        NDArray mask = m.create(new boolean[] {
                false, true, true,
                true, true, false,
                true, true, true
        }, new Shape(3, 3));
        NDArray updates = m.create(new double[] {
                0.1629, 0.2550, 0.9311,
                0.7194, 0.8809, 0.0068,
                0.4874, 0.0873, 0.8528
        }, new Shape(3, 3));
        NDArray y_actual = m.zeros(new Shape(3, 3), DataType.FLOAT64);
        Tensor.maskedScatter(x, mask, updates, y_actual);
        NDArray y_expect = m.create(new double[] {
                0.6554, 0.1629, 0.2550,
                0.9311, 0.7194, 0.4044,
                0.8809, 0.0068, 0.4874
        }, new Shape(3, 3));
        assertEquals(y_expect, y_actual);
    }

    @Test
    public void test2() {
        NDManager m = Tensor.manager;
        NDArray x = m.create(new double[] {
                0.5734, 0.8452, 0.7872,
                0.3862, 0.0967, 0.2989,
                0.1228, 0.3566, 0.1585
        }, new Shape(3, 3));
        NDArray mask = m.create(new boolean[] {
                false,
                true,
                false
        }, new Shape(3, 1));
        NDArray updates = m.create(new double[] {
                0.0070, 0.8361, 0.1293,
                0.1116, 0.7282, 0.7419,
                0.2017, 0.4891, 0.6512
        }, new Shape(3, 3));
        NDArray y_actual = m.zeros(new Shape(3, 3), DataType.FLOAT64);
        Tensor.maskedScatter(x, mask, updates, y_actual);
        NDArray y_expect = m.create(new double[] {
                0.5734, 0.8452, 0.7872,
                0.0070, 0.8361, 0.1293,
                0.1228, 0.3566, 0.1585
        }, new Shape(3, 3));
        assertEquals(y_expect, y_actual);
    }

    @Test
    public void test3() {
        NDManager m = Tensor.manager;
        NDArray x = m.create(new double[] {
                0.3301, 0.7211, 0.9263,
                0.4054, 0.3507, 0.2935,
                0.4912, 0.9217, 0.9448
        }, new Shape(3, 3));
        NDArray mask = m.create(new boolean[] {
                false, false, true
        }, new Shape(1, 3));
        NDArray updates = m.create(new double[] {
                0.5683, 0.5777, 0.7494,
                0.2767, 0.0856, 0.1244,
                0.4429, 0.3795, 0.3588
        }, new Shape(3, 3));
        NDArray y_actual = m.zeros(new Shape(3, 3), DataType.FLOAT64);
        Tensor.maskedScatter(x, mask, updates, y_actual);
        NDArray y_expect = m.create(new double[] {
                0.3301, 0.7211, 0.5683,
                0.4054, 0.3507, 0.5777,
                0.4912, 0.9217, 0.7494
        }, new Shape(3, 3));
        assertEquals(y_expect, y_actual);
    }

    @Test
    public void test4() {
        NDManager m = Tensor.manager;
        NDArray x = m.create(new double[] {
                0.4205, 0.6668, 0.2857,
                0.6784, 0.8717, 0.7571,
                0.3496, 0.4896, 0.8439
        }, new Shape(3, 3));
        NDArray mask = m.create(new boolean[] {
                false, false, true
        }, new Shape(3));
        NDArray updates = m.create(new double[] {
                0.3649, 0.6505, 0.6858,
                0.6327, 0.0650, 0.7962,
                0.9549, 0.2854, 0.8780
        }, new Shape(3, 3));
        NDArray y_actual = m.zeros(new Shape(3, 3), DataType.FLOAT64);
        Tensor.maskedScatter(x, mask, updates, y_actual);
        NDArray y_expect = m.create(new double[] {
                0.4205, 0.6668, 0.3649,
                0.6784, 0.8717, 0.6505,
                0.3496, 0.4896, 0.6858
        }, new Shape(3, 3));
        assertEquals(y_expect, y_actual);
    }

    @Test
    public void test5() {
        NDManager m = Tensor.manager;
        NDArray x = m.create(new double[] {
                0.5460, 0.5653, 0.4468,
                0.9971, 0.1945, 0.0260,
                0.9805, 0.0514, 0.1096
        }, new Shape(3, 3));
        NDArray mask = m.create(new boolean[] {

        }, new Shape(0));
        NDArray updates = m.create(new double[] {
                0.6092, 0.4120, 0.0199,
                0.4986, 0.4976, 0.0708,
                0.2935, 0.3466, 0.6362
        }, new Shape(3, 3));
        NDArray y = m.zeros(new Shape(3, 3), DataType.FLOAT64);
        boolean y_actual = Tensor.maskedScatter(x, mask, updates, y);
        boolean y_expect = false;
        assertEquals(y_expect, y_actual);
    }

    @Test
    public void test6() {
        NDManager m = Tensor.manager;
        NDArray x = m.create(new double[] {
                0.7415, 0.8753, 0.3310,
                0.2461, 0.1950, 0.6833,
                0.6065, 0.5475, 0.1922,

                0.8538, 0.5844, 0.4373,
                0.5599, 0.9492, 0.5969,
                0.1845, 0.0701, 0.3113,

                0.5240, 0.9503, 0.7695,
                0.8224, 0.7814, 0.5088,
                0.9893, 0.2588, 0.4895
        }, new Shape(3, 3, 3));
        NDArray mask = m.create(new boolean[] {
                true, true, false,
                false, false, false,
                true, false, false,

                true, true, true,
                false, true, false,
                true, true, false,

                true, true, false,
                false, false, false,
                true, true, false
        }, new Shape(3, 3, 3));
        NDArray updates = m.create(new double[] {
                0.7409, 0.3774, 0.1427,
                0.9988, 0.6073, 0.9278,
                0.0279, 0.4399, 0.7193,

                0.5362, 0.2935, 0.3921,
                0.9604, 0.7824, 0.5717,
                0.7466, 0.0836, 0.6711,

                0.9498, 0.9180, 0.7607,
                0.4307, 0.0148, 0.6480,
                0.1380, 0.4443, 0.4633
        }, new Shape(3, 3, 3));
        NDArray y_actual = m.zeros(new Shape(3, 3, 3), DataType.FLOAT64);
        Tensor.maskedScatter(x, mask, updates, y_actual);
        NDArray y_expect = m.create(new double[] {
                0.7409, 0.3774, 0.3310,
                0.2461, 0.1950, 0.6833,
                0.1427, 0.5475, 0.1922,

                0.9988, 0.6073, 0.9278,
                0.5599, 0.0279, 0.5969,
                0.4399, 0.7193, 0.3113,

                0.5362, 0.2935, 0.7695,
                0.8224, 0.7814, 0.5088,
                0.3921, 0.9604, 0.4895
        }, new Shape(3, 3, 3));
        assertEquals(y_expect, y_actual);
    }

    @Test
    public void test7() {
        NDManager m = Tensor.manager;
        NDArray x = m.create(new double[] {
                0.1646, 0.4281, 0.5371,
                0.7835, 0.2432, 0.6916,
                0.1119, 0.6043, 0.3399,

                0.9151, 0.7624, 0.0412,
                0.3181, 0.6999, 0.9643,
                0.9506, 0.0030, 0.9567,

                0.3956, 0.5942, 0.7692,
                0.8900, 0.7447, 0.9375,
                0.1393, 0.3264, 0.3757
        }, new Shape(3, 3, 3));
        NDArray mask = m.create(new boolean[] {
                true, false, true,

                false, true, false,

                false, true, false
        }, new Shape(3, 1, 3));
        NDArray updates = m.create(new double[] {
                0.0424, 0.4932, 0.5643,
                0.7200, 0.5488, 0.4701,
                0.5709, 0.2749, 0.0680,

                0.6925, 0.2260, 0.8308,
                0.6265, 0.5737, 0.4169,
                0.5991, 0.7376, 0.1182,

                0.0864, 0.3367, 0.9750,
                0.3353, 0.6659, 0.1247,
                0.8620, 0.5575, 0.9765
        }, new Shape(3, 3, 3));
        NDArray y_actual = m.zeros(new Shape(3, 3, 3), DataType.FLOAT64);
        Tensor.maskedScatter(x, mask, updates, y_actual);
        NDArray y_expect = m.create(new double[] {
                0.0424, 0.4281, 0.4932,
                0.5643, 0.2432, 0.7200,
                0.5488, 0.6043, 0.4701,

                0.9151, 0.5709, 0.0412,
                0.3181, 0.2749, 0.9643,
                0.9506, 0.0680, 0.9567,

                0.3956, 0.6925, 0.7692,
                0.8900, 0.2260, 0.9375,
                0.1393, 0.8308, 0.3757
        }, new Shape(3, 3, 3));
        assertEquals(y_expect, y_actual);
    }

    @Test
    public void test8() {
        NDManager m = Tensor.manager;
        NDArray x = m.create(new double[] {
                0.4178, 0.9017, 0.9166,
                0.3146, 0.7323, 0.4809,
                0.8234, 0.2527, 0.8187,

                0.8286, 0.1143, 0.1100,
                0.7374, 0.9138, 0.1022,
                0.1577, 0.5966, 0.6657,

                0.8588, 0.2002, 0.5108,
                0.9383, 0.3022, 0.0539,
                0.3549, 0.8921, 0.7803
        }, new Shape(3, 3, 3));
        NDArray mask = m.create(new boolean[] {
                false, false, false,
                true, true, false,
                false, false, true
        }, new Shape(3, 3));
        NDArray updates = m.create(new double[] {
                0.8541, 0.4868, 0.5432,
                0.9816, 0.2420, 0.5897,
                0.9595, 0.6241, 0.6871,

                0.7814, 0.1278, 0.3026,
                0.2162, 0.9552, 0.9733,
                0.2550, 0.9595, 0.5428,

                0.4725, 0.8093, 0.3842,
                0.7897, 0.5184, 0.1026,
                0.8586, 0.3286, 0.3165
        }, new Shape(3, 3, 3));
        NDArray y_actual = m.zeros(new Shape(3, 3, 3), DataType.FLOAT64);
        Tensor.maskedScatter(x, mask, updates, y_actual);
        NDArray y_expect = m.create(new double[] {
                0.4178, 0.9017, 0.9166,
                0.8541, 0.4868, 0.4809,
                0.8234, 0.2527, 0.5432,

                0.8286, 0.1143, 0.1100,
                0.9816, 0.2420, 0.1022,
                0.1577, 0.5966, 0.5897,

                0.8588, 0.2002, 0.5108,
                0.9595, 0.6241, 0.0539,
                0.3549, 0.8921, 0.6871
        }, new Shape(3, 3, 3));
        assertEquals(y_expect, y_actual);
    }

    @Test
    public void test9() {
        NDManager m = Tensor.manager;
        NDArray x = m.create(new double[] {
                0.2295, 0.9585, 0.4627,
                0.3375, 0.5052, 0.0782,
                0.3575, 0.2909, 0.4859,

                0.4724, 0.0799, 0.6703,
                0.1621, 0.4460, 0.7793,
                0.3813, 0.1515, 0.8263,

                0.2770, 0.9063, 0.4802,
                0.5241, 0.9289, 0.7796,
                0.2662, 0.5279, 0.5207
        }, new Shape(3, 3, 3));
        NDArray mask = m.create(new boolean[] {
                false, false, true
        }, new Shape(3));
        NDArray updates = m.create(new double[] {
                0.7579, 0.0782, 0.5192,
                0.1984, 0.5896, 0.4016,
                0.6690, 0.7407, 0.7451,

                0.0306, 0.2124, 0.9933,
                0.4163, 0.4983, 0.5664,
                0.6976, 0.4974, 0.6022,

                0.7187, 0.8300, 0.5760,
                0.7877, 0.2218, 0.9872,
                0.0927, 0.1319, 0.4965
        }, new Shape(3, 3, 3));
        NDArray y_actual = m.zeros(new Shape(3, 3, 3), DataType.FLOAT64);
        Tensor.maskedScatter(x, mask, updates, y_actual);
        NDArray y_expect = m.create(new double[] {
                0.2295, 0.9585, 0.7579,
                0.3375, 0.5052, 0.0782,
                0.3575, 0.2909, 0.5192,

                0.4724, 0.0799, 0.1984,
                0.1621, 0.4460, 0.5896,
                0.3813, 0.1515, 0.4016,

                0.2770, 0.9063, 0.6690,
                0.5241, 0.9289, 0.7407,
                0.2662, 0.5279, 0.7451
        }, new Shape(3, 3, 3));
        assertEquals(y_expect, y_actual);
    }

    @Test
    public void test10() {
        NDManager m = Tensor.manager;
        NDArray x = m.create(new double[] {
                0.5496, 0.6762, 0.0506,
                0.7485, 0.3721, 0.3077,
                0.9541, 0.5628, 0.0216,

                0.2516, 0.6748, 0.0653,
                0.3150, 0.1049, 0.6463,
                0.4745, 0.5349, 0.0029,

                0.6597, 0.6762, 0.1756,
                0.5013, 0.9355, 0.4271,
                0.8657, 0.3259, 0.5342
        }, new Shape(3, 3, 3));
        NDArray mask = m.create(new boolean[] {
                true
        }, new Shape(1));
        NDArray updates = m.create(new double[] {
                0.3837, 0.9587, 0.3371,
                0.7372, 0.0215, 0.0649,
                0.3410, 0.1331, 0.3034,

                0.5645, 0.4846, 0.2438,
                0.5410, 0.5481, 0.2211,
                0.1711, 0.8946, 0.4581,

                0.3475, 0.8980, 0.5998,
                0.0843, 0.7534, 0.6800,
                0.8551, 0.9077, 0.0443
        }, new Shape(3, 3, 3));
        NDArray y_actual = m.zeros(new Shape(3, 3, 3), DataType.FLOAT64);
        Tensor.maskedScatter(x, mask, updates, y_actual);
        NDArray y_expect = m.create(new double[] {
                0.3837, 0.9587, 0.3371,
                0.7372, 0.0215, 0.0649,
                0.3410, 0.1331, 0.3034,

                0.5645, 0.4846, 0.2438,
                0.5410, 0.5481, 0.2211,
                0.1711, 0.8946, 0.4581,

                0.3475, 0.8980, 0.5998,
                0.0843, 0.7534, 0.6800,
                0.8551, 0.9077, 0.0443
        }, new Shape(3, 3, 3));
        assertEquals(y_expect, y_actual);
    }

    @Test
    public void test11() {
        NDManager m = Tensor.manager;
        NDArray x = m.create(new double[] {
            0.8377, 0.5999, 0.5875, 0.8752,
            0.5868, 0.9540, 0.4161, 0.6542,
            0.2371, 0.9555, 0.8492, 0.7708
        }, new Shape(3, 4));
        NDArray mask = m.create(new boolean[] {
            false,  true, false,  true,
            true, false,  true, false,
           false,  true,  true, false
        }, new Shape(3, 4));
        NDArray updates = m.create(new double[] {
            0.1021, 0.4695, 0.9563,
            0.0584, 0.9399, 0.3541
        }, new Shape(2, 3));
        NDArray y_actual = m.zeros(new Shape(3, 4), DataType.FLOAT64);
        Tensor.maskedScatter(x, mask, updates, y_actual);
        NDArray y_expect = m.create(new double[] {
            0.8377, 0.1021, 0.5875, 0.4695,
            0.9563, 0.9540, 0.0584, 0.6542,
            0.2371, 0.9399, 0.3541, 0.7708
        }, new Shape(3, 4));
        assertEquals(y_expect, y_actual);
    }

    @Test
    public void test12() {
        NDManager m = Tensor.manager;
        NDArray x = m.create(new double[] {
            0.4813, 0.6845, 0.6942, 0.8727,
            0.1822, 0.2602, 0.4143, 0.1362,
            0.5056, 0.5583, 0.7134, 0.7453
        }, new Shape(3, 4));
        NDArray mask = m.create(new boolean[] {
            false,  true,  true,  true,
            true, false, false,  true,
           false,  true, false, false
        }, new Shape(3, 4));
        NDArray updates = m.create(new double[] {
            0.9037, 0.1303,
            0.3857, 0.4196
        }, new Shape(2, 2));
        NDArray y = m.zeros(new Shape(3, 4), DataType.FLOAT64);
        boolean y_actual = Tensor.maskedScatter(x, mask, updates, y);
        boolean y_expect = false;
        assertEquals(y_expect, y_actual);
    }

    @Test
    public void test13() {
        NDManager m = Tensor.manager;
        NDArray x = m.create(new double[] {

        }, new Shape(0));
        NDArray mask = m.create(new boolean[] {
            false,  true,
            true,  true
        }, new Shape(2, 2));
        NDArray updates = m.create(new double[] {
            0.4186, 0.1792,
            0.7119, 0.6155
        }, new Shape(2, 2));
        NDArray y = m.zeros(new Shape(2, 2), DataType.FLOAT64);
        boolean y_actual = Tensor.maskedScatter(x, mask, updates, y);
        boolean y_expect = false;
        assertEquals(y_expect, y_actual);
    }

    @Test
    public void test14() {
        NDManager m = Tensor.manager;
        NDArray x = m.create(new double[] {
            0.8427, 0.1193,
            0.7405, 0.4837
        }, new Shape(2, 2));
        NDArray mask = m.create(new boolean[] {
            false, false,
            true,  true
        }, new Shape(2, 2));
        NDArray updates = m.create(new double[] {

        }, new Shape(0));
        NDArray y = m.zeros(new Shape(2, 2), DataType.FLOAT64);
        boolean y_actual = Tensor.maskedScatter(x, mask, updates, y);
        boolean y_expect = false;
        assertEquals(y_expect, y_actual);
    }

    @Test
    public void test15() {
        NDManager m = Tensor.manager;
        NDArray x = m.create(new double[] {
            0.2623, 0.9701,
            0.6805, 0.9910
        }, new Shape(2, 2));
        NDArray mask = m.create(new boolean[] {
            false
        }, new Shape(1));
        NDArray updates = m.create(new double[] {

        }, new Shape(0));
        NDArray y_actual = m.zeros(new Shape(2, 2), DataType.FLOAT64);
        Tensor.maskedScatter(x, mask, updates, y_actual);
        NDArray y_expect = m.create(new double[] {
            0.2623, 0.9701,
            0.6805, 0.9910
        }, new Shape(2, 2));
        assertEquals(y_expect, y_actual);
    }

    @Test
    public void test16() {
        NDManager m = Tensor.manager;
        NDArray x = m.create(new double[] {

        }, new Shape(0));
        NDArray mask = m.create(new boolean[] {
            
        }, new Shape(0));
        NDArray updates = m.create(new double[] {

        }, new Shape(0));
        NDArray y_actual = m.zeros(new Shape(0), DataType.FLOAT64);
        Tensor.maskedScatter(x, mask, updates, y_actual);
        NDArray y_expect = m.create(new double[] {

        }, new Shape(0));
        assertEquals(y_expect, y_actual);
    }


}
