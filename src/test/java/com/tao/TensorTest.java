package com.tao;

import org.junit.Test;
import static org.junit.Assert.*;

import ai.djl.ndarray.*;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;

public class TensorTest {

    /** 正常输入 */
    /**
     * 3*3
     */
    @Test
    public void test33_1() {
        NDManager m = Tensor.manager;
        NDArray x = m.zeros(new Shape(3, 3), DataType.INT32);
        NDArray mask = m.create(new boolean[] { false, true, true }, new Shape(3));
        NDArray updates = m.create(new int[] { 1, 2, 3, 4, 5, 6, 7, 8, 9 }, new Shape(3, 3));
        NDArray y_actual = m.zeros(new Shape(3, 3), DataType.INT32);
        Tensor.maskedScatter(x, mask, updates, y_actual);
        NDArray y_expect = m.create(new int[] { 0, 1, 2, 0, 3, 4, 0, 5, 6 }, new Shape(3, 3));
        assertEquals(y_expect, y_actual);
    }

    @Test
    public void test33_2() {
        NDManager m = Tensor.manager;
        NDArray x = m.zeros(new Shape(3, 3), DataType.INT32);
        NDArray mask = m.create(new boolean[] { false, true, true }, new Shape(3, 1));
        NDArray updates = m.create(new int[] { 1, 2, 3, 4, 5, 6, 7, 8, 9 }, new Shape(3, 3));
        NDArray y_actual = m.zeros(new Shape(3, 3), DataType.INT32);
        Tensor.maskedScatter(x, mask, updates, y_actual);
        NDArray y_expect = m.create(new int[] { 0, 0, 0, 1, 2, 3, 4, 5, 6 }, new Shape(3, 3));
        assertEquals(y_expect, y_actual);
    }

    @Test
    public void test33_3() {
        NDManager m = Tensor.manager;
        NDArray x = m.zeros(new Shape(3, 3), DataType.FLOAT64);
        NDArray mask = m.create(new boolean[] { false, true, true }, new Shape(3, 1));
        NDArray updates = m.create(new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9 }, new Shape(3, 3));
        NDArray y_actual = m.zeros(new Shape(3, 3), DataType.FLOAT64);
        Tensor.maskedScatter(x, mask, updates, y_actual);
        NDArray y_expect = m.create(new double[] { 0, 0, 0, 1, 2, 3, 4, 5, 6 }, new Shape(3, 3));
        assertEquals(y_expect, y_actual);
    }

    @Test
    public void test33_4() {
        NDManager m = Tensor.manager;
        NDArray x = m.zeros(new Shape(3, 3), DataType.FLOAT64);
        NDArray mask = m.create(new boolean[] { false, true, true }, new Shape(1, 3));
        NDArray updates = m.create(new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9 }, new Shape(3, 3));
        NDArray y_actual = m.zeros(new Shape(3, 3), DataType.FLOAT64);
        Tensor.maskedScatter(x, mask, updates, y_actual);
        NDArray y_expect = m.create(new double[] { 0, 1, 2, 0, 3, 4, 0, 5, 6 }, new Shape(3, 3));
        assertEquals(y_expect, y_actual);
    }

    /**
     * 3*3*3
     */
    @Test
    public void test333_1() {
        NDManager m = Tensor.manager;
        NDArray x = m.create(new double[] {
                0.4774, 0.1360, 0.0697,
                0.1646, 0.7237, 0.0619,
                0.4612, 0.8962, 0.0488,

                0.5880, 0.7240, 0.2600,
                0.7778, 0.1454, 0.4214,
                0.2875, 0.1691, 0.5233,

                0.3404, 0.8219, 0.9836,
                0.3197, 0.3788, 0.2774,
                0.7297, 0.6689, 0.9814 },
                new Shape(3, 3, 3));
        NDArray mask = m.create(new boolean[] { false, false, true }, new Shape(3));
        NDArray updates = m.create(new double[] {
                0.6447, 0.8869, 0.4560,
                0.5209, 0.0137, 0.2536,
                0.8709, 0.2477, 0.2619,

                0.7181, 0.4074, 0.0622,
                0.5067, 0.5615, 0.9481,
                0.6292, 0.2745, 0.8706,

                0.7305, 0.2327, 0.3944,
                0.5540, 0.1753, 0.5758,
                0.9208, 0.0011, 0.1562 }, new Shape(3, 3, 3));
        NDArray y_actual = m.zeros(new Shape(3, 3, 3), DataType.FLOAT64);
        NDArray y_expect = m.create(new double[] {
                0.4774, 0.1360, 0.6447,
                0.1646, 0.7237, 0.8869,
                0.4612, 0.8962, 0.4560,

                0.5880, 0.7240, 0.5209,
                0.7778, 0.1454, 0.0137,
                0.2875, 0.1691, 0.2536,

                0.3404, 0.8219, 0.8709,
                0.3197, 0.3788, 0.2477,
                0.7297, 0.6689, 0.2619 }, new Shape(3, 3, 3));
        Tensor.maskedScatter(x, mask, updates, y_actual);
        assertEquals(y_expect, y_actual);
    }

    @Test
    public void test333_2() {
        NDManager m = Tensor.manager;
        NDArray x = m.create(new double[] {
                0.7789, 0.2602, 0.2318,
                0.1010, 0.8112, 0.9630,
                0.6214, 0.2237, 0.8044,

                0.0273, 0.2524, 0.9237,
                0.6468, 0.7424, 0.3968,
                0.8712, 0.3899, 0.6419,

                0.3414, 0.2407, 0.7195,
                0.9236, 0.2471, 0.7861,
                0.7414, 0.0676, 0.9528
        }, new Shape(3, 3, 3));
        NDArray mask = m.create(new boolean[] { true, true, false }, new Shape(3, 1));
        NDArray updates = m.create(new double[] {
                0.4461, 0.9596, 0.2282,
                0.8657, 0.8637, 0.1641,
                0.6168, 0.4141, 0.8182,

                0.0595, 0.9553, 0.1631,
                0.3746, 0.1503, 0.3976,
                0.3306, 0.7027, 0.4418,

                0.8940, 0.1191, 0.5033,
                0.7882, 0.0973, 0.3063,
                0.6692, 0.1426, 0.4431
        }, new Shape(3, 3, 3));
        NDArray y_actual = m.zeros(new Shape(3, 3, 3), DataType.FLOAT64);
        NDArray y_expect = m.create(new double[] {
                0.4461, 0.9596, 0.2282,
                0.8657, 0.8637, 0.1641,
                0.6214, 0.2237, 0.8044,

                0.6168, 0.4141, 0.8182,
                0.0595, 0.9553, 0.1631,
                0.8712, 0.3899, 0.6419,

                0.3746, 0.1503, 0.3976,
                0.3306, 0.7027, 0.4418,
                0.7414, 0.0676, 0.9528
        }, new Shape(3, 3, 3));
        Tensor.maskedScatter(x, mask, updates, y_actual);
        assertEquals(y_expect, y_actual);
    }

    @Test
    public void test333_3() {
        NDManager m = Tensor.manager;
        NDArray x = m.create(new double[] {
                0.9819, 0.6340, 0.9174,
                0.9971, 0.2606, 0.7712,
                0.6195, 0.5655, 0.0033,

                0.8214, 0.7851, 0.4661,
                0.2974, 0.4974, 0.4791,
                0.3276, 0.3975, 0.3361,

                0.0251, 0.2369, 0.6856,
                0.5940, 0.6229, 0.3101,
                0.9314, 0.4520, 0.4510 },
                new Shape(3, 3, 3));
        NDArray mask = m.create(new boolean[] {
                false, false, false,
                false, true, true,
                true, false, true }, new Shape(3, 3));
        NDArray updates = m.create(new double[] {
                0.1385, 0.5424, 0.9034,
                0.6423, 0.5326, 0.9533,
                0.4127, 0.1187, 0.8563,

                0.8838, 0.8574, 0.6585,
                0.8945, 0.2632, 0.6878,
                0.0665, 0.0019, 0.7840,

                0.5748, 0.0944, 0.5257,
                0.7905, 0.7199, 0.7111,
                0.5045, 0.3913, 0.2594 }, new Shape(3, 3, 3));
        NDArray y_actual = m.zeros(new Shape(3, 3, 3), DataType.FLOAT64);
        NDArray y_expect = m.create(new double[] {
                0.9819, 0.6340, 0.9174,
                0.9971, 0.1385, 0.5424,
                0.9034, 0.5655, 0.6423,

                0.8214, 0.7851, 0.4661,
                0.2974, 0.5326, 0.9533,
                0.4127, 0.3975, 0.1187,

                0.0251, 0.2369, 0.6856,
                0.5940, 0.8563, 0.8838,
                0.8574, 0.4520, 0.6585 }, new Shape(3, 3, 3));
        Tensor.maskedScatter(x, mask, updates, y_actual);
        assertEquals(y_expect, y_actual);
    }

    @Test
    public void test333_4() {
        NDManager m = Tensor.manager;
        NDArray x = m.create(new double[] {
                0.9819, 0.6340, 0.9174,
                0.9971, 0.2606, 0.7712,
                0.6195, 0.5655, 0.0033,

                0.8214, 0.7851, 0.4661,
                0.2974, 0.4974, 0.4791,
                0.3276, 0.3975, 0.3361,

                0.0251, 0.2369, 0.6856,
                0.5940, 0.6229, 0.3101,
                0.9314, 0.4520, 0.4510 },
                new Shape(3, 3, 3));
        NDArray mask = m.create(new boolean[] {
                false, false, false,
                false, true, true,
                true, false, true }, new Shape(3, 3));
        NDArray updates = m.create(new double[] {
                0.1385, 0.5424, 0.9034,
                0.6423, 0.5326, 0.9533,
                0.4127, 0.1187, 0.8563,

                0.8838, 0.8574, 0.6585,
                0.8945, 0.2632, 0.6878,
                0.0665, 0.0019, 0.7840,

                0.5748, 0.0944, 0.5257,
                0.7905, 0.7199, 0.7111,
                0.5045, 0.3913, 0.2594 }, new Shape(3, 3, 3));
        NDArray y_actual = m.zeros(new Shape(3, 3, 3), DataType.FLOAT64);
        NDArray y_expect = m.create(new double[] {
                0.9819, 0.6340, 0.9174,
                0.9971, 0.1385, 0.5424,
                0.9034, 0.5655, 0.6423,

                0.8214, 0.7851, 0.4661,
                0.2974, 0.5326, 0.9533,
                0.4127, 0.3975, 0.1187,

                0.0251, 0.2369, 0.6856,
                0.5940, 0.8563, 0.8838,
                0.8574, 0.4520, 0.6585 }, new Shape(3, 3, 3));
        Tensor.maskedScatter(x, mask, updates, y_actual);
        assertEquals(y_expect, y_actual);
    }

    @Test
    public void test333_5() {
        NDManager m = Tensor.manager;
        NDArray x = m.create(new double[] {
                0.7492, 0.9021, 0.6062,
                0.6009, 0.4625, 0.4314,
                0.2518, 0.0204, 0.1531,

                0.0076, 0.2375, 0.9213,
                0.4559, 0.0685, 0.6553,
                0.8130, 0.8851, 0.6454,

                0.8622, 0.9934, 0.2819,
                0.7633, 0.1386, 0.7309,
                0.8927, 0.5179, 0.3387
        },
                new Shape(3, 3, 3));
        NDArray mask = m.create(new boolean[] {
                false, false, false,
                true, true, true,
                false, false, true,

                false, true, false,
                true, false, false,
                false, true, false,

                true, true, true,
                true, true, true,
                false, false, true
        }, new Shape(3, 3, 3));
        NDArray updates = m.create(new double[] {
                0.6502, 0.8548, 0.9681,
                0.0362, 0.4069, 0.7808,
                0.2806, 0.2954, 0.6191,

                0.3253, 0.8897, 0.7351,
                0.5056, 0.4200, 0.0354,
                0.5496, 0.6933, 0.5648,

                0.8572, 0.3360, 0.9320,
                0.4687, 0.5034, 0.4238,
                0.6980, 0.2238, 0.0016
        }, new Shape(3, 3, 3));
        NDArray y_actual = m.zeros(new Shape(3, 3, 3), DataType.FLOAT64);
        NDArray y_expect = m.create(new double[] {
                0.7492, 0.9021, 0.6062,
                0.6502, 0.8548, 0.9681,
                0.2518, 0.0204, 0.0362,

                0.0076, 0.4069, 0.9213,
                0.7808, 0.0685, 0.6553,
                0.8130, 0.2806, 0.6454,

                0.2954, 0.6191, 0.3253,
                0.8897, 0.7351, 0.5056,
                0.8927, 0.5179, 0.4200
        }, new Shape(3, 3, 3));
        Tensor.maskedScatter(x, mask, updates, y_actual);
        assertEquals(y_expect, y_actual);
    }

    @Test
    public void test333_6() {
        NDManager m = Tensor.manager;
        NDArray x = m.create(new double[] {
                0.5678, 0.7887, 0.9631,
                0.7531, 0.9933, 0.5019,
                0.5097, 0.8156, 0.8831,

                0.9199, 0.4910, 0.4088,
                1.0000, 0.8208, 0.2094,
                0.2332, 0.5502, 0.4713,

                0.8723, 0.1626, 0.3167,
                0.4935, 0.1496, 0.8754,
                0.7345, 0.1606, 0.2148

        },
                new Shape(3, 3, 3));
        NDArray mask = m.create(new boolean[] {
                false,
                false,
                true,

                false,
                false,
                true,

                true,
                true,
                false
        }, new Shape(3, 3, 1));
        NDArray updates = m.create(new double[] {
                0.8737, 0.7091, 0.9660,
                0.5399, 0.3202, 0.1687,
                0.8961, 0.1126, 0.8812,

                0.7374, 0.6445, 0.9061,
                0.3117, 0.5785, 0.5141,
                0.5674, 0.7359, 0.0945,

                0.7705, 0.7767, 0.4216,
                0.5197, 0.0332, 0.6046,
                0.6820, 0.4621, 0.2223

        }, new Shape(3, 3, 3));
        NDArray y_actual = m.zeros(new Shape(3, 3, 3), DataType.FLOAT64);
        NDArray y_expect = m.create(new double[] {
                0.5678, 0.7887, 0.9631,
                0.7531, 0.9933, 0.5019,
                0.8737, 0.7091, 0.9660,

                0.9199, 0.4910, 0.4088,
                1.0000, 0.8208, 0.2094,
                0.5399, 0.3202, 0.1687,

                0.8961, 0.1126, 0.8812,
                0.7374, 0.6445, 0.9061,
                0.7345, 0.1606, 0.2148
        }, new Shape(3, 3, 3));
        Tensor.maskedScatter(x, mask, updates, y_actual);
        assertEquals(y_expect, y_actual);
    }

    @Test
    public void test333_7() {
        NDManager m = Tensor.manager;
        NDArray x = m.create(new double[] {
                0.6701, 0.7282, 0.2242,
                0.7863, 0.7011, 0.4425,
                0.0949, 0.3803, 0.4245,

                0.5604, 0.7776, 0.8324,
                0.6680, 0.1986, 0.3296,
                0.2923, 0.1464, 0.9667,

                0.4651, 0.5222, 0.3982,
                0.6596, 0.0150, 0.8581,
                0.6357, 0.1729, 0.3488
        },
                new Shape(3, 3, 3));
        NDArray mask = m.create(new boolean[] {
                true, true, false,

                false, true, false,

                false, true, false
        }, new Shape(3, 1, 3));
        NDArray updates = m.create(new double[] {
                0.1941, 0.5612, 0.5034,
                0.0981, 0.4342, 0.5079,
                0.5422, 0.4044, 0.0713,

                0.6067, 0.5399, 0.0014,
                0.4103, 0.8296, 0.9269,
                0.9784, 0.1288, 0.5173,

                0.6423, 0.3689, 0.0357,
                0.1419, 0.5529, 0.6642,
                0.1129, 0.8671, 0.9650
        }, new Shape(3, 3, 3));
        NDArray y_actual = m.zeros(new Shape(3, 3, 3), DataType.FLOAT64);
        NDArray y_expect = m.create(new double[] {
                0.1941, 0.5612, 0.2242,
                0.5034, 0.0981, 0.4425,
                0.4342, 0.5079, 0.4245,

                0.5604, 0.5422, 0.8324,
                0.6680, 0.4044, 0.3296,
                0.2923, 0.0713, 0.9667,

                0.4651, 0.6067, 0.3982,
                0.6596, 0.5399, 0.8581,
                0.6357, 0.0014, 0.3488
        }, new Shape(3, 3, 3));
        Tensor.maskedScatter(x, mask, updates, y_actual);
        assertEquals(y_expect, y_actual);
    }

    @Test
    public void test333_8() {
        NDManager m = Tensor.manager;
        NDArray x = m.create(new double[] {
                0.8533, 0.1123, 0.8300,
                0.3527, 0.1258, 0.5417,
                0.0823, 0.7613, 0.5664,

                0.5889, 0.5292, 0.1592,
                0.3630, 0.7140, 0.6095,
                0.6890, 0.9848, 0.3191,

                0.3517, 0.3624, 0.1636,
                0.7722, 0.9947, 0.0816,
                0.7312, 0.4310, 0.1233
        },
                new Shape(3, 3, 3));
        NDArray mask = m.create(new boolean[] {
                true, true, false,
                false, true, false,
                false, false, true
        }, new Shape(1, 3, 3));
        NDArray updates = m.create(new double[] {
                0.2804, 0.6767, 0.3561,
                0.4426, 0.2168, 0.8523,
                0.8281, 0.8559, 0.8546,

                0.9429, 0.4689, 0.0234,
                0.7324, 0.8547, 0.6448,
                0.1771, 0.6570, 0.5335,

                0.0794, 0.4100, 0.7405,
                0.1898, 0.3662, 0.9949,
                0.0163, 0.5092, 0.3647
        }, new Shape(3, 3, 3));
        NDArray y_actual = m.zeros(new Shape(3, 3, 3), DataType.FLOAT64);
        NDArray y_expect = m.create(new double[] {
                0.2804, 0.6767, 0.8300,
                0.3527, 0.3561, 0.5417,
                0.0823, 0.7613, 0.4426,

                0.2168, 0.8523, 0.1592,
                0.3630, 0.8281, 0.6095,
                0.6890, 0.9848, 0.8559,

                0.8546, 0.9429, 0.1636,
                0.7722, 0.4689, 0.0816,
                0.7312, 0.4310, 0.0234
        }, new Shape(3, 3, 3));
        Tensor.maskedScatter(x, mask, updates, y_actual);
        assertEquals(y_expect, y_actual);
    }

    @Test
    public void test333_9() {
        NDManager m = Tensor.manager;
        NDArray x = m.create(new double[] {
                0.7771, 0.5305, 0.7887,
                0.4738, 0.4664, 0.6520,
                0.9773, 0.5566, 0.3383,

                0.1995, 0.8435, 0.7483,
                0.1751, 0.2433, 0.9006,
                0.0745, 0.9756, 0.2979,

                0.9252, 0.0914, 0.9839,
                0.2600, 0.1081, 0.4249,
                0.4544, 0.2068, 0.7743
        },
                new Shape(3, 3, 3));
        NDArray mask = m.create(new boolean[] { true, false, true }, new Shape(1, 1, 3));
        NDArray updates = m.create(new double[] {
                0.9725, 0.2987, 0.1122,
                0.3714, 0.8254, 0.7877,
                0.1857, 0.9542, 0.4884,

                0.1791, 0.5034, 0.6371,
                0.5358, 0.6920, 0.1668,
                0.9273, 0.0961, 0.5322,

                0.8285, 0.5665, 0.8699,
                0.7944, 0.4729, 0.6335,
                0.9097, 0.9601, 0.9142
        }, new Shape(3, 3, 3));
        NDArray y_actual = m.zeros(new Shape(3, 3, 3), DataType.FLOAT64);
        NDArray y_expect = m.create(new double[] {
                0.9725, 0.5305, 0.2987,
                0.1122, 0.4664, 0.3714,
                0.8254, 0.5566, 0.7877,

                0.1857, 0.8435, 0.9542,
                0.4884, 0.2433, 0.1791,
                0.5034, 0.9756, 0.6371,

                0.5358, 0.0914, 0.6920,
                0.1668, 0.1081, 0.9273,
                0.0961, 0.2068, 0.5322
        }, new Shape(3, 3, 3));
        Tensor.maskedScatter(x, mask, updates, y_actual);
        assertEquals(y_expect, y_actual);
    }

    @Test
    public void test333_10() {
        NDManager m = Tensor.manager;
        NDArray x = m.create(new double[] {
                0.5011, 0.2961, 0.2444,
                0.1470, 0.2006, 0.1333,
                0.9428, 0.4930, 0.5194,

                0.0793, 0.1274, 0.3742,
                0.6505, 0.8602, 0.7552,
                0.5682, 0.3283, 0.1983,

                0.7657, 0.4080, 0.9531,
                0.6968, 0.1242, 0.6924,
                0.8950, 0.4364, 0.4885
        },
                new Shape(3, 3, 3));
        NDArray mask = m.create(new boolean[] { true, false, true }, new Shape(1, 3, 1));
        NDArray updates = m.create(new double[] {
                0.5659, 0.0733, 0.0483,
                0.3982, 0.8073, 0.4831,
                0.1602, 0.3550, 0.4630,

                0.6603, 0.0395, 0.3635,
                0.7736, 0.4326, 0.1766,
                0.0622, 0.8396, 0.3178,

                0.4848, 0.8578, 0.1803,
                0.1187, 0.0621, 0.9121,
                0.4390, 0.3001, 0.8609
        }, new Shape(3, 3, 3));
        NDArray y_actual = m.zeros(new Shape(3, 3, 3), DataType.FLOAT64);
        NDArray y_expect = m.create(new double[] {
                0.5659, 0.0733, 0.0483,
                0.1470, 0.2006, 0.1333,
                0.3982, 0.8073, 0.4831,

                0.1602, 0.3550, 0.4630,
                0.6505, 0.8602, 0.7552,
                0.6603, 0.0395, 0.3635,

                0.7736, 0.4326, 0.1766,
                0.6968, 0.1242, 0.6924,
                0.0622, 0.8396, 0.3178
        }, new Shape(3, 3, 3));
        Tensor.maskedScatter(x, mask, updates, y_actual);
        assertEquals(y_expect, y_actual);
    }

    @Test
    public void test333_11() {
        NDManager m = Tensor.manager;
        NDArray x = m.create(new double[] {
                0.3528, 0.1933, 0.4743,
                0.0747, 0.2394, 0.9623,
                0.7310, 0.4945, 0.5050,

                0.0648, 0.5802, 0.6920,
                0.3696, 0.0221, 0.1041,
                0.0646, 0.6132, 0.7689,

                0.8109, 0.6276, 0.2914,
                0.9766, 0.9546, 0.6133,
                0.9574, 0.0651, 0.4796
        },
                new Shape(3, 3, 3));
        NDArray mask = m.create(new boolean[] { true, false, true }, new Shape(3, 1, 1));
        NDArray updates = m.create(new double[] {
                0.0297, 0.8418, 0.9879,
                0.8052, 0.7440, 0.0471,
                0.3500, 0.7198, 0.6254,

                0.7948, 0.3280, 0.1514,
                0.3580, 0.4764, 0.3296,
                0.7061, 0.7386, 0.9806,

                0.0549, 0.0895, 0.8352,
                0.8358, 0.5277, 0.4942,
                0.3061, 0.4281, 0.8736
        }, new Shape(3, 3, 3));
        NDArray y_actual = m.zeros(new Shape(3, 3, 3), DataType.FLOAT64);
        NDArray y_expect = m.create(new double[] {
                0.0297, 0.8418, 0.9879,
                0.8052, 0.7440, 0.0471,
                0.3500, 0.7198, 0.6254,

                0.0648, 0.5802, 0.6920,
                0.3696, 0.0221, 0.1041,
                0.0646, 0.6132, 0.7689,

                0.7948, 0.3280, 0.1514,
                0.3580, 0.4764, 0.3296,
                0.7061, 0.7386, 0.9806
        }, new Shape(3, 3, 3));
        Tensor.maskedScatter(x, mask, updates, y_actual);
        assertEquals(y_expect, y_actual);
    }

    @Test
    public void test333_12() {
        NDManager m = Tensor.manager;
        NDArray x = m.create(new double[] {
                0.6873, 0.2638, 0.3573,
                0.1681, 0.6673, 0.8201,
                0.0950, 0.2872, 0.3434,

                0.5474, 0.2011, 0.9509,
                0.5429, 0.3066, 0.0499,
                0.2591, 0.4114, 0.0122,

                0.1089, 0.8882, 0.4123,
                0.8714, 0.8358, 0.6581,
                0.5174, 0.2306, 0.6823
        },
                new Shape(3, 3, 3));
        NDArray mask = m.create(new boolean[] { false }, new Shape(1, 1, 1));
        NDArray updates = m.create(new double[] {
                0.1961, 0.9552, 0.6031,
                0.8763, 0.0203, 0.9460,
                0.7715, 0.3556, 0.3111,

                0.3384, 0.2170, 0.1729,
                0.3127, 0.4532, 0.2056,
                0.7662, 0.2355, 0.7825,

                0.3179, 0.7787, 0.4391,
                0.4640, 0.5069, 0.6185,
                0.3996, 0.6212, 0.8842
        }, new Shape(3, 3, 3));
        NDArray y_actual = m.zeros(new Shape(3, 3, 3), DataType.FLOAT64);
        NDArray y_expect = m.create(new double[] {
                0.6873, 0.2638, 0.3573,
                0.1681, 0.6673, 0.8201,
                0.0950, 0.2872, 0.3434,

                0.5474, 0.2011, 0.9509,
                0.5429, 0.3066, 0.0499,
                0.2591, 0.4114, 0.0122,

                0.1089, 0.8882, 0.4123,
                0.8714, 0.8358, 0.6581,
                0.5174, 0.2306, 0.6823
        }, new Shape(3, 3, 3));
        Tensor.maskedScatter(x, mask, updates, y_actual);
        assertEquals(y_expect, y_actual);
    }

    /** 测试非正常输入 */
    /**
     * mask非法输入
     */
    @Test
    public void testInvalidMaskInput1() {
        NDManager m = Tensor.manager;
        NDArray x = m.zeros(new Shape(3, 3), DataType.FLOAT64);
        NDArray mask = m.create(new boolean[] { false, true, true }, new Shape(4, 1));
        NDArray updates = m.create(new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9 }, new Shape(3, 3));
        NDArray y = m.zeros(new Shape(3, 3), DataType.FLOAT64);
        boolean y_actual = Tensor.maskedScatter(x, mask, updates, y);
        boolean y_expect = false;
        assertEquals(y_expect, y_actual);
    }

    @Test
    public void testInvalidMaskInput2() {
        NDManager m = Tensor.manager;
        NDArray x = m.zeros(new Shape(3, 3), DataType.FLOAT64);
        NDArray mask = m.create(new boolean[] { false, true, true }, new Shape(3, 3, 3));
        NDArray updates = m.create(new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9 }, new Shape(3, 3));
        NDArray y = m.zeros(new Shape(3, 3), DataType.FLOAT64);
        boolean y_actual = Tensor.maskedScatter(x, mask, updates, y);
        boolean y_expect = false;
        assertEquals(y_expect, y_actual);
    }

    /**
     * 其他非法输入
     */
    @Test
    public void testInvalidInput1() {
        NDManager m = Tensor.manager;
        NDArray x = m.zeros(new Shape(4, 3), DataType.FLOAT64);
        NDArray mask = m.create(new boolean[] { false, true, true }, new Shape(3, 3));
        NDArray updates = m.create(new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9 }, new Shape(3, 3));
        NDArray y = m.zeros(new Shape(3, 3), DataType.FLOAT64);
        boolean y_actual = Tensor.maskedScatter(x, mask, updates, y);
        boolean y_expect = false;
        assertEquals(y_expect, y_actual);
    }

    @Test
    public void testInvalidInput2() {
        NDManager m = Tensor.manager;
        NDArray x = m.zeros(new Shape(3, 3), DataType.FLOAT64);
        NDArray mask = m.create(new boolean[] { false, true, true }, new Shape(3, 3));
        NDArray updates = m.create(new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 }, new Shape(4, 3));
        NDArray y = m.zeros(new Shape(3, 3), DataType.FLOAT64);
        boolean y_actual = Tensor.maskedScatter(x, mask, updates, y);
        boolean y_expect = false;
        assertEquals(y_expect, y_actual);
    }

    @Test
    public void testInvalidInput3() {
        NDManager m = Tensor.manager;
        NDArray x = m.zeros(new Shape(3, 3), DataType.FLOAT64);
        NDArray mask = m.create(new boolean[] { false, true, true }, new Shape(3, 3));
        NDArray updates = m.create(new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9 }, new Shape(3, 3));
        NDArray y = m.zeros(new Shape(4, 3), DataType.FLOAT64);
        boolean y_actual = Tensor.maskedScatter(x, mask, updates, y);
        boolean y_expect = false;
        assertEquals(y_expect, y_actual);
    }

    /**
     * 输入为NULL的测�?
     */
    @Test
    public void testNull1() {
        NDManager m = Tensor.manager;
        NDArray x = null;
        NDArray mask = m.create(new boolean[] { false, true, true }, new Shape(3, 3));
        NDArray updates = m.create(new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9 }, new Shape(3, 3));
        NDArray y = m.zeros(new Shape(4, 3), DataType.FLOAT64);
        boolean y_actual = Tensor.maskedScatter(x, mask, updates, y);
        boolean y_expect = false;
        assertEquals(y_expect, y_actual);
    }

    @Test
    public void testNull2() {
        NDManager m = Tensor.manager;
        NDArray x = m.zeros(new Shape(3, 3), DataType.FLOAT64);
        NDArray mask = null;
        NDArray updates = m.create(new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9 }, new Shape(3, 3));
        NDArray y = m.create(new int[] { 0, 1, 2, 0, 3, 4, 0, 5, 6 }, new Shape(3, 3));
        boolean y_actual = Tensor.maskedScatter(x, mask, updates, y);
        boolean y_expect = false;
        assertEquals(y_expect, y_actual);
    }

    @Test
    public void testNull3() {
        NDManager m = Tensor.manager;
        NDArray x = m.zeros(new Shape(3, 3), DataType.FLOAT64);
        NDArray mask = m.create(new boolean[] { false, true, true }, new Shape(3, 3));
        NDArray updates = null;
        NDArray y = m.create(new int[] { 0, 1, 2, 0, 3, 4, 0, 5, 6 }, new Shape(3, 3));
        boolean y_actual = Tensor.maskedScatter(x, mask, updates, y);
        boolean y_expect = false;
        assertEquals(y_expect, y_actual);
    }

    @Test
    public void testNull4() {
        NDManager m = Tensor.manager;
        NDArray x = m.zeros(new Shape(3, 3), DataType.FLOAT64);
        NDArray mask = m.create(new boolean[] { false, true, true }, new Shape(3, 3));
        NDArray updates = m.create(new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9 }, new Shape(3, 3));
        NDArray y = null;
        boolean y_actual = Tensor.maskedScatter(x, mask, updates, y);
        boolean y_expect = false;
        assertEquals(y_expect, y_actual);
    }

}