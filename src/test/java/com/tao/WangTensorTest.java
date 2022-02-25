package com.tao;

import org.junit.Test;
import static org.junit.Assert.*;

import javax.swing.text.StyledEditorKit.BoldAction;

import ai.djl.ndarray.*;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;

public class WangTensorTest {
        @Test
        public void test1() {

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

        @Test
        public void test2() {
                NDManager m = Tensor.manager;
                NDArray x = m.create(new double[] {

                }, new Shape(0));
                NDArray mask = m.create(new boolean[] {

                }, new Shape(0));
                NDArray updates = m.create(new double[] {
                                0.3840, 0.2100, 0.3511, 0.6027
                }, new Shape(4));
                NDArray y_actual = m.zeros(new Shape(0), DataType.FLOAT64);
                Tensor.maskedScatter(x, mask, updates, y_actual);
                NDArray y_expect = m.create(new double[] {

                }, new Shape(0));
                assertEquals(y_expect, y_actual);
        }

        @Test
        public void test3() {
                NDManager m = Tensor.manager;
                NDArray x = m.create(new double[] {

                }, new Shape(0));
                NDArray mask = m.create(new boolean[] {
                                true, true, true
                }, new Shape(3));
                NDArray updates = m.create(new double[] {
                                0.7262, 0.9153
                }, new Shape(2));
                NDArray y = m.zeros(new Shape(0), DataType.FLOAT64);
                boolean y_actual = Tensor.maskedScatter(x, mask, updates, y);
                boolean y_expect = false;
                assertEquals(y_expect, y_actual);
        }

        @Test
        public void test4() {

                NDManager m = Tensor.manager;
                NDArray x = m.create(new double[] {
                                0.3761, 0.2968, 0.2826, 0.5888
                }, new Shape(4));
                NDArray mask = m.create(new boolean[] {

                }, new Shape(0));
                NDArray updates = m.create(new double[] {
                                0.3771, 0.5770, 0.7343, 0.9323
                }, new Shape(4));
                NDArray y = m.zeros(new Shape(4), DataType.FLOAT64);
                boolean y_actual = Tensor.maskedScatter(x, mask, updates, y);
                boolean y_expect = false;
                assertEquals(y_expect, y_actual);
        }

        @Test
        public void test5() {

                NDManager m = Tensor.manager;
                NDArray x = m.create(new double[] {
                                0.3916, 0.1392, 0.7842, 0.4136, 0.6478, 0.6296, 0.9743, 0.5402, 0.1320
                }, new Shape(3, 3));
                NDArray mask = m.create(new boolean[] {
                                true, true, false, false, true, true, false, true, false
                }, new Shape(3, 3));
                NDArray updates = m.create(new double[] {
                                0.4034, 0.1730, 0.6595, 0.8018, 0.1061, 0.8899, 0.2775, 0.9095, 0.0953
                }, new Shape(3, 3));
                NDArray y_actual = m.zeros(new Shape(3, 3), DataType.FLOAT64);
                Tensor.maskedScatter(x, mask, updates, y_actual);
                NDArray y_expect = m.create(new double[] {
                                0.4034, 0.1730, 0.7842, 0.4136, 0.6595, 0.8018, 0.9743, 0.1061, 0.1320
                }, new Shape(3, 3));
                assertEquals(y_expect, y_actual);
        }

        @Test
        public void test6() {

                NDManager m = Tensor.manager;
                NDArray x = m.create(new double[] {
                                0.4161, 0.8310, 0.8963, 0.0647, 0.0827, 0.1088, 0.3405, 0.5642, 0.0898, 0.5647, 0.5582,
                                0.1023, 0.0602, 0.4248, 0.0880, 0.9113, 0.2527, 0.5460
                }, new Shape(2, 3, 3));
                NDArray mask = m.create(new boolean[] {
                                false, true, true, true, false, true, false, false, true, true, false, true, false,
                                true, false, false, true, true
                }, new Shape(2, 3, 3));
                NDArray updates = m.create(new double[] {
                                0.3877, 0.9804, 0.3160, 0.0168, 0.1470, 0.4931, 0.4390, 0.5641, 0.0539, 0.2294, 0.5492,
                                0.2082, 0.4027, 0.7244, 0.2857, 0.7976, 0.8496, 0.9047
                }, new Shape(2, 3, 3));
                NDArray y_actual = m.zeros(new Shape(2, 3, 3), DataType.FLOAT64);
                Tensor.maskedScatter(x, mask, updates, y_actual);
                NDArray y_expect = m.create(new double[] {
                                0.4161, 0.3877, 0.9804, 0.3160, 0.0827, 0.0168, 0.3405, 0.5642, 0.1470, 0.4931, 0.5582,
                                0.4390, 0.0602, 0.5641, 0.0880, 0.9113, 0.0539, 0.2294
                }, new Shape(2, 3, 3));
                assertEquals(y_expect, y_actual);
        }

        @Test
        public void test7() {

                NDManager m = Tensor.manager;
                NDArray x = m.create(new double[] {
                                0.4003, 0.3103, 0.0075, 0.5454, 0.8088, 0.5923, 0.4071, 0.3081, 0.5445, 0.5888, 0.9805,
                                0.3776, 0.5314, 0.3486, 0.6193, 0.4663, 0.9706, 0.7537, 0.7229, 0.7402, 0.7725, 0.3463,
                                0.3426, 0.5100, 0.1091, 0.7983, 0.3994
                }, new Shape(3, 3, 3));
                NDArray mask = m.create(new boolean[] {
                                true, true, false, false, true, true, true, false, false
                }, new Shape(3, 3));
                NDArray updates = m.create(new double[] {
                                0.4013, 0.5905, 0.4592, 0.8889, 0.9566, 0.7906, 0.3569, 0.4148, 0.5834, 0.7125, 0.9354,
                                0.3910, 0.2877, 0.3981, 0.3919, 0.7574, 0.1548, 0.3766, 0.2815, 0.9193, 0.2562, 0.0707,
                                0.9248, 0.6507, 0.5656, 0.1732, 0.9022
                }, new Shape(3, 3, 3));
                NDArray y_actual = m.zeros(new Shape(3, 3, 3), DataType.FLOAT64);
                Tensor.maskedScatter(x, mask, updates, y_actual);
                NDArray y_expect = m.create(new double[] {
                                0.4013, 0.5905, 0.0075, 0.5454, 0.4592, 0.8889, 0.9566, 0.3081, 0.5445, 0.7906, 0.3569,
                                0.3776, 0.5314, 0.4148, 0.5834, 0.7125, 0.9706, 0.7537, 0.9354, 0.3910, 0.7725, 0.3463,
                                0.2877, 0.3981, 0.3919, 0.7983, 0.3994
                }, new Shape(3, 3, 3));
                assertEquals(y_expect, y_actual);
        }

        @Test
        public void test8() {

                NDManager m = Tensor.manager;
                NDArray x = m.create(new double[] {
                                0.4138, 0.9204, 0.1508, 0.4176, 0.6183, 0.8897, 0.3250, 0.1588, 0.0740, 0.0719, 0.3666,
                                0.5604, 0.4164, 0.0222, 0.7255, 0.4261, 0.2757, 0.2255, 0.3211, 0.3768, 0.1512, 0.5983,
                                0.6600, 0.6233, 0.7172, 0.6966, 0.8382
                }, new Shape(3, 3, 3));
                NDArray mask = m.create(new boolean[] {
                                true, false, true, false, true, true, true, false, false
                }, new Shape(3, 3, 1));
                NDArray updates = m.create(new double[] {
                                0.4148, 0.2006, 0.6025, 0.7611, 0.7661, 0.0880, 0.2748, 0.2654, 0.1130, 0.1956, 0.3215,
                                0.5738, 0.1727, 0.0718, 0.4980, 0.7173, 0.4599, 0.8485, 0.8798, 0.5559, 0.6350, 0.3227,
                                0.2422, 0.7640, 0.1737, 0.0714, 0.3410
                }, new Shape(3, 3, 3));
                NDArray y_actual = m.zeros(new Shape(3, 3, 3), DataType.FLOAT64);
                Tensor.maskedScatter(x, mask, updates, y_actual);
                NDArray y_expect = m.create(new double[] {
                                0.4148, 0.2006, 0.6025, 0.4176, 0.6183, 0.8897, 0.7611, 0.7661, 0.0880, 0.0719, 0.3666,
                                0.5604, 0.2748, 0.2654, 0.1130, 0.1956, 0.3215, 0.5738, 0.1727, 0.0718, 0.4980, 0.5983,
                                0.6600, 0.6233, 0.7172, 0.6966, 0.8382
                }, new Shape(3, 3, 3));
                assertEquals(y_expect, y_actual);
        }

        @Test
        public void test9() {

                NDManager m = Tensor.manager;
                NDArray x = m.create(new double[] {
                                0.4236, 0.0659, 0.5776, 0.3842, 0.4663, 0.6335, 0.6335, 0.4041, 0.4556
                }, new Shape(3, 3));
                NDArray mask = m.create(new boolean[] {
                                true, true, true, true, false, true, true, true, true, false, false, true, true, false,
                                true, true, true, true, true, true, true, true, true, false, false, false, true
                }, new Shape(3, 3, 3));
                NDArray updates = m.create(new double[] {
                                0.4221, 0.1457, 0.4000, 0.3689, 0.7447, 0.8360, 0.2087, 0.7441, 0.8972, 0.4991, 0.9920,
                                0.6720, 0.4933, 0.9334, 0.9373, 0.3428, 0.2046, 0.8959, 0.9582, 0.9638, 0.0662, 0.8109,
                                0.6919, 0.3910, 0.7059, 0.7080, 0.8119
                }, new Shape(3, 3, 3));
                NDArray y = m.zeros(new Shape(3, 3, 3), DataType.FLOAT64);
                boolean y_actual = Tensor.maskedScatter(x, mask, updates, y);
                boolean y_expect = false;
                assertEquals(y_expect, y_actual);
        }

        @Test
        public void test10() {

                NDManager m = Tensor.manager;
                NDArray x = m.create(new double[] {
                                0.4302, 0.7150, 0.5589, 0.8516, 0.2415, 0.5426, 0.9024, 0.5081, 0.7126
                }, new Shape(3, 3, 1));
                NDArray mask = m.create(new boolean[] {
                                true, true, false, true, false, false, true, false, true, true, true, true, false,
                                false, false, false, false, true, false, true, false, false, false, false, true, true,
                                true
                }, new Shape(3, 3, 3));
                NDArray updates = m.create(new double[] {
                                0.4310, 0.3391, 0.9203, 0.7265, 0.7598, 0.5012, 0.6623, 0.7935, 0.7438, 0.8005, 0.5903,
                                0.7916, 0.6239, 0.8743, 0.2127, 0.4342, 0.0440, 0.9400, 0.9306, 0.8583, 0.0715, 0.8581,
                                0.4732, 0.4431, 0.3685, 0.7444, 0.6872
                }, new Shape(3, 3, 3));
                NDArray y = m.zeros(new Shape(3, 3, 3), DataType.FLOAT64);
                boolean y_actual = Tensor.maskedScatter(x, mask, updates, y);
                boolean y_expect = false;
                assertEquals(y_expect, y_actual);
        }

        @Test
        public void test11() {

                NDManager m = Tensor.manager;
                NDArray x = m.create(new double[] {
                                0.4725, 0.1215, 0.2567, 0.9521, 0.0212, 0.4721, 0.2708, 0.5415, 0.8675, 0.5613, 0.7081,
                                0.3521, 0.3595, 0.4400, 0.6264, 0.0757, 0.3252, 0.4162
                }, new Shape(2, 3, 3));
                NDArray mask = m.create(new boolean[] {
                                false, true, false, true, true, true
                }, new Shape(3, 3));
                NDArray updates = m.create(new double[] {

                }, new Shape(0));
                NDArray y = m.zeros(new Shape(3, 3, 3), DataType.FLOAT64);
                boolean y_actual = Tensor.maskedScatter(x, mask, updates, y);
                boolean y_expect = false;
                assertEquals(y_expect, y_actual);
        }

        @Test
        public void test12() {

                NDManager m = Tensor.manager;
                NDArray x = m.create(new double[] {
                                0.4856, 0.0915, 0.6741, 0.1524, 0.2569, 0.1704, 0.7138, 0.8388
                }, new Shape(4, 2));
                NDArray mask = m.create(new boolean[] {
                                true, false, false, true
                }, new Shape(4));
                NDArray updates = m.create(new double[] {
                                0.7032
                }, new Shape(1));
                NDArray y = m.zeros(new Shape(4, 2), DataType.FLOAT64);
                boolean y_actual = Tensor.maskedScatter(x, mask, updates, y);
                boolean y_expect = false;
                assertEquals(y_expect, y_actual);
        }

        @Test
        public void test13() {

                NDManager m = Tensor.manager;
                NDArray x = m.create(new double[] {
                                0.6495, 0.6767, 0.4808, 0.1649, 0.0631, 0.2978, 0.9633, 0.8855, 0.7828, 0.3408, 0.6916,
                                0.7394, 0.2692, 0.2380, 0.4242, 0.7867, 0.8395, 0.0490, 0.4515, 0.2480, 0.0547, 0.9206,
                                0.3412, 0.8954, 0.6733, 0.8538, 0.7629, 0.2029, 0.3240, 0.3082, 0.7992, 0.0384, 0.5568,
                                0.6757, 0.6496, 0.1767, 0.4260, 0.1549, 0.1332, 0.4176, 0.2637, 0.0567, 0.8958, 0.5872,
                                0.7985, 0.0730, 0.7293, 0.2069
                }, new Shape(3, 2, 4, 2));
                NDArray mask = m.create(new boolean[] {
                                true, true, true, true, true, false, true, false, false, true, true, false, true, false,
                                false, true, false, true, true, true, false, true, false, true, true, false, true,
                                false, false, true, true, true, true, true, false, true
                }, new Shape(3, 2, 3, 2));
                NDArray updates = m.create(new double[] {
                                0.6539, 0.1093, 0.4684, 0.4765, 0.9133, 0.5705, 0.1426, 0.9548, 0.9541, 0.0853, 0.4931,
                                0.7986, 0.9967, 0.4560, 0.2233, 0.0678, 0.8500, 0.7899, 0.9098, 0.8363, 0.3832, 0.7080,
                                0.5027, 0.3143, 0.2818, 0.7033, 0.3754, 0.3260, 0.7423, 0.6807, 0.8288, 0.4202, 0.2630,
                                0.2106, 0.4571, 0.3983
                }, new Shape(3, 2, 2, 2));
                NDArray y = m.zeros(new Shape(3, 2, 4, 2), DataType.FLOAT64);
                boolean y_actual = Tensor.maskedScatter(x, mask, updates, y);
                boolean y_expect = false;
                assertEquals(y_expect, y_actual);
        }

        @Test
        public void test14() {

                NDManager m = Tensor.manager;
                NDArray x = m.create(new double[] {
                                0.6566, 0.9658, 0.1880, 0.3040, 0.4122, 0.8061, 0.7071, 0.5428, 0.5592, 0.0195, 0.3712,
                                0.8349, 0.2386, 0.0897, 0.5090, 0.3540, 0.9473, 0.9718, 0.4182, 0.2200, 0.7893, 0.4639,
                                0.8746, 0.0942, 0.3142, 0.8154, 0.9331, 0.9016, 0.9763, 0.4775, 0.3014, 0.8364, 0.7418,
                                0.4025, 0.0662, 0.3070
                }, new Shape(3, 2, 3, 2));
                NDArray mask = m.create(new boolean[] {
                                true, true, false, false, true, false, false, true, true, true, false, false
                }, new Shape(3, 2, 2));
                NDArray updates = m.create(new double[] {
                                0.6584, 0.8701, 0.0010, 0.5224, 0.0782, 0.9631, 0.4169, 0.9348, 0.6293, 0.6423, 0.2899,
                                0.8591, 0.3999, 0.1789, 0.6996, 0.8781, 0.6789, 0.0931, 0.4239, 0.1425, 0.0600, 0.9678,
                                0.1224, 0.9474, 0.3359, 0.8902, 0.6382, 0.9520, 0.4656, 0.6753, 0.9499, 0.4471, 0.3944,
                                0.5304, 0.8057, 0.5795
                }, new Shape(3, 2, 3, 2));
                NDArray y = m.zeros(new Shape(3, 2, 3, 2), DataType.FLOAT64);
                boolean y_actual = Tensor.maskedScatter(x, mask, updates, y);
                boolean y_expect = false;
                assertEquals(y_expect, y_actual);
        }

        @Test
        public void test15() {

                NDManager m = Tensor.manager;
                NDArray x = m.create(new double[] {
                        0.4795,0.0825,0.4187,0.3569,0.0555,0.8605,0.9198,0.2881,0.1400
                }, new Shape(3, 3));
                NDArray mask = m.create(new boolean[] {
                        false,false,false,false,false,false,false,false,false
                }, new Shape(3, 3));
                NDArray updates = m.create(new double[] {

                }, new Shape(0));
                NDArray y_actual = m.zeros(new Shape(3, 3), DataType.FLOAT64);
                Tensor.maskedScatter(x, mask, updates, y_actual);
                NDArray y_expect = m.create(new double[] {
                        0.4795,0.0825,0.4187,0.3569,0.0555,0.8605,0.9198,0.2881,0.1400
                }, new Shape(3, 3));
                assertEquals(y_expect, y_actual);
        }
        
        @Test
        public void test16() {

                NDManager m = Tensor.manager;
                NDArray x = m.create(new double[] {
                        0.4967,0.5012,0.1881,0.6658,0.1970,0.4720,0.2571,0.9228,0.8097,0.1562,0.6157,0.6774,0.8607,0.6389,0.5211,0.1220,0.3829,0.4912,0.7483,0.9649,0.6497,0.1965,0.4376,0.5142,0.0501,0.1484,0.2252,0.9108,0.5138,0.0024,0.3130,0.5436,0.3899,0.6625,0.6343,0.1776,0.1882,0.6833,0.7713,0.8264,0.0879,0.3703,0.7812,0.8992,0.6192,0.3575,0.5860,0.6555,0.4259,0.9462,0.0839,0.4971,0.5146,0.9832,0.6370,0.1291,0.8948,0.5625,0.0925,0.1219,0.0201,0.3302,0.9931,0.9044
                }, new Shape(2, 2, 2, 2, 2));
                NDArray mask = m.create(new boolean[] {
                        false,true,false,true,false,false,false,true,false,true,true,true,true,true,true,true
                }, new Shape(2, 2, 2, 2));
                NDArray updates = m.create(new double[] {
                        0.4980,0.0934,0.8205,0.9467,0.6039,0.1497,0.5869,0.6721,0.8643,0.5295,0.5525,0.6962,0.3194,0.7082,0.0026,0.5296,0.8408,0.3633,0.5305,0.0157,0.5269,0.8107,0.8527,0.5112,0.2892,0.8733,0.3292,0.9500,0.7833,0.7119,0.5952,0.5742,0.3419,0.6509,0.2094,0.6118,0.0666,0.5286,0.7277,0.6832,0.7336,0.6047,0.1055,0.2171,0.6703,0.5557,0.8376,0.6580,0.2590,0.5892,0.7394,0.7452,0.2232,0.2083,0.7781,0.6461,0.3544,0.6586,0.8582,0.2237,0.8336,0.5872,0.9688,0.2204
                }, new Shape(2, 2, 2, 2, 2));
                NDArray y_actual = m.zeros(new Shape(2, 2, 2, 2, 2), DataType.FLOAT64);
                Tensor.maskedScatter(x, mask, updates, y_actual);
                NDArray y_expect = m.create(new double[] {
                        0.4967,0.4980,0.1881,0.0934,0.1970,0.4720,0.2571,0.8205,0.8097,0.9467,0.6039,0.1497,0.5869,0.6721,0.8643,0.5295,0.3829,0.5525,0.7483,0.6962,0.6497,0.1965,0.4376,0.3194,0.0501,0.7082,0.0026,0.5296,0.8408,0.3633,0.5305,0.0157,0.3899,0.5269,0.6343,0.8107,0.1882,0.6833,0.7713,0.8527,0.0879,0.5112,0.2892,0.8733,0.3292,0.9500,0.7833,0.7119,0.4259,0.5952,0.0839,0.5742,0.5146,0.9832,0.6370,0.3419,0.8948,0.6509,0.2094,0.6118,0.0666,0.5286,0.7277,0.6832
                }, new Shape(2, 2, 2, 2, 2));
                assertEquals(y_expect, y_actual);
        }
}
