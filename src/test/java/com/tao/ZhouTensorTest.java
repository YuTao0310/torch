package com.tao;

import org.junit.Test;
import static org.junit.Assert.*;

import ai.djl.ndarray.*;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;

public class ZhouTensorTest {

    /** 正常输入 */
    /**
     * 3*3
     */
    @Test
    public void test33_1() {
        NDManager m = Tensor.manager;
        NDArray x = m.create(new double[] {
                4.71714,1.74288,8.46838,
                7.41908,2.24844,1.96710,
                3.33810,8.22865,9.53787}, new Shape(3,3));
        NDArray mask = m.create(new boolean[] {
                true,false,false,
                true,false,false,
                true,false,false}, new Shape(3,3));
        NDArray updates = m.create(new double[] {
                2.45822,9.80655,8.47867,
                2.15475,4.88823,4.89831,
                5.06043,4.94170,9.50609}, new Shape(3, 3));
        NDArray y_actual = m.zeros(new Shape(3, 3), DataType.FLOAT64);
        Tensor.maskedScatter(x, mask, updates, y_actual);
        NDArray y_expect = m.create(new double[] {
                2.45822,1.74288,8.46838,
                9.80655,2.24844,1.96710,
                8.47867,8.22865,9.53787}, new Shape(3, 3));
        assertEquals(y_expect, y_actual);
    }

    @Test
    public void test33_2() {
        NDManager m = Tensor.manager;
        NDArray x = m.create(new double[] {
                4.71714,1.74288,8.46838,
                7.41908,2.24844,1.96710,
                3.33810,8.22865,9.53787}, new Shape(3,3));
        NDArray mask = m.create(new boolean[] {true, false, false}, new Shape(3, 1));
        NDArray updates = m.create(new double[] {
                2.45822,9.80655,8.47867,
                2.15475,4.88823,4.89831,
                5.06043,4.94170,9.50609}, new Shape(3, 3));
        NDArray y_actual = m.zeros(new Shape(3, 3), DataType.FLOAT64);
        Tensor.maskedScatter(x, mask, updates, y_actual);
        NDArray y_expect = m.create(new double[] {
                2.45822,9.80655,8.47867,
                7.41908,2.24844,1.96710,
                3.33810,8.22865,9.53787}, new Shape(3, 3));
        assertEquals(y_expect, y_actual);
    }

    @Test
    public void test33_3() {
        NDManager m = Tensor.manager;
        NDArray x = m.create(new double[] {

        }, new Shape(0));
        NDArray mask = m.create(new boolean[] {
            
        }, new Shape(0));
        NDArray updates = m.create(new double[] {

        }, new Shape(0));
        NDArray y = m.zeros(new Shape(0), DataType.FLOAT64);
        boolean y_actual = Tensor.maskedScatter(x, mask, updates, y);
        
        boolean y_expect = false;
        assertEquals(y_expect, y_actual);
    }

    @Test
    public void test33_4() {
        NDManager m = Tensor.manager;
        NDArray x = m.create(new double[] {
                4.71714,1.74288,8.46838,
                7.41908,2.24844,1.96710,
                3.33810,8.22865,9.53787}, new Shape(3,3));
        NDArray mask = m.create(new boolean[] {true, false, false, false}, new Shape(4));
        NDArray updates = m.create(new double[] {
                2.45822,9.80655,8.47867,
                2.15475,4.88823,4.89831,
                5.06043,4.94170,9.50609}, new Shape(3, 3));
        NDArray y = m.zeros(new Shape(3, 3), DataType.FLOAT64);
        boolean y_actual = Tensor.maskedScatter(x, mask, updates, y);
        boolean y_expect = false;
        Tensor.maskedScatter(x, mask, updates, y);
        assertEquals(y_expect, y_actual);
    }

    @Test
    public void test33_5() {
        NDManager m = Tensor.manager;
        NDArray x = m.create(new double[] {
                4.71714,1.74288,8.46838,
                7.41908,2.24844,1.96710,
                3.33810,8.22865,9.53787}, new Shape(3,3));
        NDArray mask = m.create(new boolean[] {true, false, false}, new Shape(3));
        NDArray updates = m.create(new double[] {2.45822,9.80655}, new Shape(2));
        NDArray y = m.zeros(new Shape(3, 3), DataType.FLOAT64);
        boolean y_actual = Tensor.maskedScatter(x, mask, updates, y);
        boolean y_expect = false;
        Tensor.maskedScatter(x, mask, updates, y);
        assertEquals(y_expect, y_actual);
    }

    @Test
    public void test33_6() {
        NDManager m = Tensor.manager;
        NDArray x = m.create(new double[] {
                4.71714,1.74288,8.46838,
                7.41908,2.24844,1.96710,
                3.33810,8.22865,9.53787}, new Shape(3,3));
        NDArray mask = m.create(new boolean[] {true, false, false}, new Shape(3));
        NDArray updates = m.create(new double[] {}, new Shape(0));
        NDArray y = m.zeros(new Shape(3, 3), DataType.FLOAT64);
        boolean y_actual = Tensor.maskedScatter(x, mask, updates, y);
        boolean y_expect = false;
        Tensor.maskedScatter(x, mask, updates, y);
        assertEquals(y_expect, y_actual);
    }

    @Test
    public void test33_7() {
        NDManager m = Tensor.manager;
        NDArray x = m.create(new double[] {
                4.71714,1.74288,8.46838,
                7.41908,2.24844,1.96710,
                3.33810,8.22865,9.53787}, new Shape(3,3));
        NDArray mask = m.create(new boolean[] {true, false, false}, new Shape(3));
        NDArray updates = m.create(new double[] {2.45822,9.80655,2.15475,4.88823}, new Shape(4));
        NDArray y = m.zeros(new Shape(3, 3), DataType.FLOAT64);
        boolean y_actual = Tensor.maskedScatter(x, mask, updates, y);
        boolean y_expect = false;
        Tensor.maskedScatter(x, mask, updates, y);
        assertEquals(y_expect, y_actual);
    }

    @Test
    public void test33_8() {
        NDManager m = Tensor.manager;
        NDArray x = m.create(new double[] {
                4.71714,1.74288,8.46838,
                7.41908,2.24844,1.96710,
                3.33810,8.22865,9.53787}, new Shape(3,3));
        NDArray mask = m.create(new boolean[] {true, false, false}, new Shape(1,3));
        NDArray updates = m.create(new double[] {
                2.45822,9.80655,8.47867,
                2.15475,4.88823,4.89831,
                5.06043,4.94170,9.50609}, new Shape(3, 3));
        NDArray y_actual = m.zeros(new Shape(3, 3), DataType.FLOAT64);
        Tensor.maskedScatter(x, mask, updates, y_actual);
        NDArray y_expect = m.create(new double[] {
                2.45822,1.74288,8.46838,
                9.80655,2.24844,1.96710,
                8.47867,8.22865,9.53787}, new Shape(3, 3));
        assertEquals(y_expect, y_actual);
    }

    @Test
    public void test33_9() {
        NDManager m = Tensor.manager;
        NDArray x = m.create(new double[] {
                4.71714,1.74288,8.46838,
                7.41908,2.24844,1.96710,
                3.33810,8.22865,9.53787}, new Shape(3,3));
        NDArray mask = m.create(new boolean[] {
                true, false, false,
                true, false, false,
                true, false, false}, new Shape(3,3));
        NDArray updates = m.create(new double[] {
                2.45822,
                2.15475,
                5.06043}, new Shape(3, 1));
        NDArray y_actual = m.zeros(new Shape(3, 3), DataType.FLOAT64);
        Tensor.maskedScatter(x, mask, updates, y_actual);
        NDArray y_expect = m.create(new double[] {
                2.45822,1.74288,8.46838,
                2.15475,2.24844,1.96710,
                5.06043,8.22865,9.53787}, new Shape(3, 3));
        assertEquals(y_expect, y_actual);
    }

}