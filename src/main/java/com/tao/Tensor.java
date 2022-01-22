package com.tao;

import ai.djl.ndarray.*;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.Shape;

/**
 * @author Yu Tao
 * @version 1.0
 * @description 基于DJL(Deep Java Library)中的NDarray创建功能，实现张量的广播功能和掩码操作
 * @date 2021.12.28
 */

public class Tensor 
{
    /** 
     * manager是DJL库创建张量所必须创建的对象，
     * 使用完后记得使用manager.close()以防造成内存的损失
     */
    static NDManager manager = NDManager.newBaseManager();
    static int loc = 0;

    /**
     * 将张量转化为字符串形
     * 
     * @param nd 张量
     * @return 张量的字符串形式
     */
    /*
    public static String getString(NDArray nd) {
        String str = nd.toDebugString(1000, 1000, 1000, 1000);
        int first = str.indexOf("[");
        str = str.substring(first);
        return str;
    }
    */

    /**
     * 按照给定的尺寸，对张量进行广播
     * 
     * 例1：
     * source = [1, 0, 1, 0], shape = new Shape(4, 4)
     * return: [[1, 0, 1, 0], [1, 0, 1, 0], [1, 0, 1, 0], [1, 0, 1, 0]]
     * 
     * 例2：
     * source = [[1], [0], [1], [0]], shape = new Shape(4, 4)
     * return: [[1, 1, 1, 1], [0, 0, 0, 0], [1, 1, 1, 1], [0, 0, 0, 0]]
     * 
     * 例3：
     * source = [1, 2, 3] , shape = new Shape(4, 4)
     * return: null ，因为尺寸为Shape(3)，与Shape(4, 4)不能进行匹配
     * 
     * @param source 需要进行广播操作的的源张量
     * @param targetShape 源张量经广播后的尺寸
     * @return 如果是null，这说明无法进行张量广播操作；如果不是null,返回的是经广播后的张量
     */
    public static NDArray broadcast(NDArray source, Shape targetShape) {
        Shape sourceShape = source.getShape();
        int sourceDim = sourceShape.dimension();
        int targetDim = targetShape.dimension();

        /* 源张量的维度大小超过目标维度大小，无法进行张量广播操作 */
        if (sourceDim > targetDim) {
            return null;
        }

        /* 源张量与目标尺寸的后缘维度的轴长度不相符，而且没有轴长为1的情况，无法进行张量广播的操作 */
        for (int i = sourceDim - 1, j = targetDim -1 ; i >= 0; i--, j--) {
            if (sourceShape.get(i) != targetShape.get(j) && sourceShape.get(i) != 1) {
                return null;
            }
        }

        /* 源张量与目标尺寸的后缘维度的轴长度相符，或者出现轴长为1的情况，进行张量广播的操作 */
        NDArray target = manager.zeros(targetShape, source.getDataType());       

        broadcastEnsured(source, target);

        return target;
        
    }

    /**
     * 在已经确保源张量能进行广播的情况下，将源张量扩展成目标张量
     * @param source 源张量
     * @param target 目标张量
     */
    public static void broadcastEnsured(NDArray source, NDArray target) {
        Shape sourceShape = source.getShape();
        Shape targetShape = target.getShape();
        int sourceDim = sourceShape.dimension();
        int targetDim = targetShape.dimension();

        /**
         * 当源张量source维度大小与目标张量target维度大小不一致时，
         * 将源张量广播至对目标张量在0轴中的子张量
         */
        if (sourceDim != targetDim) {
            NDArray temp = manager.zeros(targetShape.slice(1), source.getDataType());
            broadcastEnsured(source, temp);
            for (int i = 0; i < targetShape.get(0); i++) {
                target.set(new NDIndex("" + i), temp);
            }
        } 
        /**
         * 当源张量source维度大小与目标张量target维度大小均为1时:
         * 1)如果source和target的轴长相等，
         * 此时直接将target取值为source即可；
         * 2)如果source和target的轴长不相等，也就意味着source的轴长为1时
         * 此时将target的每项都取值为source即可。
         */
        else if (sourceDim == 1) {
            if (sourceShape.get(0) == targetShape.get(0)) {
                target.set(new NDIndex("..."), source);
            } else {
                for (int i = 0; i < targetShape.get(0); i++) {
                    target.set(new NDIndex("" + i), source);
                }
            }
            return;
        } 
        /**
         * 当源张量source维度大小和目标张量target维度大小相等，而且不为1时：
         * 1)如果source的第0轴的长度与target的第0轴的长度相等时，
         * 此时将source从第0轴中的子张量广播到target在第0轴中的对应的子张量；
         * 2)如果source的第0轴的长度与target的第0轴的长度不相等，这就意味着source的第0轴的轴长为1时，
         * 此时将source广播到target在第0轴中的所有子张量
         */
        else {
            if (sourceShape.get(0) == targetShape.get(0)) {
                for (int i = 0; i < targetShape.get(0); i++) {
                    NDIndex index = new NDIndex("" + i);
                    NDArray temp = manager.zeros(targetShape.slice(1), source.getDataType());
                    broadcastEnsured(source.get(index), temp);
                    target.set(index, temp);
                }
            } else {
                NDArray temp = manager.zeros(targetShape.slice(1), source.getDataType());
                broadcastEnsured(source.get(0), temp);
                for (int i = 0; i < targetShape.get(0); i++) {
                    target.set( new NDIndex("" + i), temp);
                }
            }
        }
    }

    /**
     * 在给定掩码张量和更新张量的前提下，对源张量进行掩码操作
     * 
     * 例1：
     * x = [[0, 0], [0, 0]] 
     * mask = [[false true], [true, false]]
     * updates = [[1, 2],[3, 4]]
     * y = [[0, 1], [2, 0]]
     * return true
     * 
     * 例2：
     * x = [[0, 0], [0, 0]]
     * mask = [[false], [true]]
     * updates = [[1, 2], [3, 4]]
     * y = [[0, 0]. [1, 2]]
     * return true
     * 
     * 例3：
     * x = [[0, 0], [0, 0]]
     * mask = [false, false, true]
     * updates = [[1, 2], [3, 4]]
     * y没有变化
     * return false
     * 
     * @param x 源张量
     * @param mask 掩码张量
     * @param updates 更新张量
     * @param y 经过掩码操作后更新的张量
     * @return 如果是ture,代表能够进行掩码操作；如果是false，代表无法进行掩码操作
     */
    public static boolean maskedScatter(NDArray x, NDArray mask, NDArray updates, NDArray y) {
        /* 如果四者均为空，其掩码为null */
        if (x == null && mask == null && updates ==null && y == null) {
            return true;
        }
        
        /* 输入值中有null，无法进行掩码操作 */
        if (x == null || mask == null || updates ==null || y == null) {
            return false;
        }

        /* 如果源张量和更新后的张量的维度不一致，依然无法进行掩码操作 */
        if (!(x.getShape().equals(y.getShape()))) {
            System.out.println("源张量、更新后的张量维度不相同");
            return false;        
        }

        /* 如果掩码张量不符合广播的要求，则无法进行掩码操作 */
        mask = broadcast(mask, x.getShape());
        if (mask == null) {
            System.out.println("mask维度不满足要求");
            return false;
        }
        
        /* 对更新张量进行扁平化处理（降成一维），方便进行掩码操作 */
        NDArray flatUpdates = updates.flatten();
        long numberOfUpdates = flatUpdates.getShape().get(0);
        long trueNumber = mask.flatten().sum().getLong();

        /* 如果更新张量无法提供足量的值来替换源张量，无法进行掩码操作 */
        if (numberOfUpdates < trueNumber) {
            System.out.println("更新张量无法提供足量的值来替换源张量");
            return false;
        }
        
        loc = 0;
        maskedScatterByFlatUpdates(x, mask, flatUpdates, y);
        return true;
    }

    /**
     * 在给定掩码张量和扁平化后更新张量的前提下，对源张量进行掩码操作
     * 
     * @param x 源张量
     * @param mask 掩码张量
     * @param updates 经过扁平化后的更新张量
     * @param y 进过掩码操作后更新的张量
     */
    public static void maskedScatterByFlatUpdates(NDArray x, NDArray mask, NDArray updates, NDArray y) {
        int dim = x.getShape().dimension();
        long length = x.getShape().get(0);

        /**
         * 1)如果维度为1时，
         * mask中取值为true的位置，y中对应的位置取值为updates中的值；
         * 2)如果维度不为1时，
         * 将所有张量减小1个维度后的张量作为子张量，
         * 对子张量进行掩码操作。
         */
        for (long i = 0; i < length; i++) {
            NDIndex index = new NDIndex("" + i);
            if (dim == 1) {
                if (mask.getBoolean(i) == true) {
                    y.set(index, updates.get(loc));
                    loc ++;
                } else {
                    y.set(index, x.get(i));
                }
            } else {
                NDArray yDec = y.get(index);
                maskedScatterByFlatUpdates(x.get(index), mask.get(index), updates, yDec);
                y.set(index, yDec);
            }
        }
    }
}
