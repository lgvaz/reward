import pytest, torch
import numpy as np, reward as rw, reward.utils as U


def test_stack():
    S = rw.space.Image(shape=[2, 4, 5, 3])
    tfms = rw.tfm.img.Stack(n=3)
    ss = np.arange(4*2*4*5*1, dtype='uint8').reshape([4, 2, 4, 5, 1])
    s1, s2, s3, s4 = [S(s) for s in ss]
    s1t = s1.apply_tfms(tfms)
    s2t = s2.apply_tfms(tfms)
    s3t = s3.apply_tfms(tfms)
    s4t = s4.apply_tfms(tfms)
    s1t, s2t, s3t, s4t = [U.to_np(o.to_tensor()) for o in [s1t, s2t, s3t, s4t]]

    s1t_expect = np.array([s1.img, s1.img, s1.img]).transpose((1, 0, 2, 3, 4))[..., 0].astype('float32') / 255.
    s2t_expect = np.array([s1.img, s1.img, s2.img]).transpose((1, 0, 2, 3, 4))[..., 0].astype('float32') / 255.
    s3t_expect = np.array([s1.img, s2.img, s3.img]).transpose((1, 0, 2, 3, 4))[..., 0].astype('float32') / 255.
    s4t_expect = np.array([s2.img, s3.img, s4.img]).transpose((1, 0, 2, 3, 4))[..., 0].astype('float32') / 255.
    np.testing.assert_allclose(s1t, s1t_expect)
    np.testing.assert_allclose(s2t, s2t_expect)
    np.testing.assert_allclose(s3t, s3t_expect)
    np.testing.assert_allclose(s3t, s3t_expect)

def test_stack_mem():
    S = rw.space.Image(shape=[2, 4, 5, 3])
    tfms = rw.tfm.img.Stack(n=3)
    ss = np.arange(2*2*4*5*1, dtype='uint8').reshape([2, 2, 4, 5, 1])
    s1, s2 = [S(s) for s in ss]
    s1t = s1.apply_tfms(tfms)
    s2t = s2.apply_tfms(tfms)
    assert np.shares_memory(s1t.img.arr[0], s2t.img.arr[0])
