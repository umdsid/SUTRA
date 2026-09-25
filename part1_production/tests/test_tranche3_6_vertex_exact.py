import numpy as np
from strata.mechanics.vertex_exact import exact_area_derivative,exact_length_derivatives


def test_area_derivative_square_vertex():
    # Square vertices (0,0),(1,0),(1,1),(0,1), derivative at (1,0).
    prev=np.array([0.0,0.0])
    nxt=np.array([1.0,1.0])
    d=exact_area_derivative(prev,nxt)
    assert np.allclose(d,[0.5,-0.5])


def test_length_derivative():
    g0,g1,L=exact_length_derivatives([0,0],[3,4])
    assert abs(L-5.0)<1e-12
    assert np.allclose(g0,[-0.6,-0.8])
    assert np.allclose(g1,[0.6,0.8])


def test_translation_invariance_of_area_gradient_sum():
    # Sum of area gradients around a polygon should be zero.
    xy=np.array([[0.,0.],[2.,0.],[2.,1.],[0.,1.]])
    grads=[]
    for k in range(len(xy)):
        grads.append(exact_area_derivative(xy[(k-1)%4],xy[(k+1)%4]))
    assert np.allclose(np.sum(grads,axis=0),[0.,0.])
