import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import fem_utils


def example(n=16, num_refine=3):
    """
    -\Delta u = f, \boldsymbol{x} \in \Omega
    u = g, \boldsymbol{x} \in \partial \Omega

    check items:
    * IsotropicMesh.inner_node_ids
    * IsotropicMesh.bound_node_ids
    * FiniteElement.gram_p1
    * FiniteElement.integer_p1
    * LinearSystem.node_mul_node
    * LinearSystem.node_mul_func
    """
    w_1 = 0.25
    w_2 = 0.75

    def func_u(x):
        return np.cos(w_1 * x[0] + w_2 * x[1])

    def func_f(x):
        return -(w_1 ** 2 + w_2 ** 2) * np.cos(w_1 * x[0] + w_2 * x[1])

    func_g = func_u

    # Start to solve
    mesh = fem_utils.SquareMesh(n=n)

    inner_nn = mesh.inner_node_ids.__len__()
    bound_nn = mesh.bound_node_ids.__len__()

    # compute mat of (\nabla \phi_{\boldsymbol{u}}, \nabla \phi_{\boldsymbol{u}})
    # P_1 X P_1
    gram_tensor = mesh.gram_grad_p1()
    mat = mesh.node_mul_node(gram_tensor)

    mat_1 = mat[mesh.inner_node_ids, mesh.inner_node_ids]
    mat_1 = coo_matrix((mat_1.data, (mat_1.idx[0], mat_1.idx[1])), shape=(inner_nn, inner_nn))

    mat_2 = mat[mesh.inner_node_ids, mesh.bound_node_ids]
    mat_2 = coo_matrix((mat_2.data, (mat_2.idx[0], mat_2.idx[1])), shape=(inner_nn, bound_nn))

    bound_vertices = mesh.vertices[mesh.bound_node_ids]
    rhs_1 = func_g(bound_vertices.T)

    # compute mat of (\phi_p, f)
    integer_tensor = mesh.integer_p1(func_f, num_refine=num_refine)
    rhs = mesh.node_mul_func(integer_tensor)
    rhs_2 = rhs[mesh.inner_node_ids]

    coeff = spsolve(mat_1, rhs_2 - mat_2@rhs_1)

    # Check L2 error
    error = mesh.p1_error()
    print(error)

    # Show figure
    fig = plt.figure(figsize=(10, 6))

    xx = mesh.vertices[:, 0]
    yy = mesh.vertices[:, 1]
    u_val = func_u(mesh.vertices.T)
    u_h = func_g(mesh.vertices.T)
    u_h[mesh.inner_node_ids] = coeff

    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.set_title("z = u_val(x, y)\nz = u_h(x, y)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("u[0]")
    ax.plot_trisurf(xx, yy, u_val, alpha=0.5)
    ax.plot_trisurf(xx, yy, u_h, alpha=0.5)

    plt.show()


def data_generate(w, n=16, num_refine=3):
    """
    -\Delta u = f, \boldsymbol{x} \in \Omega
    u = g, \boldsymbol{x} \in \partial \Omega
    """
    def func_u(x):
        return np.cos(w * x[0] + (1 - w) * x[1])

    def func_f(x):
        return -(w ** 2 + (1 - w) ** 2) * np.cos(w * x[0] + (1 - w) * x[1])

    func_g = func_u

    # Start to solve
    mesh = fem_utils.SquareMesh(n=n)

    inner_nn = mesh.inner_node_ids.__len__()
    bound_nn = mesh.bound_node_ids.__len__()

    # compute mat of (\nabla \phi_{\boldsymbol{u}}, \nabla \phi_{\boldsymbol{u}})
    # P_1 X P_1
    gram_tensor = mesh.gram_grad_p1()
    mat = mesh.node_mul_node(gram_tensor)

    mat_1 = mat[mesh.inner_node_ids, mesh.inner_node_ids]
    mat_1 = coo_matrix((mat_1.data, (mat_1.idx[0], mat_1.idx[1])), shape=(inner_nn, inner_nn))

    mat_2 = mat[mesh.inner_node_ids, mesh.bound_node_ids]
    mat_2 = coo_matrix((mat_2.data, (mat_2.idx[0], mat_2.idx[1])), shape=(inner_nn, bound_nn))

    bound_vertices = mesh.vertices[mesh.bound_node_ids]
    rhs_1 = func_g(bound_vertices.T)

    # compute mat of (\phi_p, f)
    integer_tensor = mesh.integer_p1(func_f, num_refine=num_refine)
    rhs = mesh.node_mul_func(integer_tensor)
    rhs_2 = rhs[mesh.inner_node_ids]

    mat = mat_1
    rhs = rhs_2 - mat_2@rhs_1
    numerical_roots = spsolve(mat, rhs)
    genuine_roots = func_u(mesh.vertices[mesh.inner_node_ids].T)

    return mat, rhs, numerical_roots, genuine_roots


# example()
data_generate(0.1)
