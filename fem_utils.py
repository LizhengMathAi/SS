from functools import reduce
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix


# ------------------------------------- Sparse Tensor -------------------------------------
class SparseTensor:
    format = 'coo'

    def __str__(self):
        string = ''
        if self.dtype == np.int:
            for i in range(self.data.__len__()):
                string += str(self.idx[:, i]) + '\t{}\n'.format(self.data[i])
        else:
            for i in range(self.data.__len__()):
                string += str(self.idx[:, i]) + '\t{:.4e}\n'.format(self.data[i])

        return string

    def __init__(self, arg1, shape=None, dtype=None, copy=False):
        """
        Examples
        --------
        # >>> data = np.arange(6)
        # >>> idx_0 = [1, 1, 1, 1, 0, 1]
        # >>> idx_1 = [3, 0, 0, 3, 3, 3]
        # >>> idx_2 = [2, 3, 3, 2, 2, 3]
        # >>> sparse_tensor = utils.SparseTensor((data, [idx_0, idx_1, idx_2]), shape=[2, 4, 4])
        [0 3 2]	4
        [1 0 3]	3
        [1 3 2]	3
        [1 3 3]	5
        """
        data, idx = arg1
        self.shape = shape

        if dtype is not None:
            self.dtype = dtype
        else:
            self.dtype = data.dtype

        self.data = np.array(data, dtype=dtype)
        self.idx = np.array(idx)

        # Flatten the tensor for prepare.
        flatten_idx = self.idx[0]
        for i in range(1, self.idx.__len__()):
            flatten_idx = flatten_idx * self.shape[i] + self.idx[i]
        flatten_idx, indices = np.unique(flatten_idx, return_inverse=True)

        # Combine self.data with the same index.
        trans_data = np.ones_like(data)
        trans_row = indices
        trans_col = np.arange(data.__len__())
        trans_shape = [flatten_idx.__len__(), self.data.__len__()]
        trans_mat = coo_matrix((trans_data, (trans_row, trans_col)), shape=trans_shape)
        self.data = trans_mat @ data

        # Select self.idx with the different indices.
        trans_mat = trans_mat.tocsr()
        trans_data = trans_mat.data[trans_mat.indptr[:-1]]
        trans_indices = trans_mat.indices[trans_mat.indptr[:-1]]
        trans_indptr = np.arange(flatten_idx.__len__() + 1)
        trans_mat = csr_matrix((trans_data, trans_indices, trans_indptr), shape=trans_shape)
        self.idx = np.array([trans_mat @ idx for idx in self.idx], dtype=np.int)

        # Delete invalid items with value zero.
        valid_indices = [i for i, value in enumerate(self.data) if value != 0]
        self.data = self.data[valid_indices]
        self.idx = self.idx[:, valid_indices]

        self._flatten_idx = None

    @property
    def flatten_idx(self):
        if self._flatten_idx is None:
            axes = [i for i in range(self.shape.__len__())]

            idx = self.idx[axes[0]]
            dim = self.shape[axes[0]]
            for i in range(1, axes.__len__()):
                idx = idx * self.shape[axes[i]] + self.idx[axes[i]]
                dim = dim * self.shape[axes[i]]

            self._flatten_idx = idx

        return self._flatten_idx

    def __getitem__(self, item):
        if isinstance(item, int):
            if self.shape.__len__() == 1:
                try:
                    index = list(self.idx[0]).index(item)
                except ValueError:
                    return 0
                return self.data[index]
            else:
                result = self.copy()

                valid_axes = list(range(1, result.shape.__len__()))
                valid_indices = [i for i in range(result.data.__len__()) if result.idx[0, i] == item]

                result.data = result.data[valid_indices]
                result.idx = result.idx[valid_axes][:, valid_indices]
                result.shape = [result.shape[axis] for axis in valid_axes]
                result._flatten_idx = None

                return result

        if isinstance(item, tuple):
            slices = [it if isinstance(it, int) else list(np.arange(self.shape[i])[it]) for i, it in enumerate(item)]
        elif isinstance(item, list):
            slices = [item]
        else:
            assert isinstance(item, np.ndarray) and item.dtype == np.int
            slices = [list(item)]

        result = self.copy()

        for i, indices in enumerate(slices):
            if isinstance(item[i], slice) and item[i] == slice(None, None, None):  # indices == `:`
                continue
            if isinstance(item, tuple) and isinstance(item[i], int):  # indices in list(range(self.shape[i]))
                continue

            trans_idx_0 = np.arange(indices.__len__())
            trans_idx_1 = np.array(indices)
            trans_data = np.ones_like(trans_idx_0)
            trans_shape = [indices.__len__(), self.shape[i]]
            trans_mat = SparseTensor((trans_data, (trans_idx_0, trans_idx_1)), shape=trans_shape)

            left_str = chr(122) + chr(97 + i)
            right_str = ''.join([chr(j) for j in range(97, 97 + self.shape.__len__())])
            result_str = ''.join([chr(j) if j != 97 + i else chr(122) for j in range(97, 97 + self.shape.__len__())])
            einsum_str = left_str + ',' + right_str + '->' + result_str

            result = self.einsum(einsum_str, trans_mat, result)

        # In the case like item == (?, ..., ?, 4, ?, ..., ?, 0, ?, ..., ?)
        if isinstance(item, tuple):
            int_item = [it for it in item if isinstance(it, int)]
            if not int_item:
                return result

            invalid_axes = [i for i, it in enumerate(item) if isinstance(it, int)]
            valid_axes = [i for i in range(result.shape.__len__()) if i not in invalid_axes]

            valid_indices = []
            for i in range(result.data.__len__()):
                flag = result.idx[invalid_axes, i] - np.array([it for it in item if isinstance(it, int)])
                if np.sum(np.abs(flag)) == 0:
                    valid_indices.append(i)

            result.data = result.data[valid_indices]
            result.idx = result.idx[valid_axes][:, valid_indices]
            result.shape = [result.shape[axis] for axis in valid_axes]
            result._flatten_idx = None

        if not result.shape:
            return 0 if result.data.__len__() == 0 else result.data[0]
        else:
            return result

    def __add__(self, other):
        assert isinstance(other, SparseTensor)
        assert self.shape == other.shape

        data = np.array(list(self.data) + list(other.data))
        idx = np.hstack([self.idx, other.idx])
        return SparseTensor((data, idx), shape=self.shape)

    def __sub__(self, other):
        assert isinstance(other, SparseTensor)
        assert self.shape == other.shape

        data = np.array(list(self.data) + list(-other.data))
        idx = np.hstack([self.idx, other.idx])
        return SparseTensor((data, idx), shape=self.shape)

    def __mul__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            return SparseTensor((self.data * other, self.idx), shape=self.shape)
        else:
            assert isinstance(other, SparseTensor)
            assert self.shape == other.shape

            valid_idx = list(set(self.flatten_idx) & set(other.flatten_idx))
            vaild_left = [list(self.flatten_idx).index(idx) for idx in valid_idx]
            vaild_right = [list(other.flatten_idx).index(idx) for idx in valid_idx]

            data = self.data[vaild_left] * other.data[vaild_right]
            idx = self.idx[:, vaild_left]

            return SparseTensor((data, idx), shape=self.shape)

    def __neg__(self):
        return SparseTensor((-self.data, self.idx), shape=self.shape)

    def copy(self):
        result = SparseTensor(([1], [[0] for _ in self.shape]), shape=self.shape, dtype=self.dtype)
        result.data = self.data
        result.idx = self.idx
        result._flatten_idx = self._flatten_idx
        return result

    @classmethod
    def stack(cls, tensors, axis):  # TODO: can be accelerated.
        """
        Examples
        --------
        # >>> data = np.arange(6)
        # >>> idx_0 = [1, 1, 1, 1, 0, 1]
        # >>> idx_1 = [3, 0, 0, 3, 3, 3]
        # >>> idx_2 = [2, 3, 3, 2, 2, 3]
        # >>> sparse_tensor = utils.SparseTensor((data, [idx_0, idx_1, idx_2]), shape=[2, 4, 4])
        [0 3 2]	4
        [1 0 3]	3
        [1 3 2]	3
        [1 3 3]	5

        # >>> SparseTensor.stack([sparse_tensor, -sparse_tensor, sparse_tensor * 2], axis=2)
        [0 3 0 2]	4
        [0 3 1 2]	-4
        [0 3 2 2]	8
        [1 0 0 3]	3
        [1 0 1 3]	-3
        [1 0 2 3]	6
        [1 3 0 2]	3
        [1 3 0 3]	5
        [1 3 1 2]	-3
        [1 3 1 3]	-5
        [1 3 2 2]	6
        [1 3 2 3]	10
        """
        for t in tensors:
            assert tensors[0].shape == t.shape

        left_axes = list(range(axis))
        right_axes = list(range(axis, tensors[0].shape.__len__()))

        left_shape = [tensors[0].shape[i] for i in range(axis)]
        right_shape = [tensors[0].shape[i] for i in range(axis, tensors[0].shape.__len__())]

        data_list, idx_list = [], []
        for i, t in enumerate(tensors):
            data_list.append(t.data)

            left_idx = [t.idx[left_axis] for left_axis in left_axes]
            right_idx = [t.idx[right_axis] for right_axis in right_axes]
            new_idx = np.array(left_idx + [i * np.ones(shape=(t.data.__len__(),))] + right_idx)
            idx_list.append(new_idx)

        data = np.hstack(data_list)
        idx = np.hstack(idx_list)
        shape = left_shape + [tensors.__len__()] + right_shape

        return SparseTensor((data, idx), shape=shape)

    def flatten(self, axes=None):
        """
        Examples
        --------
        # >>> data = np.arange(6)
        # >>> idx_0 = [1, 1, 1, 1, 0, 1]
        # >>> idx_1 = [3, 0, 0, 3, 3, 3]
        # >>> idx_2 = [2, 3, 3, 2, 2, 3]
        # >>> sparse_tensor = utils.SparseTensor((data, [idx_0, idx_1, idx_2]), shape=[2, 4, 4])
        [0 3 2]	4
        [1 0 3]	3
        [1 3 2]	3
        [1 3 3]	5

        # >>> sparse_tensor.flatten()
        [14] 4
        [19] 3
        [30] 3
        [31] 5

        # Flat the axes to [0, 1, 2] to [new axis, 0].
        # >>> sparse_tensor.flatten([1, 2])
        [3  1]	3
        [14 0]	4
        [14 1]	3
        [15 1]	5
        """
        if axes is None:
            axes = [i for i in range(self.shape.__len__())]

        idx = self.idx[axes[0]]
        dim = self.shape[axes[0]]
        for i in range(1, axes.__len__()):
            idx = idx * self.shape[axes[i]] + self.idx[axes[i]]
            dim = dim * self.shape[axes[i]]

        res_axes = [i for i in range(self.shape.__len__()) if i not in axes]
        idx = [idx] + list(self.idx[res_axes])
        shape = [dim] + list([self.shape[ax] for ax in res_axes])

        # Skip the stage of merging the `result.data` and sorting the `result.idx`.
        result = self.copy()
        result.idx = idx
        result.shape = shape
        return result

    def reshape(self, shape):
        new_dim = -reduce(lambda x, y: x * y, self.shape) // reduce(lambda x, y: x * y, shape)
        shape = [dim if dim != -1 else new_dim for dim in shape]
        assert reduce(lambda x, y: x * y, self.shape) == reduce(lambda x, y: x * y, shape)

        new_idx = [None for _ in shape]

        res = self.flatten_idx
        for i in range(shape.__len__() - 1):
            new_idx[-(i + 1)] = res % shape[-(i + 1)]
            res = res // shape[-(i + 1)]
        new_idx[0] = res

        # Skip the stage of merging the `result.data` and sorting the `result.idx`.
        result = self.copy()
        result.idx = np.array(new_idx)
        result.shape = shape
        return result

    def transpose(self, axes):
        assert axes.__len__() == self.shape.__len__()
        result = self.copy()
        result.idx = self.idx[axes]
        result.shape = [self.shape[ax] for ax in axes]

        sorted_arg = np.argsort(result.flatten_idx)  # The property `flatten_idx` has been changed!
        result.idx = result.idx[:, sorted_arg]
        result.data = result.data[sorted_arg]
        result._flatten_idx = result._flatten_idx[sorted_arg]

        return result

    def sum(self, axes=None):
        """
        Examples
        --------
        # >>> data = np.arange(6)
        # >>> idx_0 = [1, 1, 1, 1, 0, 1]
        # >>> idx_1 = [3, 0, 0, 3, 3, 3]
        # >>> idx_2 = [2, 3, 3, 2, 2, 3]
        # >>> sparse_tensor = utils.SparseTensor((data, [idx_0, idx_1, idx_2]), shape=[2, 4, 4])
        [0 3 2]	4
        [1 0 3]	3
        [1 3 2]	3
        [1 3 3]	5

        # >>> sparse_tensor.sum()
        15

        # >>> sparse_tensor.sum([1, 2])
        [0]	4
        [1]	11
        """
        if axes is None:
            return sum(self.data)
        if isinstance(axes, int):
            axes = [axes]

        axes = [self.shape.__len__() + axis if axis < 0 else axis for axis in axes]
        valid_axes = [axis for axis in range(self.shape.__len__()) if axis not in axes]
        return SparseTensor((self.data, self.idx[valid_axes]), shape=[self.shape[axis] for axis in valid_axes])

    def triu(self, axes):
        """
        Examples
        --------
        # >>> data = np.arange(6)
        # >>> idx_0 = [1, 1, 1, 1, 0, 1]
        # >>> idx_1 = [3, 0, 0, 3, 3, 3]
        # >>> idx_2 = [2, 3, 3, 2, 2, 3]
        # >>> sparse_tensor = utils.SparseTensor((data, [idx_0, idx_1, idx_2]), shape=[2, 4, 4])
        [0 3 2]	4
        [1 0 3]	3
        [1 3 2]	3
        [1 3 3]	5

        # >>> sparse_tensor.triu(axes=[1, 2])
        [1 0 3]	3
        [1 3 3]	5
        """
        assert axes.__len__() == 2
        assert axes[0] != axes[1]

        valid_indices = []
        for i in range(self.flatten_idx.__len__()):
            if self.idx[axes[0], i] <= self.idx[axes[1], i]:
                valid_indices.append(i)

        # Skip the stage of merging the `result.data` and sorting the `result.idx`.
        result = self.copy()
        result.data = self.data[valid_indices]
        result.idx = self.idx[:, valid_indices]
        result._flatten_idx = None  # The property `flatten_idx` has been changed!
        return result

    def toarray(self):
        idx = self.idx[0]
        dim = self.shape[0]
        for i in range(1, self.shape.__len__()):
            idx = idx * self.shape[i] + self.idx[i]
            dim = dim * self.shape[i]

        array = np.zeros(shape=(dim,), dtype=self.dtype)
        array[idx] += self.data
        return np.reshape(array, self.shape)

    @classmethod
    def ones_like(cls, other, dtype=None):
        result = other.copy()
        if dtype is None:
            result.data = np.ones_like(other.data)
        else:
            result.data = np.ones_like(other.data, dtype=dtype)
            result.dtype = dtype

        return result

    @classmethod
    def einsum(cls, einsum_str, *operands):
        """
        Examples
        --------
        size_1 = 32
        shape_1 = [4, 5, 6, 7]
        data = np.arange(1, size_1 + 1)
        idx = [np.random.randint(0, dim, size=size_1) for dim in shape_1]
        # >>> sparse_tensor_1 = SparseTensor((data, idx), shape=shape_1)

        size_2 = 64
        shape_2 = [3, 3, 5, 7, 4]
        data = np.arange(1, size_2 + 1)
        idx = [np.random.randint(0, dim, size=size_2) for dim in shape_2]
        # >>> sparse_tensor_2 = SparseTensor((data, idx), shape=shape_2)

        # >>> sparse_tensor = sparse_einsum('ijkl,abjlc->aikb', sparse_tensor_1, sparse_tensor_2)
        """
        input_str, results_str = einsum_str.split('->')
        left_str, right_str = input_str.split(',')

        left_operand, right_operand = operands
        assert left_operand.shape.__len__() == left_str.__len__()
        assert right_operand.shape.__len__() == right_str.__len__()

        # contraction `left_operand`
        left_axes = [left_str.index(s) for s in left_str if s in results_str]
        left_res_axes = [left_str.index(s) for s in list(set(left_str) & set(right_str))]
        left_contract_axes = [i for i in range(left_str.__len__()) if i not in left_axes + left_res_axes]

        left = left_operand.transpose(left_contract_axes + left_axes + left_res_axes)
        left = left.sum(list(range(left_contract_axes.__len__())))

        # contraction `right_operand`, then update `right_str`
        right_axes = [right_str.index(s) for s in right_str if s in results_str]
        right_res_axes = [right_str.index(s) for s in list(set(left_str) & set(right_str))]
        right_contract_axes = [i for i in range(right_str.__len__()) if i not in right_axes + right_res_axes]

        right = right_operand.transpose(right_contract_axes + right_axes + right_res_axes)
        right = right.sum(list(range(right_contract_axes.__len__())))

        # update `left_str` and `right_str`
        left_str = ''.join([left_str[axis] for axis in left_axes + left_res_axes])
        right_str = ''.join([right_str[axis] for axis in right_axes + right_res_axes])

        # trans `left_operand` to coo matrix
        new_shape = [-1] + [left.shape[axis] for axis in range(left_axes.__len__(), left_str.__len__())]
        left_mat = left.reshape(new_shape)
        left_mat = left_mat.reshape([left_mat.shape[0], -1])
        left_mat = coo_matrix((left_mat.data, (left_mat.idx[0], left_mat.idx[1])), shape=left_mat.shape)

        # trans `right_operand` to coo matrix
        new_shape = [-1] + [right.shape[axis] for axis in range(right_axes.__len__(), right_str.__len__())]
        right_mat = right.reshape(new_shape)
        right_mat = right_mat.reshape([right_mat.shape[0], -1])
        right_mat = coo_matrix((right_mat.data, (right_mat.idx[0], right_mat.idx[1])), shape=right_mat.shape)

        # mat_mul
        result = (left_mat @ right_mat.T).tocoo()

        # trans `result` to sparse tensor
        tensor = SparseTensor((result.data, [result.row, result.col]), shape=result.shape)
        shape_row = [left.shape[left_str.index(s)] for s in left_str if s in results_str]
        shape_col = [right.shape[right_str.index(s)] for s in right_str if s in results_str]
        if not (shape_row + shape_col):  # the argument `einsum_str` is in format '???, ???->'
            return tensor.data[0]
        tensor = tensor.reshape(shape_row + shape_col)

        # transpose `result` to the format `result_str`
        row_str = ''.join([s for s in left_str if s in results_str])
        col_str = ''.join([s for s in right_str if s in results_str])
        return tensor.transpose([(row_str + col_str).index(s) for s in results_str])

    @classmethod
    def unit_test(cls, loop=16):
        for _ in range(loop):
            # tensor 1 in sparse format
            size_1 = 32
            shape_1 = [2, 5, 3, 7]
            data_1 = np.random.rand(size_1)
            idx_1 = np.array([np.random.randint(0, dim, size=size_1) for dim in shape_1])
            sparse_tensor_1 = SparseTensor((data_1, idx_1), shape=shape_1)

            # tensor 1 in dense format
            tensor_1 = np.zeros(shape=shape_1, dtype=np.float)
            for data, indices in zip(data_1, idx_1.T):
                tensor_1[indices[0], indices[1], indices[2], indices[3]] += data
            assert np.sum(np.abs(sparse_tensor_1.toarray() - tensor_1)) == 0

            # tensor 2 in sparse format
            size_2 = 64
            shape_2 = [3, 2, 5, 7, 4]
            data_2 = np.random.rand(size_2)
            idx_2 = np.array([np.random.randint(0, dim, size=size_2) for dim in shape_2])
            sparse_tensor_2 = SparseTensor((data_2, idx_2), shape=shape_2)

            # tensor 2 in dense format
            tensor_2 = np.zeros(shape=shape_2, dtype=np.float)
            for data, indices in zip(data_2, idx_2.T):
                tensor_2[indices[0], indices[1], indices[2], indices[3], indices[4]] += data
            assert np.sum(np.abs(sparse_tensor_2.toarray() - tensor_2)) == 0

            # tensor 3 in sparse format
            sparse_tensor_3 = sparse_tensor_2.transpose([1, 2, 0, 3, 4]).sum([-1])

            # tensor 3 in dense format
            tensor_3 = np.transpose(tensor_2, [1, 2, 0, 3, 4]).sum(-1)

            # the `einsum` of tensor 1 and tensor 2 in sparse format
            sparse_tensor = cls.einsum('ijkl,abjlc->aikb', sparse_tensor_1, sparse_tensor_2)

            # the `einsum` of tensor 1 and tensor 2 in dense format
            tensor = np.einsum('ijkl,abjlc->aikb', tensor_1, tensor_2)

            # check the operator `slice`
            check = sparse_tensor_1[0, 1, 2, 3] - tensor_1[0, 1, 2, 3]
            assert np.sum(np.abs(check)) == 0
            check = sparse_tensor_1.flatten()[1] - tensor_1.flatten()[1]
            assert np.sum(np.abs(check)) == 0
            check = sparse_tensor_1[1].toarray() - tensor_1[1]
            assert np.sum(np.abs(check)) == 0
            check = sparse_tensor_1[[1, 0, 0]].toarray() - tensor_1[[1, 0, 0]]
            assert np.sum(np.abs(check)) == 0
            check = sparse_tensor_1[:, [1, 0, 0]].toarray() - tensor_1[:, [1, 0, 0]]
            assert np.sum(np.abs(check)) == 0
            check = sparse_tensor_1[:, :, [1, 0, 0], :].toarray() - tensor_1[:, :, [1, 0, 0], :]
            assert np.sum(np.abs(check)) == 0
            check = sparse_tensor_1[[1, 0, 0], 1::2, ::-1].toarray() - tensor_1[[1, 0, 0], 1::2, ::-1]
            assert np.sum(np.abs(check)) == 0
            check = sparse_tensor_1[[1, 0, 0], 1::2, ::-1, 1:6:2].toarray() - tensor_1[[1, 0, 0], 1::2, ::-1, 1:6:2]
            assert np.sum(np.abs(check)) == 0
            check = sparse_tensor_1[[1, 0, 0], 2, ::-1, 1:6:2].toarray() - tensor_1[[1, 0, 0], 2, ::-1, 1:6:2]
            assert np.sum(np.abs(check)) == 0

            # check the operator `+`
            check = (sparse_tensor_1 + sparse_tensor_3).toarray() - (tensor_1 + tensor_3)
            assert np.sum(np.abs(check)) == 0

            # check the operator `-`(sub)
            check = (sparse_tensor_1 - sparse_tensor_3).toarray() - (tensor_1 - tensor_3)
            assert np.sum(np.abs(check)) == 0

            # check the operator `*`
            check = (sparse_tensor_1 * 2).toarray() - (tensor_1 * 2)
            assert np.sum(np.abs(check)) == 0
            check = (sparse_tensor_1 * np.pi).toarray() - (tensor_1 * np.pi)
            assert np.sum(np.abs(check)) == 0
            check = (sparse_tensor_1 * sparse_tensor_3).toarray() - (tensor_1 * tensor_3)
            assert np.sum(np.abs(check)) == 0

            # check the operator `-`(neg)
            check = (-sparse_tensor_1).toarray() - (-tensor_1)
            assert np.sum(np.abs(check)) == 0

            # check the function `copy`
            check = sparse_tensor_1.copy().toarray() - tensor_1
            assert np.sum(np.abs(check)) == 0

            # check the function `stack`
            sparse_tensor_4 = cls.stack([sparse_tensor_1, sparse_tensor_1 * 0.5, -sparse_tensor_1], axis=2)
            tensor_4 = np.stack([tensor_1, 0.5 * tensor_1, -tensor_1], axis=2)
            check = sparse_tensor_4.toarray() - tensor_4
            assert np.sum(np.abs(check)) == 0

            # check the function `flatten`
            check = sparse_tensor.flatten().toarray() - sparse_tensor.toarray().flatten()
            assert np.sum(np.abs(check)) == 0
            check = sparse_tensor_1.flatten(axes=[0, 1]).toarray() - tensor_1.reshape([-1, 3, 7])
            assert np.sum(np.abs(check)) == 0

            # check the function `reshape`
            check = sparse_tensor_1.reshape([2, 3, 5, 7]).toarray() - sparse_tensor_1.toarray().reshape([2, 3, 5, 7])
            assert np.sum(np.abs(check)) == 0
            check = sparse_tensor_1.reshape([2, -1, 7]).toarray() - sparse_tensor_1.toarray().reshape([2, -1, 7])
            assert np.sum(np.abs(check)) == 0

            # check the function `transpose` and `sum`
            check = sparse_tensor_3.toarray() - tensor_3
            assert np.sum(np.abs(check)) == 0

            # check the function `uptri`
            sparse_uptri_tensor_1 = sparse_tensor_1.triu([1, 3])
            uptri_tensor_1 = tensor_1.copy()
            for i in range(5):
                for j in range(i):
                    uptri_tensor_1[:, i, :, j] *= 0
            check = sparse_uptri_tensor_1.toarray() - uptri_tensor_1
            assert np.sum(np.abs(check)) == 0

            # check the function `einsum`
            check = sparse_tensor.toarray() - tensor
            assert np.max(np.abs(check)) < 1e-15
            sparse_tensor_5 = cls.einsum('ijkl,ijcd->klcd', sparse_tensor_1, sparse_tensor_3)
            tensor_5 = np.einsum('ijkl,ijcd->klcd', tensor_1, tensor_3)
            check = sparse_tensor_5.toarray() - tensor_5
            assert np.max(np.abs(check)) < 1e-15
            sparse_tensor_5 = cls.einsum('ij,i->j', sparse_tensor_2.reshape([-1, 4]), sparse_tensor_1.reshape([-1]))
            tensor_5 = np.einsum('ij,i->j', tensor_2.reshape([-1, 4]), tensor_1.reshape([-1]))
            check = sparse_tensor_5.toarray() - tensor_5
            assert np.max(np.abs(check)) < 1e-15
            sparse_tensor_5 = cls.einsum('ij,i->', sparse_tensor_2.reshape([-1, 4]), sparse_tensor_1.reshape([-1]))
            tensor_5 = np.einsum('ij,i->', tensor_2.reshape([-1, 4]), tensor_1.reshape([-1]))
            check = sparse_tensor_5 - tensor_5
            assert np.max(np.abs(check)) < 1e-15
            sparse_tensor_5 = cls.einsum('ijkl,ijkl->', sparse_tensor_1, sparse_tensor_3)
            tensor_5 = np.einsum('ijkl,ijkl->', tensor_1, tensor_3)
            check = sparse_tensor_5 - tensor_5
            assert np.max(np.abs(check)) < 1e-14

        print(">>> SparseTensor.unit_test(loop={})\nCompleted!".format(loop))


# ------------------------------------- Mesh -------------------------------------
class IsotropicMesh:
    _directed_graph = None  # Sparse int matrix with shape [NN, NN].
    _undirected_graph = None  # Sparse int matrix with shape [NN, NN].
    _bound_node_ids = None  # Int list with shape [?], all bound vertices indices.
    _inner_node_ids = None  # Int list with shape [?], all inner vertices indices.

    _bound_graph = None  # Sparse int matrix with shape [NN, NN].
    _inner_graph = None  # Sparse int matrix with shape [NN, NN].
    _bound_edge_ids = None  # Int list with shape [?], all bound edges indices.
    _inner_edge_ids = None  # Int list with shape [?], all inner edges indices.

    _bound_triangle_ids = None  # Int list with shape [?], all bound triangles indices.
    _inner_triangle_ids = None  # Int list with shape [?], all inner triangles indices.

    _tri_tensor = None  # Float tensor with shape [NT, 3, 2].
    _area = None  # Float tensor with shape [NT].
    _height = None  # Float tensor with shape [NT, 3].
    _outer_normal = None  # Float tensor with shape [NT, 3, 2].

    def __init__(self):
        """
        Properties:
            * `vertexices`: float matrix with shape [NN, 2], all vertexes.
            * `triangles`: Int matrix with shape [NT, 3], all the triangle units in triangle mesh.
            * `neighbors`: Int matrix with shape [NT, 3], all the neighbor triangle unit indices in triangle mesh.
        """
        self.vertices = None
        self.triangles = None
        self.neighbors = None

    @property
    def directed_graph(self):
        if self._directed_graph is None:
            nt = self.triangles.__len__()
            nn = self.vertices.__len__()

            data = np.ones(shape=(nt,), dtype=np.int)
            shape = [nn, nn]
            g_i = SparseTensor((data, self.triangles[:, [1, 2]].T), shape=shape)
            g_j = SparseTensor((data, self.triangles[:, [2, 0]].T), shape=shape)
            g_k = SparseTensor((data, self.triangles[:, [0, 1]].T), shape=shape)

            self._directed_graph = g_i + g_j + g_k
        return self._directed_graph

    @property
    def undirected_graph(self):
        if self._undirected_graph is None:
            undirected_graph = self.directed_graph + self.directed_graph.transpose([1, 0])
            undirected_graph = undirected_graph.triu([0, 1])

            self._undirected_graph = SparseTensor.ones_like(undirected_graph)
        return self._undirected_graph

    @property
    def bound_node_ids(self):
        if self._bound_node_ids is None:
            bound_graph = self.directed_graph - self.directed_graph.transpose([1, 0])
            self._bound_node_ids = list(set(bound_graph.idx[0]))
        return self._bound_node_ids

    @property
    def inner_node_ids(self):
        if self._inner_node_ids is None:
            self._inner_node_ids = [i for i in range(self.vertices.__len__()) if i not in self.bound_node_ids]
        return self._inner_node_ids

    @property
    def bound_graph(self):
        if self._bound_graph is None:
            nn = self.vertices.__len__()

            idx_0 = []
            idx_1 = []
            for node_ind, tri_ind in zip(self.triangles, self.neighbors):
                for i in range(3):
                    if tri_ind[i] == -1:
                        directed_edge = [node_ind[(i + 1) % 3], node_ind[(i + 2) % 3]]
                        idx_0.append(min(directed_edge))
                        idx_1.append(max(directed_edge))
            bound_graph = SparseTensor((np.ones_like(idx_0), (idx_0, idx_1)), shape=[nn, nn])
            self._bound_graph = SparseTensor.ones_like(bound_graph)

        return self._bound_graph

    @property
    def inner_graph(self):
        if self._inner_graph is None:
            self._inner_graph = self.undirected_graph - self.bound_graph

        return self._inner_graph

    @property
    def bound_edge_ids(self):
        return self.bound_graph.flatten_idx

    @property
    def inner_edge_ids(self):
        return self.inner_graph.flatten_idx

    @property
    def bound_triangle_ids(self):
        if self._bound_triangle_ids is None:
            self._bound_triangle_ids = [i for i, tri in enumerate(self.neighbors) if min(tri) == -1]
        return self._bound_triangle_ids

    @property
    def inner_triangle_ids(self):
        if self._inner_triangle_ids is None:
            self._inner_triangle_ids = [i for i, tri in enumerate(self.neighbors) if min(tri) != -1]
        return self._inner_triangle_ids

    @property
    def tri_tensor(self):
        """
        $\begin{bmatrix} \boldsymbol{x}_i & \boldsymbol{x}_j & \boldsymbol{x}_k \end{bmatrix}^T$.
        :return: 3-order tensor with shape [NT, 3, 2].
        """
        if self._tri_tensor is None:
            tensor = [np.array([self.vertices[vertex_ids, 0], self.vertices[vertex_ids, 1]]).T
                      for vertex_ids in self.triangles]
            self._tri_tensor = np.array(tensor)
        return self._tri_tensor

    @property
    def outer_normal(self):
        """
        $\begin{bmatrix} \boldsymbol{n}_i & \boldsymbol{n}_j & \boldsymbol{n}_k \end{bmatrix}^T$.
        :return: 3-order tensor with shape [NT, 3, 2].
        """
        if self._outer_normal is None:
            n_i = np.array([self.tri_tensor[:, 2, 1] - self.tri_tensor[:, 1, 1],
                            self.tri_tensor[:, 1, 0] - self.tri_tensor[:, 2, 0]]).T
            n_j = np.array([self.tri_tensor[:, 0, 1] - self.tri_tensor[:, 2, 1],
                            self.tri_tensor[:, 2, 0] - self.tri_tensor[:, 0, 0]]).T
            n_k = np.array([self.tri_tensor[:, 1, 1] - self.tri_tensor[:, 0, 1],
                            self.tri_tensor[:, 0, 0] - self.tri_tensor[:, 1, 0]]).T

            outer_normals = np.stack([n_i, n_j, n_k], axis=1)
            self._outer_normal = outer_normals / np.sqrt(np.sum(np.square(outer_normals), axis=2, keepdims=True))

        return self._outer_normal

    @property
    def area(self):
        """
        $\begin{bmatrix} \frac{Delta_e}{2} \end{bmatrix}$.
        :return: 1-order tensor with shape [NT].
        """
        if self._area is None:
            delta_e = []
            for vertex_ids in self.triangles:
                mat = np.array([self.vertices[vertex_ids, 0], self.vertices[vertex_ids, 1], [1., 1., 1.]])
                delta_e.append(np.linalg.det(mat))

            self._area = np.array(delta_e) / 2
        return self._area

    @property
    def height(self):
        """
        $\begin{bmatrix} \frac{Delta_e}{\| \boldsymbol{t}_{ij} \|_2} \end{bmatrix}$.
        :return: 1-order tensor with shape [NT].
        """
        if self._height is None:
            ts = self.tri_tensor[:, [1, 2, 0], :] - self.tri_tensor[:, [2, 0, 1], :]
            ts = np.sqrt(np.sum(np.square(ts), axis=2))
            self._height = 2 * self.area.reshape(-1, 1) / ts
        return self._height


class FiniteElement(IsotropicMesh):
    def build(self, *args, **kwargs):
        pass

    _grad_p1 = None  # Float tensor with shape [NT, 3, 2].

    @property
    def grad_p1(self):
        """
        $\begin{bmatrix} \nabla \lambda_i & \nabla \lambda_j & \nabla \lambda_k \end{bmatrix}^T$.
        :return: 3-order tensor with shape [NT, 3, 2].
        """
        if self._grad_p1 is None:
            # \nabla \lambda_i, \nabla \lambda_j, \nabla \lambda_k
            grad_i = np.array([self.tri_tensor[:, 1, 1] - self.tri_tensor[:, 2, 1],
                               self.tri_tensor[:, 2, 0] - self.tri_tensor[:, 1, 0]]).T
            grad_j = np.array([self.tri_tensor[:, 2, 1] - self.tri_tensor[:, 0, 1],
                               self.tri_tensor[:, 0, 0] - self.tri_tensor[:, 2, 0]]).T
            grad_k = np.array([self.tri_tensor[:, 0, 1] - self.tri_tensor[:, 1, 1],
                               self.tri_tensor[:, 1, 0] - self.tri_tensor[:, 0, 0]]).T

            outer_normals = np.stack([grad_i, grad_j, grad_k], axis=1)
            self._grad_p1 = outer_normals / (2 * self.area.reshape(-1, 1, 1))

        return self._grad_p1

    # ======== Gram matrix ========
    # -------- Gram matrix(node X node) --------
    def gram_p0(self):
        """
        :return: 3-order tensor with shape [NT, 3, 3].
        """
        return np.einsum("i,jk->ijk", self.area, np.ones((3, 3)))

    def gram_grad_p1(self):
        """
        :return: 3-order tensor with shape [NT, 3, 3].
        """
        return np.einsum("ijk,ilk,i->ijl", self.grad_p1, self.grad_p1, self.area)

    def gram_p1(self):
        """
                                                                              \alpha!\beta!gamma!
        \iint_K \lambda_i^\alpha \lambda_j^\beta \lambda_k^\gamma dxdy = ------------------------------ * 2 * area
                                                                         (\alpha + \beta + \gamma + 2)!
        :return: 3-order tensor with shape [NT, 3, 3].
        """
        return np.einsum("i,jk->ijk", self.area / 12, np.array([[2, 1, 1], [1, 2, 1], [1, 1, 2]]))

    def gram_p1_grad_p1(self):
        """
        \iint (p1, [\frac{\partial p1}{\partial x} \\ \frac{\partial p1}{\partial y}]) dx dy
        :return: 3-order tensor with shape [NT, 3, 3, 2].
        """
        return np.einsum("t,i,tjd->tijd", self.area / 3, np.ones(shape=(3,)), self.grad_p1)

    def gram_p1_div_p1(self):
        """
        :return: 3-order tensor with shape [NT, 3, 3].
        """
        return np.einsum("t,i,tjd->tij", self.area / 3, np.ones(shape=(3,)), self.grad_p1)

    def gram_p0_p1(self):
        """
                                                                              \alpha!\beta!gamma!
        \iint_K \lambda_i^\alpha \lambda_j^\beta \lambda_k^\gamma dxdy = ------------------------------ * 2 * area
                                                                         (\alpha + \beta + \gamma + 2)!
        :return: 3-order tensor with shape [NT, 3, 3].
        """
        return np.einsum("jk,i->ijk", np.ones(shape=(3, 3), dtype=np.float), self.area / 3)

    # -------- Gram matrix(surface X surface) --------
    def gram_s0(self):
        """
        :return: 1-order tensor with shape [NT].
        """
        return self.area

    # -------- Gram matrix(surface X node) --------
    def gram_s0_p1(self):
        """
                                                                              \alpha!\beta!gamma!
        \iint_K \lambda_i^\alpha \lambda_j^\beta \lambda_k^\gamma dxdy = ------------------------------ * 2 * area
                                                                         (\alpha + \beta + \gamma + 2)!
        :return: 2-order tensor with shape [NT, 3].
        """
        return np.einsum("j,i->ij", np.ones(shape=(3,), dtype=np.float), self.area / 3)

    def gram_s0_grad_p1(self):
        """
        \iint (s0, [\frac{\partial p1}{\partial x} \\ \frac{\partial p1}{\partial y}]) dx dy
        :return: 2-order tensor with shape [NT, 3, 2].
        """
        return np.einsum("i,ijd->ijd", self.area, self.grad_p1)

    # ======== Integer vector ========
    @classmethod
    def refine(cls, tri_vertices, num_refine=0, display_weight=False):
        """
        Examples
        --------
        from matplotlib import pyplot as plt

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        tri_vertices = np.array([[0, 0], [0.9, 0.1], [0.5, 0.6]])

        ax.scatter(tri_vertices[:, 0], tri_vertices[:, 1], color='k')
        ax.scatter(refine(tri_vertices, 2)[:, 0], refine(tri_vertices, 2)[:, 1], color='b')
        ax.scatter(refine(tri_vertices, 1)[:, 0], refine(tri_vertices, 1)[:, 1], color='g')
        ax.scatter(refine(tri_vertices, 0)[:, 0], refine(tri_vertices, 0)[:, 1], color='r')

        plt.show()
        """
        if num_refine == 0:
            return tri_vertices

        n = 2 ** num_refine
        coeff = []
        for q in range(n + 1):
            for p in range(n + 1 - q):
                coeff.append([1 - p / n - q / n, p / n, q / n])
        coeff = np.array(coeff)

        if display_weight:
            weights = []
            for q in range(n + 1):
                for p in range(n + 1 - q):
                    w = (q == 0) + (q == n) + (p == 0) + (p == n - q)
                    if w == 2:
                        weights.append(1)
                    elif w == 1:
                        weights.append(3)
                    else:
                        weights.append(6)
            weights = np.array(weights) / sum(weights)
            return coeff @ tri_vertices, weights
        else:
            return coeff @ tri_vertices

    @classmethod
    def estimate_integer(cls, func, tri_vertices, num_refine=0):
        check_points, weights = cls.refine(tri_vertices, num_refine=num_refine, display_weight=True)
        area = np.linalg.det(np.hstack([tri_vertices, np.ones(shape=(3, 1))])) / 2
        return np.inner(func(check_points.T), weights) * area

    # -------- Integer vector(node X function) --------
    def integer_p0(self, func, num_refine=0):
        """
        :return: sparse tensor. shape = [NT, 3]
        """
        integer = [self.estimate_integer(func, tri_vertices, num_refine) for tri_vertices in self.tri_tensor]

        return np.einsum("i,j->ij", np.array(integer), np.ones(shape=(3,), dtype=np.float))

    def integer_p1(self, func, num_refine=0):
        """
        :return: sparse tensor. shape = [NT, 3]
        """
        integer = []
        for tri_vertices, area in zip(self.tri_tensor, self.area):
            refine_coeff = self.refine(np.eye(3, dtype=np.float), num_refine=num_refine)
            check_points = refine_coeff @ tri_vertices
            func_val = func(check_points.T)

            integer.append(np.mean(func_val.reshape(-1, 1) * refine_coeff, axis=0) * area)
        return np.array(integer)

    # -------- Integer vector(surface X function) --------
    def integer_s0(self, func, num_refine=0):
        """
        :return: sparse tensor. shape = [NT]
        """
        integer = [self.estimate_integer(func, tri_vertices, num_refine) for tri_vertices in self.tri_tensor]
        return np.array(integer)


class LinearSystem(IsotropicMesh):
    # ======== left matrix ========
    def node_mul_node(self, gram_tensor):
        """
        gram_tensor: gram_p0, gram_p1, gram_grad_p1, ...
        :return: sparse tensor [(?, ?)]. shape = [NN, NN]
        """
        nn = self.vertices.__len__()

        data, idx_0, idx_1 = [], [], []
        for i in range(3):
            for j in range(3):
                data.append(gram_tensor[:, i, j])
                idx_0.append(self.triangles[:, i])
                idx_1.append(self.triangles[:, j])
        data = np.hstack(data)
        idx_0 = np.hstack(idx_0)
        idx_1 = np.hstack(idx_1)
        return SparseTensor((data, (idx_0, idx_1)), shape=[nn, nn])

    def edge_mul_edge(self, gram_tensor, directed=False):
        """
        gram_tensor: gram_rt0, gram_div_rt0, ...
        :return: sparse tensor [(?, ?)]. shape = [NN * NN, NN * NN]
        """
        nn = self.vertices.__len__()

        data = []
        idx_0, idx_1, idx_2, idx_3 = [], [], [], []
        for i in range(3):
            for j in range(3):
                data.append(gram_tensor[:, i, j])
                idx_0.append(self.triangles[:, (i + 1) % 3])
                idx_1.append(self.triangles[:, (i + 2) % 3])
                idx_2.append(self.triangles[:, (j + 1) % 3])
                idx_3.append(self.triangles[:, (j + 2) % 3])
        data = np.hstack(data)
        idx_0 = np.hstack(idx_0)
        idx_1 = np.hstack(idx_1)
        idx_2 = np.hstack(idx_2)
        idx_3 = np.hstack(idx_3)
        if not directed:
            idx_0, idx_1 = [np.minimum(idx_0, idx_1), np.maximum(idx_0, idx_1)]
            idx_2, idx_3 = [np.minimum(idx_2, idx_3), np.maximum(idx_2, idx_3)]
        sparse_tensor = SparseTensor((data, (idx_0, idx_1, idx_2, idx_3)), shape=[nn, nn, nn, nn])
        sparse_tensor = sparse_tensor.reshape(shape=[nn * nn, nn * nn])

        return sparse_tensor

    def node_mul_edge(self, gram_tensor, directed=False):
        """
        gram_tensor: gram_p0_div_rt0, gram_p1_div_rt0, gram_p1_rt0[:, :, :, 0], gram_p1_rt0[:, :, :, 1], ...
        :return: sparse tensor [(?, ?)]. shape = [NN, NN * NN]
        """
        nn = self.vertices.__len__()

        data, idx_0, idx_1, idx_2 = [], [], [], []
        for i in range(3):
            for j in range(3):
                data.append(gram_tensor[:, i, j])
                idx_0.append(self.triangles[:, i])
                idx_1.append(self.triangles[:, (j + 1) % 3])
                idx_2.append(self.triangles[:, (j + 2) % 3])
        data = np.hstack(data)
        idx_0 = np.hstack(idx_0)
        idx_1 = np.hstack(idx_1)
        idx_2 = np.hstack(idx_2)
        if not directed:
            idx_1, idx_2 = [np.minimum(idx_1, idx_2), np.maximum(idx_1, idx_2)]
        sparse_tensor = SparseTensor((data, (idx_0, idx_1, idx_2)), shape=[nn, nn, nn])
        sparse_tensor = sparse_tensor.reshape(shape=[nn, nn * nn])

        return sparse_tensor

    def surface_mul_surface(self, gram_tensor):
        """
        gram_tensor: gram_s0, ...
        :return: sparse tensor [(?, ?)]. shape = [NT, NT]
        """
        nt = self.triangles.__len__()

        data = gram_tensor
        idx_0 = np.arange(self.triangles.__len__())
        idx_1 = np.arange(self.triangles.__len__())
        idx = (idx_0, idx_1)

        return SparseTensor((data, idx), shape=[nt, nt])

    def surface_mul_node(self, gram_tensor):
        """
        gram_tensor: gram_s0_p1, gram_s0_grad_p1[0], gram_s0_grad_p1[1], ...
        :return: sparse tensor [(?, ?)]. shape = [NT, NN]
        """
        nt = self.triangles.__len__()
        nn = self.vertices.__len__()

        data, idx_0, idx_1 = [], [], []
        for i in range(3):
            data.append(gram_tensor[:, i])
            idx_0.append(np.arange(nt))
            idx_1.append(self.triangles[:, i])
        data = np.hstack(data)
        idx_0 = np.hstack(idx_0)
        idx_1 = np.hstack(idx_1)

        return SparseTensor((data, (idx_0, idx_1)), shape=[nt, nn])

    def surface_mul_edge(self, gram_tensor, directed=False):
        """
        gram_tensor: gram_s0_div_rt0, ...
        :return: sparse tensor [(?, ?)]. shape = [NT, NN * NN]
        """
        nt = self.triangles.__len__()
        nn = self.vertices.__len__()

        data, idx_0, idx_1, idx_2 = [], [], [], []
        for i in range(3):
            data.append(gram_tensor[:, i])
            idx_0.append(np.arange(nt))
            idx_1.append(self.triangles[:, (i + 1) % 3])
            idx_2.append(self.triangles[:, (i + 2) % 3])
        data = np.hstack(data)
        idx_0 = np.hstack(idx_0)
        idx_1 = np.hstack(idx_1)
        idx_2 = np.hstack(idx_2)
        if not directed:
            idx_1, idx_2 = [np.minimum(idx_1, idx_2), np.maximum(idx_1, idx_2)]
        sparse_tensor = SparseTensor((data, (idx_0, idx_1, idx_2)), shape=[nt, nn, nn])

        sparse_tensor = sparse_tensor.reshape(shape=[nt, nn * nn])

        return sparse_tensor

    def node_mul_surface(self, gram_tensor):
        """
        gram_tensor: gram_p1_s0, ...
        :return: sparse tensor [(?, ?)]. shape = [NN, NT]
        """
        sparse_tensor = self.surface_mul_node(gram_tensor)
        return sparse_tensor.transpose(axes=[1, 0])

    # ======== right vector ========
    def node_mul_func(self, integer_tensor):
        """
        integer_tensor: integer_p0, integer_p1, ...
        :return: sparse tensor [?]. shape = [NN]
        """
        nn = self.vertices.__len__()

        data, idx_0 = [], []
        for i in range(3):
            data.append(integer_tensor[:, i])
            idx_0.append(self.triangles[:, i])
        data = np.hstack(data)
        idx_0 = np.hstack(idx_0)
        return SparseTensor((data, (idx_0,)), shape=[nn, ]).toarray()

    def edge_mul_func(self, integer_tensor, directed=False):
        """
        integer_tensor: integer_rt0, integer_divrt0, ...
        :return: sparse tensor [?]. shape = [NN * NN]
        """
        nn = self.vertices.__len__()

        data, idx_0, idx_1 = [], [], []
        for i in range(3):
            data.append(integer_tensor[:, i])
            idx_0.append(self.triangles[:, (i + 1) % 3])
            idx_1.append(self.triangles[:, (i + 2) % 3])
        data = np.hstack(data)
        idx_0 = np.hstack(idx_0)
        idx_1 = np.hstack(idx_1)
        if not directed:
            idx_0, idx_1 = [np.minimum(idx_0, idx_1), np.maximum(idx_0, idx_1)]
        sparse_tensor = SparseTensor((data, (idx_0, idx_1)), shape=[nn, nn])
        sparse_tensor = sparse_tensor.reshape(shape=[-1])

        return sparse_tensor.toarray()

    def surface_mul_func(self, integer_tensor):
        """
        integer_tensor: integer_s0, ...
        :return: sparse tensor [?]. shape = [NT]
        """
        return integer_tensor

    # ======== linear system ========
    def lagrangian(self, objectives, conditions=None):
        """
        Lagrangian function:
            L(x, \lambda) = \frac{1}{2} \| A@x - b \|^2 + \lambda^T@(W@x - r)

        objectives:
            [(mat_1, rhs_1), (mat_2, rhs_2), ...]

        conditions:
            [(mat_a, rhs_a), (mat_b, rhs_b), ...]

        A: np.vstack([mat_1, mat_2, ...])
        b: np.hstack([rhs_1, rhs_2, ...])
        W: np.vstack([mat_a, mat_b, ...])
        r: np.hstack([rhs_a, rhs_b, ...])
        """
        data = np.hstack([mat_item.data for mat_item, _ in objectives])
        row_length = [mat_item.shape[0] for mat_item, _ in objectives]
        row_gap = [0] + [sum(row_length[:i]) for i in range(1, row_length.__len__())]
        row = np.hstack([mat_item.row + gap for [mat_item, _], gap in zip(objectives, row_gap)])
        col = np.hstack([mat_item.col for mat_item, _ in objectives])
        mat = coo_matrix((data, (row, col)), shape=(sum(row_length), objectives[0][0].shape[1]))
        rhs = np.hstack([rhs_item for _, rhs_item in objectives])

        print("mat:", mat.shape)
        print("rank(mat):", np.linalg.matrix_rank(mat.toarray()))

        if conditions is None:
            return (mat.T @ mat).tocoo(), mat.T @ rhs
        else:
            mat_11 = (mat.T @ mat).tocoo()
            rhs_1 = mat.T @ rhs

            data = np.hstack([mat_item.data for mat_item, _ in conditions])
            row_length = [mat_item.shape[0] for mat_item, _ in conditions]
            row_gap = [0] + [sum(row_length[:i]) for i in range(1, row_length.__len__())]
            row = np.hstack([mat_item.row + gap for [mat_item, _], gap in zip(conditions, row_gap)])
            col = np.hstack([mat_item.col for mat_item, _ in conditions])
            mat_21 = coo_matrix((data, (row, col)), shape=(sum(row_length), conditions[0][0].shape[1]))
            rhs_2 = np.hstack([rhs_item for _, rhs_item in conditions])

            mat_12 = mat_21.T

            data = np.hstack([
                mat_11.data, mat_12.data,
                mat_21.data
            ])
            row = np.hstack([
                mat_11.row, mat_12.row,
                mat_21.row + mat_11.shape[0]
            ])
            col = np.hstack([
                mat_11.col, mat_12.col + mat_11.shape[1],
                mat_21.col
            ])
            shape = (mat_11.shape[0] + mat_21.shape[0], mat_11.shape[1] + mat_12.shape[1])
            mat = coo_matrix((data, (row, col)), shape=shape)
            rhs = np.hstack([rhs_1, rhs_2])

            print("MAT:", mat.shape)
            print("rank(MAT):", np.linalg.matrix_rank(mat.toarray()))

            return mat, rhs


class SquareMesh(LinearSystem, FiniteElement):
    def __init__(self, n):
        super(SquareMesh, self).__init__()
        self.n = n
        self.rectangle_bound = [0, 0, 1, 1]
        self.h = 1 / self.n

        # generate origin vertices
        x = np.linspace(self.rectangle_bound[0], self.rectangle_bound[2], self.n + 1, endpoint=True)
        y = np.linspace(self.rectangle_bound[1], self.rectangle_bound[3], self.n + 1, endpoint=True)
        X, Y = np.meshgrid(x, y)
        self.vertices = np.vstack((X.reshape(-1), Y.reshape(-1))).T

        bottom_triangles = sum([[
            [xi + (self.n + 1) * yi, (xi + 1) + (self.n + 1) * yi, (xi + 1) + (self.n + 1) * (yi + 1)]
            for xi in range(self.n)] for yi in range(self.n)], [])
        top_triangles = sum([[
            [xi + (self.n + 1) * yi, (xi + 1) + (self.n + 1) * (yi + 1), xi + (self.n + 1) * (yi + 1)]
            for xi in range(self.n)] for yi in range(self.n)], [])
        self.triangles = np.array(bottom_triangles + top_triangles, dtype=np.int64)

        triangles_count = bottom_triangles.__len__()
        bottom_neighbors = sum([[[i + self.n * j + 1 + triangles_count if i != self.n - 1 else -1,
                                  i + self.n * j + triangles_count,
                                  i + self.n * j - self.n + triangles_count if j != 0 else -1
                                  ] for i in range(self.n)] for j in range(self.n)], [])
        top_neighbors = sum([[[i + self.n * j + self.n if j != self.n - 1 else -1,
                               i + self.n * j - 1 if i != 0 else -1,
                               i + self.n * j
                               ] for i in range(self.n)] for j in range(self.n)], [])
        self.neighbors = np.array(bottom_neighbors + top_neighbors, dtype=np.int64)

    # ======== L2 error ========
    def s0_error(self):
        pass

    def p0_error(self):
        pass

    def p1_error(self):
        pass

    def rt0(self, x, start_node, end_node):
        """broadcast function."""
        assert end_node in [start_node + 1, start_node + self.n + 1, start_node + self.n + 2]

        nn = self.vertices.__len__()

        if end_node == start_node + 1:
            supp = self.vertices[start_node, 0] <= x[0]
            supp *= x[0] <= self.vertices[end_node, 0]
            supp *= (self.vertices[end_node, 1] - self.vertices[end_node, 0]) <= (x[1] - x[0])
            supp *= (x[1] - x[0]) <= (self.vertices[start_node, 1] - self.vertices[start_node, 0])

            # top triangle
            anchor = start_node + self.n + 2
            if 0 <= anchor < nn:
                tri_supp = supp * (self.vertices[start_node, 1] <= x[1])
                tri_supp *= x[1] <= self.vertices[anchor, 1]

                rt0_x = tri_supp * (x[0] - self.vertices[anchor, 0]) * self.n
                rt0_y = tri_supp * (x[1] - self.vertices[anchor, 1]) * self.n
                rt0 = np.array([rt0_x, rt0_y])
            else:
                rt0 = np.zeros_like(x)

            # bottom triangle
            anchor = start_node - self.n - 1
            if 0 <= anchor < nn:
                tri_supp = supp * (self.vertices[anchor, 1] <= x[1])
                tri_supp *= x[1] <= self.vertices[end_node, 1]

                rt0_x = tri_supp * (x[0] - self.vertices[anchor, 0]) * self.n
                rt0_y = tri_supp * (x[1] - self.vertices[anchor, 1]) * self.n
                rt0 += np.array([rt0_x, rt0_y])

            return rt0

        if end_node == start_node + self.n + 1:
            supp = self.vertices[start_node, 1] <= x[1]
            supp *= x[1] <= self.vertices[end_node, 1]
            supp *= (self.vertices[start_node, 1] - self.vertices[start_node, 0]) <= (x[1] - x[0])
            supp *= (x[1] - x[0]) <= (self.vertices[end_node, 1] - self.vertices[end_node, 0])

            # left triangle
            anchor = start_node - 1
            if 0 <= anchor < nn:
                tri_supp = supp * (self.vertices[anchor, 0] <= x[0])
                tri_supp *= x[0] <= self.vertices[start_node, 0]

                rt0_x = tri_supp * (x[0] - self.vertices[anchor, 0]) * self.n
                rt0_y = tri_supp * (x[1] - self.vertices[anchor, 1]) * self.n
                rt0 = np.array([rt0_x, rt0_y])
            else:
                rt0 = np.zeros_like(x)

            # right triangle
            anchor = start_node + self.n + 2
            if 0 <= anchor < nn:
                tri_supp = supp * (self.vertices[end_node, 0] <= x[0])
                tri_supp *= x[0] <= self.vertices[anchor, 0]

                rt0_x = tri_supp * (x[0] - self.vertices[anchor, 0]) * self.n
                rt0_y = tri_supp * (x[1] - self.vertices[anchor, 1]) * self.n
                rt0 += np.array([rt0_x, rt0_y])

            return rt0

        if end_node == start_node + self.n + 2:
            supp = self.vertices[start_node, 0] <= x[0]
            supp *= x[0] <= self.vertices[end_node, 0]
            supp *= self.vertices[start_node, 1] <= x[1]
            supp *= x[1] <= self.vertices[end_node, 1]

            # top triangle
            anchor = start_node + self.n + 1
            tri_supp = supp * ((self.vertices[start_node, 1] - self.vertices[start_node, 0]) <= (x[1] - x[0]))
            tri_supp *= (x[1] - x[0]) <= (self.vertices[anchor, 1] - self.vertices[anchor, 0])

            rt0_x = tri_supp * (x[0] - self.vertices[anchor, 0]) * self.n * np.sqrt(2)
            rt0_y = tri_supp * (x[1] - self.vertices[anchor, 1]) * self.n * np.sqrt(2)
            rt0 = np.array([rt0_x, rt0_y])

            # bottom triangle
            anchor = start_node + 1
            tri_supp = supp * ((self.vertices[anchor, 1] - self.vertices[anchor, 0]) <= (x[1] - x[0]))
            tri_supp *= (x[1] - x[0]) <= (self.vertices[start_node, 1] - self.vertices[start_node, 0])

            rt0_x = tri_supp * (x[0] - self.vertices[anchor, 0]) * self.n * np.sqrt(2)
            rt0_y = tri_supp * (x[1] - self.vertices[anchor, 1]) * self.n * np.sqrt(2)
            rt0 += np.array([rt0_x, rt0_y])

            return rt0

    def supp_rt0(self, start_node, end_node):
        """broadcast function."""
        assert end_node in [start_node + 1, start_node + self.n + 1, start_node + self.n + 2]

        anchor_1 = None
        anchor_2 = None

        if end_node == start_node + 1:
            anchor_1 = start_node + self.n + 2
            anchor_2 = start_node - self.n - 1
        if end_node == start_node + self.n + 1:
            anchor_1 = start_node - 1
            anchor_2 = start_node + self.n + 2
        if end_node == start_node + self.n + 2:
            anchor_1 = start_node + self.n + 1
            anchor_2 = start_node + 1

        result = []
        try:
            result.append(self.vertices[[anchor_1, start_node, end_node]])
        except IndexError:
            result.append(None)
        try:
            result.append(self.vertices[[anchor_2, end_node, start_node]])
        except IndexError:
            result.append(None)

        return result

    def rt0_interpolation(self, x, coeff):
        result = np.zeros_like(x)
        if coeff.__len__() == self.inner_graph.data.__len__():
            for c, edge in zip(coeff, self.inner_graph.idx.T):
                result += c * self.rt0(x, start_node=edge[0], end_node=edge[1])
        else:
            assert coeff.__len__() == self.undirected_graph.data.__len__()
            for c, edge in zip(coeff, self.undirected_graph.idx.T):
                result += c * self.rt0(x, start_node=edge[0], end_node=edge[1])
        return result

    @classmethod
    def rt0_figure(cls, n=3):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        self = cls(n=n)
        eps = 1e-6

        ne = self.undirected_graph.flatten_idx.__len__()
        for edge_index in range(ne):
            start_node, end_node = self.undirected_graph.idx[:, edge_index]
            x_collection, y_collection, zx_collection, zy_collection = [], [], [], []
            for tri in self.triangles:
                ei = self.vertices[tri[0]]
                ej = self.vertices[tri[1]]
                ek = self.vertices[tri[2]]
                e_center = (ei + ej + ek) / 3
                ei = (1 - eps) * ei + eps * e_center
                ej = (1 - eps) * ej + eps * e_center
                ek = (1 - eps) * ek + eps * e_center

                x_collection.append([e[0] for e in [ei, ej, ek]])
                y_collection.append([e[1] for e in [ei, ej, ek]])
                zx_collection.append([self.rt0(e, start_node, end_node)[0] for e in [ei, ej, ek]])
                zy_collection.append([self.rt0(e, start_node, end_node)[1] for e in [ei, ej, ek]])

            # plot
            fig = plt.figure(figsize=(10, 6))

            ax = fig.add_subplot(1, 2, 1, projection='3d')
            ax.set_title("z = rt0[{}, {}](x, y)[0]".format(start_node, end_node))
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.scatter(self.vertices[:, 0], self.vertices[:, 1], np.zeros_like(self.vertices[:, 0]))
            for i, xy in enumerate(self.vertices):
                ax.text(xy[0], xy[1], 0, str(i))
            for x, y, zx, zy in zip(x_collection, y_collection, zx_collection, zy_collection):
                ax.plot_trisurf(x, y, zx, linewidth=0.2, antialiased=True, alpha=0.2)

            ax = fig.add_subplot(1, 2, 2, projection='3d')
            ax.set_title("z = rt0[{}, {}](x, y)[1]".format(start_node, end_node))
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.scatter(self.vertices[:, 0], self.vertices[:, 1], np.zeros_like(self.vertices[:, 0]))
            for i, xy in enumerate(self.vertices):
                ax.text(xy[0], xy[1], 0, str(i))
            for x, y, zx, zy in zip(x_collection, y_collection, zx_collection, zy_collection):
                ax.plot_trisurf(x, y, zy, linewidth=0.2, antialiased=True, alpha=0.2)

            plt.show()

    @classmethod
    def unit_test(cls):
        import matplotlib.pyplot as plt

        tri_grid = cls(n=10)

        print("-" * 32, "start to check `SquareMesh`", "-" * 32)

        # show triangle grid
        fig = plt.figure(0)
        ax = fig.add_subplot(1, 1, 1)
        plt.axis("equal")

        ax.scatter(tri_grid.vertices[:, 0], tri_grid.vertices[:, 1])
        ax.scatter(tri_grid.vertices[tri_grid.bound_node_ids, 0], tri_grid.vertices[tri_grid.bound_node_ids, 1])
        for i, xy in enumerate(tri_grid.vertices):
            ax.text(xy[0], xy[1], str(i))

        for vertex_ids in tri_grid.triangles:
            ax.fill(tri_grid.vertices[vertex_ids, 0], tri_grid.vertices[vertex_ids, 1], alpha=0.5)

        triangle_id = 156
        vertex_ids = tri_grid.triangles[triangle_id]
        ax.fill(tri_grid.vertices[vertex_ids, 0], tri_grid.vertices[vertex_ids, 1], facecolor='magenta')
        neighbor_id = tri_grid.neighbors[triangle_id]
        for i in neighbor_id:
            if i == -1:
                continue
            vertex_ids = tri_grid.triangles[i]
            ax.fill(tri_grid.vertices[vertex_ids, 0], tri_grid.vertices[vertex_ids, 1], facecolor='cyan')

        middle_point_0 = (tri_grid.tri_tensor[triangle_id, 1] + tri_grid.tri_tensor[triangle_id, 2]) / 2
        middle_point_1 = (tri_grid.tri_tensor[triangle_id, 2] + tri_grid.tri_tensor[triangle_id, 0]) / 2
        middle_point_2 = (tri_grid.tri_tensor[triangle_id, 0] + tri_grid.tri_tensor[triangle_id, 1]) / 2
        middle_point = np.array([middle_point_0, middle_point_1, middle_point_2])
        outer_normal = tri_grid.outer_normal[triangle_id]
        for i in range(3):
            ax.arrow(middle_point[i, 0], middle_point[i, 1], outer_normal[i, 0], outer_normal[i, 1])

        plt.show()
        print("-" * 32, "end of check `SquareMesh`", "-" * 32)


class MultiSquareMesh(SquareMesh):
    class DefRT0:
        def __init__(self, tri_vertices_list):
            """tri_vertices = [anchor, start_node, end_node]: np.ndarray with shape [3, 2]"""
            self.tri_vertices_list = tri_vertices_list

            self.height_list = []
            self.func_list = []
            self.inv_mat_list = []
            for tri_vertices in self.tri_vertices_list:
                mat = np.vstack([tri_vertices.T, np.ones(shape=(1, 3))])
                double_area = np.linalg.det(mat)
                assert double_area > 0

                height = double_area / np.linalg.norm(tri_vertices[1, :] - tri_vertices[2, :])
                self.height_list.append(height)

                func = lambda x: np.vstack([x[0] - tri_vertices[0, 0], x[1] - tri_vertices[0, 1]]) / height
                self.func_list.append(func)

                self.inv_mat_list.append(np.linalg.inv(mat))

        def value(self, x):
            """
            x: np.ndarray with shape [2, N]
            func(x): np.ndarray with shape [2, N]
            :return: np.ndarray with shape [2, N]
            """
            result = 0
            rhs = np.vstack([x, np.ones(shape=(1, x.shape[1]))])
            for func, inv_mat in zip(self.func_list, self.inv_mat_list):
                weights = inv_mat @ rhs
                supp = np.min(weights, axis=0) >= 0

                val = func(x)
                if val.shape.__len__() == 1:
                    result += val * supp
                else:
                    result += val * supp.reshape(1, -1)
            return result

        def div(self, x):
            """
            x: np.ndarray with shape [2, N]
            :return: np.ndarray with shape [N, ]
            """
            result = 0
            rhs = np.vstack([x, np.ones(shape=(1, x.shape[1]))])
            for height, inv_mat in zip(self.height_list, self.inv_mat_list):
                weights = inv_mat @ rhs
                supp = np.min(weights, axis=0) >= 0

                val = 2 / height * supp
                result += val * supp
            return result

    def gram_p1_def_rt0(self, def_rt0_list, num_refine=3):
        """
        :return: sparse tensor [(?, ?)]. shape = [NN, NE, 2]
        """
        ne = def_rt0_list.__len__()

        gram_mat = []
        for dim in range(2):
            for i in range(ne):
                func = lambda x: def_rt0_list[i].value(x)[dim]
                integer_tensor = self.integer_p1(func, num_refine=num_refine)
                item = self.node_mul_func(integer_tensor)
                gram_mat.append(item)

        mat_1 = np.vstack(gram_mat[:ne]).T
        mat_1 = coo_matrix(mat_1)
        tensor_1 = SparseTensor((mat_1.data, (mat_1.row, mat_1.col)), shape=mat_1.shape, dtype=mat_1.dtype)

        mat_2 = np.vstack(gram_mat[ne:]).T
        mat_2 = coo_matrix(mat_2)
        tensor_2 = SparseTensor((mat_2.data, (mat_2.row, mat_2.col)), shape=mat_2.shape, dtype=mat_2.dtype)

        tensor = SparseTensor.stack([tensor_1, tensor_2], axis=2)
        return tensor

    def gram_p1_def_div_rt0(self, def_rt0_list, num_refine=3):
        """
        :return: sparse tensor [(?, ?)]. shape = [NN, NE]
        """
        ne = def_rt0_list.__len__()

        gram_mat = []
        for i in range(ne):
            func = lambda x: def_rt0_list[i].div(x)
            integer_tensor = self.integer_p1(func, num_refine=num_refine)
            item = self.node_mul_func(integer_tensor)
            gram_mat.append(item)

        mat = np.vstack(gram_mat).T
        mat = coo_matrix(mat)
        tensor = SparseTensor((mat.data, (mat.row, mat.col)), shape=mat.shape, dtype=mat.dtype)

        return tensor

    def gram_def_rt0(self, def_rt0_list, num_refine=3):
        """
        :return: sparse tensor [(?, ?)]. shape = [NN, NE, 2]
        """
        ne = def_rt0_list.__len__()

        mat = np.zeros(shape=[ne, ne], dtype=np.float)
        for i in range(ne):
            for j in range(ne):
                inner_prod = lambda x: np.sum(def_rt0_list[i].value(x) * def_rt0_list[j].value(x), axis=0)
                integer = 0
                for tri_vertices in def_rt0_list[i].tri_vertices_list:
                    integer += self.estimate_integer(inner_prod, tri_vertices=tri_vertices, num_refine=num_refine)
                mat[i, j] = integer
        mat = coo_matrix(mat)

        return SparseTensor((mat.data, (mat.row, mat.col)), shape=mat.shape, dtype=mat.dtype)

    def integer_def_rt0(self, def_rt0_list, func, num_refine=0):
        """
        :return: np.ndarray. shape = [NE, ]
        """
        ne = def_rt0_list.__len__()

        vec = np.zeros(shape=[ne, ], dtype=np.float)
        for i in range(ne):
            inner_prod = lambda x: np.sum(def_rt0_list[i].value(x) * func(x), axis=0)
            integer = 0
            for tri_vertices in def_rt0_list[i].tri_vertices_list:
                integer += self.estimate_integer(inner_prod, tri_vertices=tri_vertices, num_refine=num_refine)
            vec[i] = integer
        return vec


# ------------------------------------- Element -------------------------------------
class RT0(FiniteElement):

    @property
    def bound_graph(self):
        if self._bound_graph is None:
            nn = self.vertices.__len__()

            idx_0 = []
            idx_1 = []
            for node_ind, tri_ind in zip(self.triangles, self.neighbors):
                for i in range(3):
                    if tri_ind[i] == -1:
                        idx_0.append(node_ind[(i + 1) % 3])
                        idx_1.append(node_ind[(i + 2) % 3])
            bound_graph = SparseTensor((np.ones_like(idx_0), (idx_0, idx_1)), shape=[nn, nn])
            self._bound_graph = SparseTensor.ones_like(bound_graph)

        return self._bound_graph

    @property
    def inner_graph(self):
        if self._inner_graph is None:
            self._inner_graph = self.directed_graph - self.bound_graph

        return self._inner_graph

    _div_rt0 = None  # Float tensor with shape [NT, 3].

    @property
    def div_rt0(self):
        """
        $\begin{bmatrix} \mathrm{div} rt0 \end{bmatrix}$.
        :return: 1-order tensor with shape [NT, 3].
        """
        if self._div_rt0 is None:
            self._div_rt0 = 2 / self.height
        return self._div_rt0

    # -------- Gram matrix(edge X edge) --------
    def gram_rt0(self, gram_p1):
        """
        :return: 3-order tensor with shape [NT, 3, 3].

        Check
        --------
        cls = utils.SquareMesh
        tri_vertices = np.array([[0, 0], [0.5, 0], [0.5, 0.5]])


        def func_1(x, ind_0, ind_1):
            rt0_i = np.array([x[0], x[1]]) / (1 / 2)
            rt0_j = np.array([x[0] - 0.5, x[1]]) / (1.414213 / 4)
            rt0_k = np.array([x[0] - 0.5, x[1] - 0.5]) / (1 / 2)
            rt0 = [rt0_i, rt0_j, rt0_k]
            return rt0[ind_0][0] * rt0[ind_1][0] + rt0[ind_0][1] * rt0[ind_1][1]


        gram_rt0 = [[cls.estimate_integer(lambda x: func_1(x, ind_0, ind_1), tri_vertices, num_refine=6)
                     for ind_1 in range(3)] for ind_0 in range(3)]
        print(np.array(gram_rt0))
        print(cls(n=2).gram_rt0[0])


        def func_2(x, ind_0, ind_1):
            rt0_i = np.array([x[0], x[1]]) / (1 / 2)
            rt0_j = np.array([x[0] - 0.5, x[1] - 0.5]) / (1 / 2)
            rt0_k = np.array([x[0], x[1] - 0.5]) / (1.414213 / 4)
            rt0 = [rt0_i, rt0_j, rt0_k]
            return rt0[ind_0][0] * rt0[ind_1][0] + rt0[ind_0][1] * rt0[ind_1][1]


        tri_vertices = np.array([[0, 0], [0.5, 0.5], [0, 0.5]])
        gram_rt0 = [[cls.estimate_integer(lambda x: func_2(x, ind_0, ind_1), tri_vertices, num_refine=6)
                     for ind_1 in range(3)] for ind_0 in range(3)]
        print(np.array(gram_rt0))
        print(cls(n=2).gram_rt0[-1])
        """
        # shape = [NT, 3, 3, 2]
        rt0_coeff = np.reshape(self.tri_tensor, [-1, 1, 3, 2]) - np.reshape(self.tri_tensor, [-1, 3, 1, 2])
        rt0_coeff = rt0_coeff / np.reshape(self.height, [-1, 3, 1, 1])

        # shape = [NT, 3, 3]
        gram = np.einsum("tipd,tjqd,tpq->tij", rt0_coeff, rt0_coeff, gram_p1)

        return gram * (np.abs(gram) > 1e-8)  # TODO: Is it necessary?

    # -------- Gram matrix(node X edge) --------
    def gram_p0_div_rt0(self):
        """
        :return: 3-order tensor with shape [NT, 3, 3].
        """
        return np.einsum("j,ik,i->ijk", np.ones((3,)), self.div_rt0, self.area)

    def gram_p1_rt0(self, gram_p1):
        """
        \iint (p1, [rt0_1 \\ rt0_2]) dx dy
        :return: 3-order tensor with shape [NT, 3, 3, 2].
        """
        # shape = [NT, 3, 3, 2]
        rt0_coeff = np.reshape(self.tri_tensor, [-1, 1, 3, 2]) - np.reshape(self.tri_tensor, [-1, 3, 1, 2])
        rt0_coeff = rt0_coeff / np.reshape(self.height, [-1, 3, 1, 1])

        return np.einsum("tik,tjkd->tijd", gram_p1, rt0_coeff)

    def gram_p1_div_rt0(self):
        """
        :return: 3-order tensor with shape [NT, 3, 3].
        """
        return np.einsum("j,ik,i->ijk", np.ones((3,)) / 3, self.div_rt0, self.area)

    # -------- Gram matrix(surface X edge) --------
    def gram_s0_div_rt0(self):
        """
        :return: 3-order tensor with shape [NT, 3].
        """
        return np.einsum("t,ti->ti", self.area, self.div_rt0)

    # -------- Integer vector(edge X function) --------
    def integer_rt0(self, func, num_refine=0):
        """
        :return: sparse tensor. shape = [NT, 3]
        """
        integer = []
        for tri_vertices, height in zip(self.tri_tensor, self.height):
            def inner_prod(x, i):
                rt0_i = np.array([x[0] - tri_vertices[i, 0], x[1] - tri_vertices[i, 1]]) / height[i]
                return np.sum(func(x) * rt0_i, axis=0)

            inner_prod_i = self.estimate_integer(lambda x: inner_prod(x, 0), tri_vertices, num_refine)
            inner_prod_j = self.estimate_integer(lambda x: inner_prod(x, 1), tri_vertices, num_refine)
            inner_prod_k = self.estimate_integer(lambda x: inner_prod(x, 2), tri_vertices, num_refine)
            integer.append([inner_prod_i, inner_prod_j, inner_prod_k])
        return np.array(integer)

    def integer_div_rt0(self, func, num_refine=0):
        """
        :return: sparse tensor. shape = [NT, 3]
        """
        integer = [self.estimate_integer(func, tri_vertices, num_refine) for tri_vertices in self.tri_tensor]
        return np.einsum('i,ij->ij', np.array(integer), self.div_rt0)

    # ======== L2 error ========
    def rt0_l2_error(self, f, coeff, num_refine=4):
        if coeff.__len__() == self.undirected_graph.data.__len__():
            sparse_coeff = self.undirected_graph.flatten().copy()
        else:
            assert coeff.__len__() == self.inner_graph.data.__len__()
            sparse_coeff = self.inner_graph.flatten().copy()
        sparse_coeff.data = np.array(coeff)
        sparse_coeff.dtype = np.float
        dense_coeff = sparse_coeff.toarray()

        nn = self.vertices.__len__()

        errors = []
        for tri_ind, triangle in enumerate(self.triangles):

            check_points = self.refine(self.vertices[triangle], num_refine=num_refine)
            f_val = f(check_points.T).T

            rt0_val = 0
            for i in range(3):
                start_node = min(triangle[(i + 1) % 3], triangle[(i + 2) % 3])
                end_node = max(triangle[(i + 1) % 3], triangle[(i + 2) % 3])
                coeff = dense_coeff[start_node * nn + end_node]
                rt0_val += coeff * (check_points - self.vertices[triangle[i]].reshape(1, 2)) / self.height[tri_ind, i]

            errors.append(np.mean(np.sum(np.square(f_val - rt0_val), axis=1)))
        return np.sqrt(np.sum(np.array(errors) * self.area))


# CubeHermite Element.
class CubeHermite(FiniteElement):
    xi = None
    xj = None
    xk = None

    yi = None
    yj = None
    yk = None

    wi = None
    wj = None
    wk = None

    def build(self):
        self.xi = self.tri_tensor[:, 0, 0]
        self.yi = self.tri_tensor[:, 0, 1]
        self.xj = self.tri_tensor[:, 1, 0]
        self.yj = self.tri_tensor[:, 1, 1]
        self.xk = self.tri_tensor[:, 2, 0]
        self.yk = self.tri_tensor[:, 2, 1]

        self.wi = np.array([self.yj - self.yk, self.xk - self.xj, self.xj * self.yk - self.xk * self.yj]) / (
                    2 * self.area)
        self.wj = np.array([self.yk - self.yi, self.xi - self.xk, self.xk * self.yi - self.xi * self.yk]) / (
                    2 * self.area)
        self.wk = np.array([self.yi - self.yj, self.xj - self.xi, self.xi * self.yj - self.xj * self.yi]) / (
                    2 * self.area)

    @classmethod
    def factorial(cls, n):
        if n == 0:
            return 1
        else:
            return n * cls.factorial(n - 1)

    def gram_cube_hermite(self):
        """
        :return: 3-order tensor with shape [NT, 10, 10].
        """

        def polynomial_integer(coeff_1, coeff_2):
            """
                                                                                  \alpha!\beta!gamma!
            \iint_K \lambda_i^\alpha \lambda_j^\beta \lambda_k^\gamma dxdy = ------------------------------ * 2 * area
                                                                             (\alpha + \beta + \gamma + 2)!
            coeff = [[weight, alpha, beta, gamma], ...]
            """
            result = 0
            for weight_1, alpha_1, beta_1, gamma_1 in coeff_1:
                for weight_2, alpha_2, beta_2, gamma_2 in coeff_2:
                    weight = weight_1 * weight_2
                    alpha = alpha_1 + alpha_2
                    beta = beta_1 + beta_2
                    gamma = gamma_1 + gamma_2
                    result += weight * self.factorial(alpha) * self.factorial(beta) * self.factorial(gamma) / \
                              self.factorial(alpha + beta + gamma_1 + 2)
            return result

        nt = self.triangles.__len__()
        gram_ch = np.zeros(shape=(nt, 10, 10), dtype=np.float)
        for t in range(nt):

            coeff_gi = [[3, 2, 0, 0], [-2, 3, 0, 0], [-7, 1, 1, 1]]
            coeff_gj = [[3, 0, 2, 0], [-2, 0, 3, 0], [-7, 1, 1, 1]]
            coeff_gk = [[3, 0, 0, 2], [-2, 0, 0, 3], [-7, 1, 1, 1]]

            coeff_ri_ = [[1, 3, 0, 0], [-1, 2, 0, 0], [2, 1, 1, 1]]
            coeff_rj_ = [[1, 1, 2, 0], [-1, 1, 1, 1]]
            coeff_rk_ = [[1, 1, 0, 2], [-1, 1, 1, 1]]

            coeff_si_ = [[1, 2, 1, 0], [-1, 1, 1, 1]]
            coeff_sj_ = [[1, 0, 3, 0], [-1, 0, 2, 0], [2, 1, 1, 1]]
            coeff_sk_ = [[1, 0, 1, 2], [-1, 1, 1, 1]]

            xi = self.xi[t]
            xj = self.xj[t]
            xk = self.xk[t]

            yi = self.yi[t]
            yj = self.yj[t]
            yk = self.yk[t]

            coeff_ri = [[(xi - xk) * it[0], it[1], it[2], it[3]] for it in coeff_ri_] + \
                [[(xj - xk) * it[0], it[1], it[2], it[3]] for it in coeff_si_]
            coeff_rj = [[(xi - xk) * it[0], it[1], it[2], it[3]] for it in coeff_rj_] + \
                [[(xj - xk) * it[0], it[1], it[2], it[3]] for it in coeff_sj_]
            coeff_rk = [[(xi - xk) * it[0], it[1], it[2], it[3]] for it in coeff_rk_] + \
                [[(xj - xk) * it[0], it[1], it[2], it[3]] for it in coeff_sk_]

            coeff_si = [[(yi - yk) * it[0], it[1], it[2], it[3]] for it in coeff_ri_] + \
                [[(yj - yk) * it[0], it[1], it[2], it[3]] for it in coeff_si_]
            coeff_sj = [[(yi - yk) * it[0], it[1], it[2], it[3]] for it in coeff_rj_] + \
                [[(yj - yk) * it[0], it[1], it[2], it[3]] for it in coeff_sj_]
            coeff_sk = [[(yi - yk) * it[0], it[1], it[2], it[3]] for it in coeff_rk_] + \
                [[(yj - yk) * it[0], it[1], it[2], it[3]] for it in coeff_sk_]

            coeff_q = [[27, 1, 1, 1]]

            coeff_collection = [
                coeff_gi, coeff_gj, coeff_gk,
                coeff_ri, coeff_rj, coeff_rk,
                coeff_si, coeff_sj, coeff_sk,
                coeff_q
            ]

            for i in range(10):
                for j in range(10):
                    gram_ch[t, i, j] = 2 * self.area[t] * polynomial_integer(coeff_collection[i], coeff_collection[j])

        return gram_ch

    def gram_p1_cube_hermite(self):
        """
        gram_p1_chg: 3-order tensor with shape [NT, 3, 3].
        gram_p1_chr: 3-order tensor with shape [NT, 3, 3].
        gram_p1_chs: 3-order tensor with shape [NT, 3, 3].
        gram_p1_chq: 2-order tensor with shape [NT, 3].
        """

        def polynomial_integer(coeff):
            """
                                                                                  \alpha!\beta!gamma!
            \iint_K \lambda_i^\alpha \lambda_j^\beta \lambda_k^\gamma dxdy = ------------------------------ * 2 * area
                                                                             (\alpha + \beta + \gamma + 2)!
            coeff = [[weight, alpha, beta, gamma], ...]
            """
            w1, w2, w3 = 0, 0, 0
            for weight, alpha, beta, gamma in coeff:
                w1 += weight * self.factorial(alpha + 1) * self.factorial(beta) * self.factorial(gamma) / \
                      self.factorial(alpha + beta + gamma + 3)
                w2 += weight * self.factorial(alpha) * self.factorial(beta + 1) * self.factorial(gamma) / \
                      self.factorial(alpha + beta + gamma + 3)
                w3 += weight * self.factorial(alpha) * self.factorial(beta) * self.factorial(gamma + 1) / \
                      self.factorial(alpha + beta + gamma + 3)
            return np.array([w1, w2, w3])

        trans_mat = np.vstack([
            polynomial_integer([[3, 2, 0, 0], [-2, 3, 0, 0], [-7, 1, 1, 1]]),
            polynomial_integer([[3, 0, 2, 0], [-2, 0, 3, 0], [-7, 1, 1, 1]]),
            polynomial_integer([[3, 0, 0, 2], [-2, 0, 0, 3], [-7, 1, 1, 1]]),
        ]).T
        gram_p1_chg = np.einsum("t,ij->tij", 2 * self.area, trans_mat)

        trans_mat = np.vstack([
            polynomial_integer([[1, 3, 0, 0], [-1, 2, 0, 0], [2, 1, 1, 1]]),
            polynomial_integer([[1, 1, 2, 0], [-1, 1, 1, 1]]),
            polynomial_integer([[1, 1, 0, 2], [-1, 1, 1, 1]])
        ]).T
        chr = np.einsum("t,ij->tij", 2 * self.area, trans_mat)

        trans_mat = np.vstack([
            polynomial_integer([[1, 2, 1, 0], [-1, 1, 1, 1]]),
            polynomial_integer([[1, 0, 3, 0], [-1, 0, 2, 0], [2, 1, 1, 1]]),
            polynomial_integer([[1, 0, 1, 2], [-1, 1, 1, 1]])
        ]).T
        chs = np.einsum("t,ij->tij", 2 * self.area, trans_mat)

        gram_p1_chr = np.einsum("t,tij->tij", self.xi - self.xk, chr) + np.einsum("t,tij->tij", self.xj - self.xk, chs)
        gram_p1_chs = np.einsum("t,tij->tij", self.yi - self.yk, chr) + np.einsum("t,tij->tij", self.yj - self.yk, chs)

        trans_mat = polynomial_integer([[27, 1, 1, 1]])
        gram_p1_chq = np.einsum("t,i->ti", 2 * self.area, trans_mat)

        return gram_p1_chg, gram_p1_chr, gram_p1_chs, gram_p1_chq

    def gram_grad_p1_cube_hermite(self):
        """
        gram_grad_p1_chg: 4-order tensor with shape [NT, 3, 2, 3].
        gram_grad_p1_chr: 4-order tensor with shape [NT, 3, 2, 3].
        gram_grad_p1_chs: 4-order tensor with shape [NT, 3, 2, 3].
        gram_grad_p1_chq: 3-order tensor with shape [NT, 3, 2].
        """

        def polynomial_integer(coeff):
            """
                                                                                  \alpha!\beta!gamma!
            \iint_K \lambda_i^\alpha \lambda_j^\beta \lambda_k^\gamma dxdy = ------------------------------ * 2 * area
                                                                             (\alpha + \beta + \gamma + 2)!
            coeff = [[weight, alpha, beta, gamma], ...]
            """
            w = 0
            for weight, alpha, beta, gamma in coeff:
                w += weight * self.factorial(alpha) * self.factorial(beta) * self.factorial(gamma) / \
                     self.factorial(alpha + beta + gamma + 2)
            return w

        trans_vec = np.array([
            polynomial_integer([[3, 2, 0, 0], [-2, 3, 0, 0], [-7, 1, 1, 1]]),
            polynomial_integer([[3, 0, 2, 0], [-2, 0, 3, 0], [-7, 1, 1, 1]]),
            polynomial_integer([[3, 0, 0, 2], [-2, 0, 0, 3], [-7, 1, 1, 1]]),
        ])
        gram_grad_p1_chg = np.einsum("t,tid,j->tidj", 2 * self.area, self.grad_p1, trans_vec)

        trans_vec = np.array([
            polynomial_integer([[1, 3, 0, 0], [-1, 2, 0, 0], [2, 1, 1, 1]]),
            polynomial_integer([[1, 1, 2, 0], [-1, 1, 1, 1]]),
            polynomial_integer([[1, 1, 0, 2], [-1, 1, 1, 1]])
        ])
        chr = np.einsum("t,tid,j->tidj", 2 * self.area, self.grad_p1, trans_vec)

        trans_vec = np.array([
            polynomial_integer([[1, 2, 1, 0], [-1, 1, 1, 1]]),
            polynomial_integer([[1, 0, 3, 0], [-1, 0, 2, 0], [2, 1, 1, 1]]),
            polynomial_integer([[1, 0, 1, 2], [-1, 1, 1, 1]])
        ])
        chs = np.einsum("t,tid,j->tidj", 2 * self.area, self.grad_p1, trans_vec)

        gram_grad_p1_chr = np.einsum("t,tidj->tidj", self.xi - self.xk, chr) + np.einsum(
            "t,tidj->tidj", self.xj - self.xk, chs)
        gram_grad_p1_chs = np.einsum("t,tidj->tidj", self.yi - self.yk, chr) + np.einsum(
            "t,tidj->tidj", self.yj - self.yk, chs)

        trans = polynomial_integer([[27, 1, 1, 1]])
        gram_grad_p1_chq = np.einsum("t,tid->tid", 2 * self.area, self.grad_p1) * trans

        return gram_grad_p1_chg, gram_grad_p1_chr, gram_grad_p1_chs, gram_grad_p1_chq

    def integer_cube_hermite(self, func, num_refine=0):
        """
        integer_chg: 2-order tensor. shape = [NT, 3]
        integer_chr: 2-order tensor. shape = [NT, 3]
        integer_chs: 2-order tensor. shape = [NT, 3]
        integer_chq: 1-order tensor. shape = [NT]
        """
        integer_chg = []
        integer_chr = []
        integer_chs = []
        integer_chq = []

        for tri_vertices, area in zip(self.tri_tensor, self.area):
            def func_element(x):
                delta = np.hstack([tri_vertices, np.ones(shape=(3, 1))])
                delta = np.linalg.det(delta)

                [xi, yi], [xj, yj], [xk, yk] = tri_vertices
                wi = np.array([yj - yk, xk - xj, xj * yk - xk * yj]) / delta
                wj = np.array([yk - yi, xi - xk, xk * yi - xi * yk]) / delta
                wk = np.array([yi - yj, xj - xi, xi * yj - xj * yi]) / delta

                lami = wi[0] * x[0] + wi[1] * x[1] + wi[2]
                lamj = wj[0] * x[0] + wj[1] * x[1] + wj[2]
                lamk = wk[0] * x[0] + wk[1] * x[1] + wk[2]

                # g
                gi = 3 * lami ** 2 - 2 * lami ** 3 - 7 * lami * lamj * lamk
                gj = 3 * lamj ** 2 - 2 * lamj ** 3 - 7 * lami * lamj * lamk
                gk = 3 * lamk ** 2 - 2 * lamk ** 3 - 7 * lami * lamj * lamk

                # r & s
                r = [
                    lami ** 3 - lami ** 2 + 2 * lami * lamj * lamk,
                    lami * lamj ** 2 - lami * lamj * lamk,
                    lami * lamk ** 2 - lami * lamj * lamk
                ]

                s = [
                    lami ** 2 * lamj - lami * lamj * lamk,
                    lamj ** 3 - lamj ** 2 + 2 * lami * lamj * lamk,
                    lamk ** 2 * lamj - lami * lamj * lamk
                ]

                ri = (xi - xk) * r[0] + (xj - xk) * s[0]
                rj = (xi - xk) * r[1] + (xj - xk) * s[1]
                rk = (xi - xk) * r[2] + (xj - xk) * s[2]

                si = (yi - yk) * r[0] + (yj - yk) * s[0]
                sj = (yi - yk) * r[1] + (yj - yk) * s[1]
                sk = (yi - yk) * r[2] + (yj - yk) * s[2]

                # q
                q = 27 * lami * lamj * lamk

                return gi, gj, gk, ri, rj, rk, si, sj, sk, q, lami, lamj, lamk

            func_gi = lambda x: func(x) * func_element(x)[0]
            func_gj = lambda x: func(x) * func_element(x)[1]
            func_gk = lambda x: func(x) * func_element(x)[2]

            func_ri = lambda x: func(x) * func_element(x)[3]
            func_rj = lambda x: func(x) * func_element(x)[4]
            func_rk = lambda x: func(x) * func_element(x)[5]

            func_si = lambda x: func(x) * func_element(x)[6]
            func_sj = lambda x: func(x) * func_element(x)[7]
            func_sk = lambda x: func(x) * func_element(x)[8]

            func_q = lambda x: func(x) * func_element(x)[9]

            integer_func_gi = self.estimate_integer(func_gi, tri_vertices, num_refine=num_refine)
            integer_func_gj = self.estimate_integer(func_gj, tri_vertices, num_refine=num_refine)
            integer_func_gk = self.estimate_integer(func_gk, tri_vertices, num_refine=num_refine)
            integer_chg.append([integer_func_gi, integer_func_gj, integer_func_gk])

            integer_func_ri = self.estimate_integer(func_ri, tri_vertices, num_refine=num_refine)
            integer_func_rj = self.estimate_integer(func_rj, tri_vertices, num_refine=num_refine)
            integer_func_rk = self.estimate_integer(func_rk, tri_vertices, num_refine=num_refine)
            integer_chr.append([integer_func_ri, integer_func_rj, integer_func_rk])

            integer_func_si = self.estimate_integer(func_si, tri_vertices, num_refine=num_refine)
            integer_func_sj = self.estimate_integer(func_sj, tri_vertices, num_refine=num_refine)
            integer_func_sk = self.estimate_integer(func_sk, tri_vertices, num_refine=num_refine)
            integer_chs.append([integer_func_si, integer_func_sj, integer_func_sk])

            integer_func_q = self.estimate_integer(func_q, tri_vertices, num_refine=num_refine)
            integer_chq.append(integer_func_q)

        return np.array(integer_chg), np.array(integer_chr), np.array(integer_chs), np.array(integer_chq)

    @classmethod
    def unit_test(cls):
        def func_element(x, vertices):
            delta = np.hstack([vertices, np.ones(shape=(3, 1))])
            delta = np.linalg.det(delta)

            [xi, yi], [xj, yj], [xk, yk] = vertices
            wi = np.array([yj - yk, xk - xj, xj * yk - xk * yj]) / delta
            wj = np.array([yk - yi, xi - xk, xk * yi - xi * yk]) / delta
            wk = np.array([yi - yj, xj - xi, xi * yj - xj * yi]) / delta

            lami = wi[0] * x[0] + wi[1] * x[1] + wi[2]
            lamj = wj[0] * x[0] + wj[1] * x[1] + wj[2]
            lamk = wk[0] * x[0] + wk[1] * x[1] + wk[2]

            # g
            gi = 3 * lami ** 2 - 2 * lami ** 3 - 7 * lami * lamj * lamk
            gj = 3 * lamj ** 2 - 2 * lamj ** 3 - 7 * lami * lamj * lamk
            gk = 3 * lamk ** 2 - 2 * lamk ** 3 - 7 * lami * lamj * lamk

            # r & s
            r = [
                lami ** 3 - lami ** 2 + 2 * lami * lamj * lamk,
                lami * lamj ** 2 - lami * lamj * lamk,
                lami * lamk ** 2 - lami * lamj * lamk
            ]

            s = [
                lami ** 2 * lamj - lami * lamj * lamk,
                lamj ** 3 - lamj ** 2 + 2 * lami * lamj * lamk,
                lamk ** 2 * lamj - lami * lamj * lamk
            ]

            ri = (xi - xk) * r[0] + (xj - xk) * s[0]
            rj = (xi - xk) * r[1] + (xj - xk) * s[1]
            rk = (xi - xk) * r[2] + (xj - xk) * s[2]

            si = (yi - yk) * r[0] + (yj - yk) * s[0]
            sj = (yi - yk) * r[1] + (yj - yk) * s[1]
            sk = (yi - yk) * r[2] + (yj - yk) * s[2]

            # q
            q = 27 * lami * lamj * lamk

            return gi, gj, gk, ri, rj, rk, si, sj, sk, q, lami, lamj, lamk

        vertices = np.random.rand(3, 2)

        for element in func_element(vertices.T, vertices):
            print(element)
        print()
        for element in func_element(np.mean(vertices, axis=0), vertices):
            print(element)
        print()

        eps = 1e-6

        for i in range(3):
            for j in range(3):
                x, y = vertices[j]

                dgidx = (func_element([x + eps, y], vertices)[i] - func_element([x - eps, y], vertices)[i]) / (2 * eps)
                dgidy = (func_element([x, y + eps], vertices)[i] - func_element([x, y - eps], vertices)[i]) / (2 * eps)
                r = func_element([x, y], vertices)[i + 3]
                s = func_element([x, y], vertices)[i + 6]
                print(i, j, dgidx - r, dgidy - s)

        # check gram matrix
        mesh = cls()
        mesh.vertices = vertices
        mesh.triangles = np.array([[0, 1, 2]])
        mesh.neighbors = np.array([[-1, -1, -1]])
        mesh.build()

        func_lami = lambda x: func_element(x, vertices)[-3]
        func_lamj = lambda x: func_element(x, vertices)[-2]
        func_lamk = lambda x: func_element(x, vertices)[-1]

        func_gi = lambda x: func_element(x, vertices)[0]
        func_gj = lambda x: func_element(x, vertices)[1]
        func_gk = lambda x: func_element(x, vertices)[2]

        func_ri = lambda x: func_element(x, vertices)[3]
        func_rj = lambda x: func_element(x, vertices)[4]
        func_rk = lambda x: func_element(x, vertices)[5]

        func_si = lambda x: func_element(x, vertices)[6]
        func_sj = lambda x: func_element(x, vertices)[7]
        func_sk = lambda x: func_element(x, vertices)[8]

        func_q = lambda x: func_element(x, vertices)[9]

        gram_p1_chg = np.zeros(shape=(3, 3), dtype=np.float)
        for i, func_lam in enumerate([func_lami, func_lamj, func_lamk]):
            for j, func_g in enumerate([func_gi, func_gj, func_gk]):
                gram_p1_chg[i, j] = cls.estimate_integer(lambda x: func_lam(x) * func_g(x), vertices, num_refine=6)
        print("gram_p1_chg:\n", gram_p1_chg)
        print("error(gram_p1_chg):\n", mesh.gram_p1_cube_hermite()[0][0] - gram_p1_chg)

        gram_p1_chr = np.zeros(shape=(3, 3), dtype=np.float)
        for i, func_lam in enumerate([func_lami, func_lamj, func_lamk]):
            for j, func_r in enumerate([func_ri, func_rj, func_rk]):
                gram_p1_chr[i, j] = cls.estimate_integer(lambda x: func_lam(x) * func_r(x), vertices, num_refine=6)
        print("gram_p1_chr:\n", gram_p1_chr)
        print("error(gram_p1_chr):\n", mesh.gram_p1_cube_hermite()[1][0] - gram_p1_chr)

        gram_p1_chs = np.zeros(shape=(3, 3), dtype=np.float)
        for i, func_lam in enumerate([func_lami, func_lamj, func_lamk]):
            for j, func_s in enumerate([func_si, func_sj, func_sk]):
                gram_p1_chs[i, j] = cls.estimate_integer(lambda x: func_lam(x) * func_s(x), vertices, num_refine=6)
        print("gram_p1_chs:\n", gram_p1_chs)
        print("error(gram_p1_chs):\n", mesh.gram_p1_cube_hermite()[2][0] - gram_p1_chs)

        gram_p1_chq = np.zeros(shape=(3,), dtype=np.float)
        for i, func_lam in enumerate([func_lami, func_lamj, func_lamk]):
            gram_p1_chq[i] = cls.estimate_integer(lambda x: func_lam(x) * func_q(x), vertices, num_refine=6)
        print("gram_p1_chq:\n", gram_p1_chq)
        print("error(gram_p1_chq):\n", mesh.gram_p1_cube_hermite()[3][0] - gram_p1_chq)

        gram_grad_p1_chg = np.zeros(shape=(3, 2, 3), dtype=np.float)
        for i, grad in enumerate([mesh.wi, mesh.wj, mesh.wk]):
            for d in range(2):
                for j, func_g in enumerate([func_gi, func_gj, func_gk]):
                    gram_grad_p1_chg[i, d, j] = cls.estimate_integer(
                        lambda x: grad[d, 0] * func_g(x), vertices, num_refine=6)
        print("gram_grad_p1_chg:\n", gram_grad_p1_chg)
        print("error(gram_grad_p1_chg):\n", mesh.gram_grad_p1_cube_hermite()[0][0] - gram_grad_p1_chg)

        gram_grad_p1_chr = np.zeros(shape=(3, 2, 3), dtype=np.float)
        for i, grad in enumerate([mesh.wi, mesh.wj, mesh.wk]):
            for d in range(2):
                for j, func_r in enumerate([func_ri, func_rj, func_rk]):
                    gram_grad_p1_chr[i, d, j] = cls.estimate_integer(
                        lambda x: grad[d, 0] * func_r(x), vertices, num_refine=6)
        print("gram_grad_p1_chr:\n", gram_grad_p1_chr)
        print("error(gram_grad_p1_chr):\n", mesh.gram_grad_p1_cube_hermite()[1][0] - gram_grad_p1_chr)

        gram_grad_p1_chs = np.zeros(shape=(3, 2, 3), dtype=np.float)
        for i, grad in enumerate([mesh.wi, mesh.wj, mesh.wk]):
            for d in range(2):
                for j, func_s in enumerate([func_si, func_sj, func_sk]):
                    gram_grad_p1_chs[i, d, j] = cls.estimate_integer(
                        lambda x: grad[d, 0] * func_s(x), vertices, num_refine=6)
        print("gram_grad_p1_chs:\n", gram_grad_p1_chs)
        print("error(gram_grad_p1_chs):\n", mesh.gram_grad_p1_cube_hermite()[2][0] - gram_grad_p1_chs)

        gram_grad_p1_chq = np.zeros(shape=(3, 2), dtype=np.float)
        for i, grad in enumerate([mesh.wi, mesh.wj, mesh.wk]):
            for d in range(2):
                gram_grad_p1_chq[i, d] = cls.estimate_integer(
                    lambda x: grad[d, 0] * func_q(x), vertices, num_refine=6)
        print("gram_grad_p1_chq:\n", gram_grad_p1_chq)
        print("error(gram_grad_p1_chq):\n", mesh.gram_grad_p1_cube_hermite()[3][0] - gram_grad_p1_chq)


# TangentP1 Element.
class TangentP1(FiniteElement):
    tangent = None  # [NT, 3, 2]

    def build(self, func_tangent):
        node_ids = self.triangles.reshape(-1)
        vertices = self.vertices[node_ids, :].T
        tx, ty = func_tangent(vertices)
        self.tangent = np.stack([tx.reshape(-1, 3), ty.reshape(-1, 3)], axis=2)  # [NT, 3, 2]

    def gram_p1_st1(self, gram_p1):
        """
        :return: 4-order tensor with shape [NT, 3, 3, 2].
        """
        return np.einsum("tij,tjd->tijd", gram_p1, self.tangent)

    def gram_st1(self, gram_p1):
        """
        :return: 3-order tensor with shape [NT, 3, 3].
        """
        return np.einsum("tij,tid,tjd->tij", gram_p1, self.tangent, self.tangent)

    def gram_p1_div_st1(self, gram_p1_grad_p1):
        """
        :return: 3-order tensor with shape [NT, 3, 3].
        """
        return np.einsum("tijd,tjd->tij", gram_p1_grad_p1, self.tangent)

    def gram_div_st1_cube_hermite(self, gram_grad_p1_cube_hermite):
        """
        gram_div_p1_chg: 3-order tensor with shape [NT, 3, 3].
        gram_div_p1_chr: 3-order tensor with shape [NT, 3, 3].
        gram_div_p1_chs: 3-order tensor with shape [NT, 3, 3].
        gram_div_p1_chq: 2-order tensor with shape [NT, 3].
        """
        gram_grad_p1_chg, gram_grad_p1_chr, gram_grad_p1_chs, gram_grad_p1_chq = gram_grad_p1_cube_hermite

        gram_div_st1_chg = np.einsum("tidj,tid->tij", gram_grad_p1_chg, self.tangent)
        gram_div_st1_chr = np.einsum("tidj,tid->tij", gram_grad_p1_chr, self.tangent)
        gram_div_st1_chs = np.einsum("tidj,tid->tij", gram_grad_p1_chs, self.tangent)
        gram_div_st1_chq = np.einsum("tid,tid->ti", gram_grad_p1_chq, self.tangent)
        return gram_div_st1_chg, gram_div_st1_chr, gram_div_st1_chs, gram_div_st1_chq

    def integer_st1(self, func, num_refine=0):
        """
        :return: sparse tensor. shape = [NT, 3]
        """
        integer_tensor_0 = self.integer_p1(lambda x: func(x)[0], num_refine=num_refine)
        integer_tensor_1 = self.integer_p1(lambda x: func(x)[1], num_refine=num_refine)
        return self.tangent[:, :, 0] * integer_tensor_0 + self.tangent[:, :, 1] * integer_tensor_1


# TangentCubeHermite Element.
class TangentCubeHermite(FiniteElement):
    tangent_in_node = None  # [NT, 3, 2]
    tangent_in_triangle = None  # [NT, 2]

    def build(self, func_tangent):
        node_ids = self.triangles.reshape(-1)
        vertices = self.vertices[node_ids, :].T
        tx, ty = func_tangent(vertices)
        self.tangent_in_node = np.stack([tx.reshape(-1, 3), ty.reshape(-1, 3)], axis=2)  # [NT, 3, 2]

        vertices = np.mean(self.tri_tensor, axis=1).T
        self.tangent_in_triangle = func_tangent(vertices)


class S12(FiniteElement):
    def build(self, anchors):
        # shuffle nodes
        for i in range(self.triangles.__len__()):
            ind = list(self.triangles[i]).index(anchors[i])
            self.triangles[i] = self.triangles[i, [ind % 3, (ind + 1) % 3, (ind + 2) % 3]]
            self.neighbors[i] = self.neighbors[i, [ind % 3, (ind + 1) % 3, (ind + 2) % 3]]

    # -------- Gram matrix(surface X surface) --------
    def gram_s12(self):
        """
        x - xi = (xj - xi) * lam_j + (xk - xi) * lam_k
        y - yi = (yj - yi) * lam_j + (yk - yi) * lam_k

        :return: 3-order tensor with shape [NT, 2].
        """
        xi = self.tri_tensor[:, 0, 0]
        yi = self.tri_tensor[:, 0, 1]
        xj = self.tri_tensor[:, 1, 0]
        yj = self.tri_tensor[:, 1, 1]
        xk = self.tri_tensor[:, 2, 0]
        yk = self.tri_tensor[:, 2, 1]

        intx = ((xj - xi) ** 2 + (xk - xi) ** 2 + (xj - xi) * (xk - xi)) * self.area / 6
        inty = ((yj - yi) ** 2 + (yk - yi) ** 2 + (yj - yi) * (yk - yi)) * self.area / 6

        return np.stack([intx, inty], axis=1)

    # -------- Integer vector(edge X function) --------
    def integer_s12(self, func, num_refine=0):
        """
        :return: sparse tensor. shape = [NT, 2]
        """
        integer_x = []
        integer_y = []

        for tri_vertices in self.tri_tensor:
            def inner_prod(x):
                return func(x)[0] * (x[0] - tri_vertices[0, 0])

            integer_x.append(self.estimate_integer(inner_prod, tri_vertices, num_refine))

            def inner_prod(x):
                return func(x)[1] * (x[1] - tri_vertices[0, 1])

            integer_y.append(self.estimate_integer(inner_prod, tri_vertices, num_refine))

        return np.stack([integer_x, integer_y], axis=1)

# SparseTensor.unit_test()
# SquareMesh.rt0_figure()
# SquareMesh.unit_test()
