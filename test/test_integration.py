import bitarray as ba
import numpy as np
from queue import PriorityQueue


import dyada.coordinates
import dyada.drawing
import dyada.linearization
import dyada.refinement


def test_partitioning_queued():
    # random input tensor
    random_length = np.random.randint(10, 100)
    # values in the tensor correspond to "importance" of the data
    tensor = np.random.rand(240, 160, random_length)
    num_dimensions = tensor.ndim
    possible_refinements = list(dyada.linearization.single_bit_set_gen(num_dimensions))
    num_desired_partitions = 100
    minimum_partition_size = [16, 16, 1]

    def get_dimensionwise_importance(subtensor):
        # to assign the refinement importance per partition and dimension,
        # get the difference quotients for each dimension and add them to the absolute values
        # (can use other dimension-sensitive filters here,
        # can even be different for each dimension) #TODO
        importances = []
        for d in range(num_dimensions):
            difference_quotient = np.diff(subtensor, axis=d)
            # normalize the accumulated quotient by the size of the difference quotient array
            difference_quotient_normalized = np.sum(difference_quotient) / (
                difference_quotient.size * subtensor.size
            )
            importances.append(
                np.sum(np.abs(subtensor)) + difference_quotient_normalized
            )
        return importances

    # represent whole tensor domain by discretization
    discretization = dyada.refinement.Discretization(
        dyada.linearization.MortonOrderLinearization(),
        dyada.refinement.RefinementDescriptor(num_dimensions, 0),
    )
    full_domain = dyada.coordinates.interval_from_sequences(
        [0] * num_dimensions, tensor.shape
    )
    while len(discretization) < num_desired_partitions:
        priority_queue = PriorityQueue()
        for i in range(len(discretization)):
            # what area does the i-th partition represent?
            interval = dyada.refinement.coordinates_from_box_index(
                discretization, i, full_domain
            )
            # use exactly the voxels, slightly more if necessary
            sub_tensor = tensor[
                tuple(
                    slice(
                        int(np.floor(interval.lower_bound[d])),
                        int(np.ceil(interval.upper_bound[d])),
                    )
                    for d in range(num_dimensions)
                )
            ]
            importance = get_dimensionwise_importance(sub_tensor)
            for d in range(num_dimensions):
                # skip if the partition would be too small
                if sub_tensor.shape[d] < minimum_partition_size[d] * 2:
                    continue
                priority_queue.put((-importance[d], (i, possible_refinements[d])))

        # partition the tensor into two parts based on the highest importance
        neg_importance, next_refinement = priority_queue.get()
        # can be sped up by re-mapping instead of recomputing all priorities
        discretization, index_mapping = dyada.refinement.apply_single_refinement(
            discretization, *next_refinement
        )

    # plot the final partitioning -- look for "patches_solid.pdf" in the current directory
    dyada.drawing.plot_all_boxes_3d(discretization, labels=None, filename="patches")

    # interpolation / integration of each partition to desired resolution... processing... other things...#TODO
    # dummy variant: take the mean in the last dimension, resample to 16 in the other dimensions
    partitions_resampled = []
    for i in range(len(discretization)):
        interval = dyada.refinement.coordinates_from_box_index(
            discretization, i, full_domain
        )
        sub_tensor = tensor[
            tuple(
                slice(
                    int(np.floor(interval.lower_bound[d])),
                    int(np.ceil(interval.upper_bound[d])),
                )
                for d in range(num_dimensions)
            )
        ]
        sub_tensor_averaged = np.mean(sub_tensor, axis=-1)
        x = np.arange(sub_tensor_averaged.shape[0])
        y = np.arange(sub_tensor_averaged.shape[1])
        try:
            from scipy.interpolate import RectBivariateSpline  # type: ignore

            # bicubic interpolation w/ scipy
            sub_tensor_spline = RectBivariateSpline(x, y, sub_tensor_averaged)
            x_new = np.linspace(0, sub_tensor_averaged.shape[0], 16)
            y_new = np.linspace(0, sub_tensor_averaged.shape[1], 16)
            sub_tensor_resampled = sub_tensor_spline(x_new, y_new)
        except ImportError as e:
            # fallback: subsampling
            sub_tensor_resampled = np.zeros((16, 16))
            for x_new in range(16):
                for y_new in range(16):
                    x_old = x_new * sub_tensor_averaged.shape[0] // 16
                    y_old = y_new * sub_tensor_averaged.shape[1] // 16
                    sub_tensor_resampled[x_new, y_new] = sub_tensor_averaged[
                        x_old, y_old
                    ]
        assert sub_tensor_resampled.shape == (16, 16)
        partitions_resampled.append(sub_tensor_resampled)

        # questions:
        # 1. do we need to do slicing (i.e., know which patches occur at a certain frame)?
        # 2. can/should neighbor information be included?
