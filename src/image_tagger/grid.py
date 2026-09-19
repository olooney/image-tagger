import importlib
from typing import Any, cast

import numpy as np

linear_sum_assignment = cast(
    "Any",
    importlib.import_module("scipy.optimize").linear_sum_assignment,
)
cdist = cast(
    "Any",
    importlib.import_module("scipy.spatial.distance").cdist,
)
TSNE = cast(
    "Any",
    importlib.import_module("sklearn.manifold").TSNE,
)


def load_clip_vectors(
    npz_path: str,
    n_images: int | None = None,
    key: str = "vectors",
) -> np.ndarray:
    """
    Read the first n_images CLIP vectors from an NPZ archive.

    Returns
    -------
    vectors : np.ndarray, shape (n_images, embedding_dim)
    """
    with np.load(npz_path, allow_pickle=False) as archive:
        if n_images is not None:
            vectors = np.asarray(
                archive[key][:n_images],
                dtype=np.float64,
            )
        else:
            vectors = np.asarray(
                archive[key],
                dtype=np.float64,
            )

    if vectors.ndim != 2 or vectors.shape[0] != n_images:
        raise ValueError(
            f"Expected {n_images} vectors in a 2D array; got {vectors.shape}."
        )

    if not np.isfinite(vectors).all():
        raise ValueError("Vectors must contain only finite values.")

    return vectors


def normalize_clip_vectors(
    vectors: np.ndarray,
) -> np.ndarray:
    """
    L2-normalize each CLIP vector.

    For normalized vectors, squared Euclidean distance is
    proportional to cosine distance:

        ||x - y||^2 = 2 * (1 - cosine_similarity(x, y))

    This allows t-SNE to use Euclidean distances while
    retaining the original cosine-similarity relationships.
    """
    vectors = np.asarray(vectors, dtype=np.float64)

    norms = np.linalg.norm(
        vectors,
        axis=1,
        keepdims=True,
    )

    if np.any(norms == 0):
        raise ValueError("Cannot normalize a zero-length CLIP vector.")

    return vectors / norms


def tsne_2d(
    vectors: np.ndarray,
    *,
    perplexity: float = 30.0,
    random_state: int = 42,
) -> np.ndarray:
    """
    Embed the supplied vectors into a continuous 2D space.

    Parameters
    ----------
    vectors : np.ndarray
        Input feature vectors, preferably L2-normalized.

    perplexity : float
        t-SNE perplexity. Must be smaller than the number
        of input vectors.

    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    coordinates : np.ndarray, shape (n_images, 2)
        Continuous 2D coordinates.

    Notes
    -----
    PCA initialization is used internally by t-SNE.
    This is not the same as applying PCA dimensionality
    reduction to the input vectors beforehand.
    """
    if not 0 < perplexity < len(vectors):
        raise ValueError(
            "perplexity must be strictly between 0 and the number of images."
        )

    model = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=random_state,
        max_iter=1000,
    )

    return model.fit_transform(vectors)


def scale_to_grid(
    coordinates: np.ndarray,
    rows: int,
    cols: int,
) -> np.ndarray:
    """Scale continuous 2D coordinates to the grid's bounding box."""

    coordinates = np.asarray(
        coordinates,
        dtype=np.float64,
    )

    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("coordinates must have shape (n_images, 2).")

    if rows < 1 or cols < 1:
        raise ValueError("rows and cols must be positive.")

    minimum = coordinates.min(axis=0)

    span = coordinates.max(axis=0) - minimum

    extent = np.array(
        [cols - 1, rows - 1],
        dtype=np.float64,
    )

    # Initialize at the center of the grid.
    # This also handles any axis with zero variation.
    scaled = np.broadcast_to(
        extent / 2,
        coordinates.shape,
    ).copy()

    varying = span > 0

    scaled[:, varying] = (
        (coordinates[:, varying] - minimum[varying]) / span[varying] * extent[varying]
    )

    return scaled


def hungarian_grid_assignment(
    coordinates: np.ndarray,
    rows: int,
    cols: int,
) -> np.ndarray:
    """
    Assign each image to a unique grid cell.

    Minimizes total squared Euclidean displacement between
    the continuous image coordinates and discrete grid cells.

    Objective:

        minimize sum_i ||coordinates[i] - cell[assignment[i]]||^2
    """
    coordinates = np.asarray(
        coordinates,
        dtype=np.float64,
    )

    if coordinates.shape != (rows * cols, 2):
        raise ValueError(
            f"Expected {(rows * cols, 2)} coordinates; got {coordinates.shape}."
        )

    # Construct the target grid.
    # Each grid cell has coordinates (column, row).
    x, y = np.meshgrid(
        np.arange(cols),
        np.arange(rows),
    )

    cells = np.column_stack((x.ravel(), y.ravel()))

    # Cost matrix:
    # costs[i, j] = squared distance from image i to cell j.
    costs = cdist(
        coordinates,
        cells,
        metric="sqeuclidean",
    )

    # Solve the global minimum-cost assignment.
    image_indices, cell_indices = linear_sum_assignment(costs)

    # Convert the assignment into a row-major grid.
    grid = np.empty(
        rows * cols,
        dtype=int,
    )

    grid[cell_indices] = image_indices

    return grid.reshape(rows, cols)


def arrange_clip_grid(
    vectors: np.ndarray,
    *,
    rows: int = 10,
    cols: int = 10,
    perplexity: float = 30.0,
    random_state: int = 42,
) -> np.ndarray:
    """Arrange CLIP embeddings into a grid."""

    if rows < 1 or cols < 1:
        raise ValueError("rows and cols must be positive.")

    n_images = rows * cols
    vectors = np.asarray(vectors)

    if vectors.ndim != 2 or len(vectors) < n_images:
        raise ValueError(f"At least {n_images} vectors in a 2D array are required.")

    normalized = normalize_clip_vectors(vectors[:n_images])

    positions = tsne_2d(
        normalized,
        perplexity=perplexity,
        random_state=random_state,
    )

    scaled_positions = scale_to_grid(
        positions,
        rows=rows,
        cols=cols,
    )

    grid = hungarian_grid_assignment(
        scaled_positions,
        rows=rows,
        cols=cols,
    )

    return grid
