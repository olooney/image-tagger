const updateReviewState = () => {
    const hasImages = document.querySelectorAll('#review-list .gallery-image').length > 0;
    const emptyMessage = document.querySelector('#empty-review-message');
    const shelveControls = document.querySelector('#shelve-controls');
    if (emptyMessage) {
        emptyMessage.hidden = hasImages;
    }
    if (shelveControls) {
        shelveControls.hidden = hasImages;
    }
};

const markDirty = (event) => {
    const form = event.target.closest('form');
    if (form) {
        form.querySelector('button[type="submit"]').disabled = false;
        form.querySelector('.save-status').textContent = '';
    }
};

const cropEditor = (() => {
    const margin = 100;
    const state = {
        rowId: null,
        image: null,
        imageData: null,
        points: [],
        view: null,
        draggedHandle: null,
        lastTouchedHandle: null,
        previewTimer: null,
        mode: 'perspective',
        rectangleAspect: 1,
        rectangleNaturalWidth: 1,
        rectangleNaturalHeight: 1,
        dragStartAspect: null,
        canvasBackgroundIndex: 2,
        eyedropperActive: false,
    };

    const canvasBackgrounds = [
        { name: 'black', color: '#000000' },
        { name: 'white', color: '#ffffff' },
        { name: 'gray', color: '#666666' },
    ];

    const modal = () => document.querySelector('#crop-modal');
    const sourceCanvas = () => document.querySelector('#crop-source-canvas');
    const previewCanvas = () => document.querySelector('#crop-preview-canvas');
    const colorInput = () => document.querySelector('#crop-background');
    const errorBox = () => document.querySelector('#crop-error');
    const loading = () => document.querySelector('#crop-loading');
    const outputWidthInput = () => document.querySelector('#crop-output-width');
    const outputHeightInput = () => document.querySelector('#crop-output-height');
    const downsamplingSelect = () => document.querySelector('#crop-downsampling');
    const resultSize = () => document.querySelector('#crop-result-size');
    const eyedropperButton = () => document.querySelector('#crop-eyedropper');

    const downsamplingRatios = [
        ['80%', 0.8],
        ['75%', 0.75],
        ['60%', 0.6],
        ['50%', 0.5],
        ['33%', 0.33333333333],
        ['25%', 0.25],
    ];

    const setCanvasBackground = () => {
        const background = canvasBackgrounds[state.canvasBackgroundIndex];
        document.querySelector('.crop-workspace').style.backgroundColor = background.color;
        document.querySelector('#crop-background-swatch').style.backgroundColor = background.color;
        const button = document.querySelector('#crop-canvas-background');
        button.setAttribute('aria-label', `Canvas background: ${background.name}`);
        button.title = `Canvas background: ${background.name}`;
        renderSource();
    };

    const cycleCanvasBackground = () => {
        state.canvasBackgroundIndex = (state.canvasBackgroundIndex + 1) % canvasBackgrounds.length;
        setCanvasBackground();
    };

    const selectAlgorithm = (algorithm) => {
        document.querySelectorAll('[data-crop-detector]').forEach((button) => {
            const selected = button.dataset.cropDetector === algorithm;
            button.classList.toggle('btn-light', selected);
            button.classList.toggle('btn-outline-light', !selected);
            button.setAttribute('aria-pressed', selected ? 'true' : 'false');
        });
    };

    const selectMode = (mode) => {
        document.querySelectorAll('[data-crop-mode]').forEach((button) => {
            const selected = button.dataset.cropMode === mode;
            button.classList.toggle('btn-light', selected);
            button.classList.toggle('btn-outline-light', !selected);
            button.setAttribute('aria-pressed', selected ? 'true' : 'false');
        });
        setMode(mode);
    };

    const parseColor = (hex) => [
        Number.parseInt(hex.slice(1, 3), 16),
        Number.parseInt(hex.slice(3, 5), 16),
        Number.parseInt(hex.slice(5, 7), 16),
    ];

    const setEyedropperActive = (active) => {
        state.eyedropperActive = active;
        sourceCanvas().classList.toggle('eyedrop-active', active);
        eyedropperButton().classList.toggle('btn-light', active);
        eyedropperButton().classList.toggle('btn-outline-light', !active);
        eyedropperButton().setAttribute('aria-pressed', active ? 'true' : 'false');
    };

    const solveLinearSystem = (matrix, values) => {
        const rows = matrix.map((row, index) => [...row, values[index]]);
        for (let column = 0; column < rows.length; column += 1) {
            let pivot = column;
            for (let row = column + 1; row < rows.length; row += 1) {
                if (Math.abs(rows[row][column]) > Math.abs(rows[pivot][column])) {
                    pivot = row;
                }
            }
            [rows[column], rows[pivot]] = [rows[pivot], rows[column]];
            if (Math.abs(rows[column][column]) < 1e-9) {
                return null;
            }
            const divisor = rows[column][column];
            for (let entry = column; entry <= rows.length; entry += 1) {
                rows[column][entry] /= divisor;
            }
            for (let row = 0; row < rows.length; row += 1) {
                if (row === column) continue;
                const factor = rows[row][column];
                for (let entry = column; entry <= rows.length; entry += 1) {
                    rows[row][entry] -= factor * rows[column][entry];
                }
            }
        }
        return rows.map((row) => row[rows.length]);
    };

    const homography = (destination, source) => {
        const matrix = [];
        const values = [];
        destination.forEach(([x, y], index) => {
            const [u, v] = source[index];
            matrix.push([x, y, 1, 0, 0, 0, -u * x, -u * y]);
            values.push(u);
            matrix.push([0, 0, 0, x, y, 1, -v * x, -v * y]);
            values.push(v);
        });
        return solveLinearSystem(matrix, values);
    };

    const outputSize = () => {
        let size;
        if (state.mode === 'rectangle') {
            size = [
                Math.max(1, Number.parseInt(outputWidthInput().value, 10) || 1),
                Math.max(1, Number.parseInt(outputHeightInput().value, 10) || 1),
            ];
        } else {
            const distance = (first, second) => Math.hypot(
                second[0] - first[0], second[1] - first[1],
            );
            size = [
                Math.max(2, Math.round(Math.max(distance(state.points[0], state.points[1]), distance(state.points[3], state.points[2])))),
                Math.max(2, Math.round(Math.max(distance(state.points[0], state.points[3]), distance(state.points[1], state.points[2])))),
            ];
        }
        resultSize().textContent = `${size[0]}x${size[1]}`;
        return size;
    };

    const renderPreview = () => {
        if (!state.imageData || state.points.length !== 4) return;
        const [naturalWidth, naturalHeight] = outputSize();
        const scale = Math.min(1, 1600 / Math.max(naturalWidth, naturalHeight));
        const width = Math.max(2, Math.round(naturalWidth * scale));
        const height = Math.max(2, Math.round(naturalHeight * scale));
        const destination = [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]];
        const source = state.imageData;

        // Shift overscan into a virtual padded raster so negative source coordinates
        // have the same constant-border behavior as OpenCV's warpPerspective.
        const padding = {
            left: Math.max(0, Math.ceil(-Math.min(...state.points.map(([x]) => x)))) + 2,
            top: Math.max(0, Math.ceil(-Math.min(...state.points.map(([, y]) => y)))) + 2,
            right: Math.max(0, Math.ceil(Math.max(...state.points.map(([x]) => x)) - (source.width - 1))) + 2,
            bottom: Math.max(0, Math.ceil(Math.max(...state.points.map(([, y]) => y)) - (source.height - 1))) + 2,
        };
        const paddedPoints = state.points.map(([x, y]) => [x + padding.left, y + padding.top]);
        const mapping = homography(destination, paddedPoints);
        if (!mapping) return;
        const canvas = previewCanvas();
        canvas.width = width;
        canvas.height = height;
        const context = canvas.getContext('2d');
        const output = context.createImageData(width, height);
        const background = parseColor(colorInput().value);
        for (let y = 0; y < height; y += 1) {
            for (let x = 0; x < width; x += 1) {
                const denominator = mapping[6] * x + mapping[7] * y + 1;
                const paddedX = Math.floor((mapping[0] * x + mapping[1] * y + mapping[2]) / denominator);
                const paddedY = Math.floor((mapping[3] * x + mapping[4] * y + mapping[5]) / denominator);
                const sourceX = paddedX - padding.left;
                const sourceY = paddedY - padding.top;
                const targetIndex = (y * width + x) * 4;
                if (sourceX >= 0 && sourceX < source.width && sourceY >= 0 && sourceY < source.height) {
                    const sourceIndex = (sourceY * source.width + sourceX) * 4;
                    output.data[targetIndex] = source.data[sourceIndex];
                    output.data[targetIndex + 1] = source.data[sourceIndex + 1];
                    output.data[targetIndex + 2] = source.data[sourceIndex + 2];
                } else {
                    output.data[targetIndex] = background[0];
                    output.data[targetIndex + 1] = background[1];
                    output.data[targetIndex + 2] = background[2];
                }
                output.data[targetIndex + 3] = 255;
            }
        }
        context.putImageData(output, 0, 0);
    };

    const bufferPreview = () => {
        window.clearTimeout(state.previewTimer);
        state.previewTimer = window.setTimeout(renderPreview, 100);
    };

    const imageToCanvas = ([x, y]) => [
        state.view.left + x * state.view.scale,
        state.view.top + y * state.view.scale,
    ];

    const canvasToImage = ([x, y]) => [
        (x - state.view.left) / state.view.scale,
        (y - state.view.top) / state.view.scale,
    ];

    const sampleEyedropperColor = (pointer) => {
        if (!state.imageData || !state.view
            || pointer[0] < state.view.left || pointer[0] >= state.view.left + state.view.width
            || pointer[1] < state.view.top || pointer[1] >= state.view.top + state.view.height) {
            return false;
        }
        // Sample the original raster rather than the scaled source canvas.
        const [imageX, imageY] = canvasToImage(pointer);
        const x = Math.max(0, Math.min(state.imageData.width - 1, Math.floor(imageX)));
        const y = Math.max(0, Math.min(state.imageData.height - 1, Math.floor(imageY)));
        const pixelIndex = (y * state.imageData.width + x) * 4;
        const color = [0, 1, 2]
            .map((offset) => state.imageData.data[pixelIndex + offset].toString(16).padStart(2, '0'))
            .join('');
        colorInput().value = `#${color}`;
        colorInput().dispatchEvent(new Event('input', { bubbles: true }));
        setEyedropperActive(false);
        return true;
    };

    const renderSource = () => {
        if (!state.image) return;
        const canvas = sourceCanvas();
        const bounds = canvas.getBoundingClientRect();
        canvas.width = Math.max(1, Math.round(bounds.width));
        canvas.height = Math.max(1, Math.round(bounds.height));
        const context = canvas.getContext('2d');
        context.fillStyle = canvasBackgrounds[state.canvasBackgroundIndex].color;
        context.fillRect(0, 0, canvas.width, canvas.height);
        const scale = Math.max(0.01, Math.min(
            (canvas.width - margin * 2) / state.image.naturalWidth,
            (canvas.height - margin * 2) / state.image.naturalHeight,
        ));
        const drawWidth = state.image.naturalWidth * scale;
        const drawHeight = state.image.naturalHeight * scale;
        state.view = {
            scale,
            left: (canvas.width - drawWidth) / 2,
            top: (canvas.height - drawHeight) / 2,
            width: drawWidth,
            height: drawHeight,
        };
        context.drawImage(state.image, state.view.left, state.view.top, drawWidth, drawHeight);
        const canvasPoints = state.points.map(imageToCanvas);
        context.beginPath();
        context.moveTo(...canvasPoints[0]);
        canvasPoints.slice(1).forEach((point) => context.lineTo(...point));
        context.closePath();
        context.strokeStyle = 'rgba(0, 255, 80, 0.5)';
        context.lineWidth = 1;
        context.stroke();
        canvasPoints.forEach(([x, y]) => {
            context.beginPath();
            context.arc(x, y, 7, 0, Math.PI * 2);
            context.fillStyle = '#00e85b';
            context.fill();
            context.strokeStyle = '#fff';
            context.stroke();
        });
        if (state.mode === 'rectangle') {
            [[0, 1], [1, 2], [2, 3], [3, 0]].forEach(([start, end]) => {
                const x = (canvasPoints[start][0] + canvasPoints[end][0]) / 2;
                const y = (canvasPoints[start][1] + canvasPoints[end][1]) / 2;
                context.fillStyle = '#00e85b';
                context.fillRect(x - 5, y - 5, 10, 10);
                context.strokeStyle = '#fff';
                context.strokeRect(x - 5, y - 5, 10, 10);
            });
        }
    };

    const pointerPosition = (event) => {
        const bounds = sourceCanvas().getBoundingClientRect();
        return [event.clientX - bounds.left, event.clientY - bounds.top];
    };

    const pointToSegmentDistance = (point, start, end) => {
        const dx = end[0] - start[0];
        const dy = end[1] - start[1];
        const lengthSquared = dx * dx + dy * dy;
        if (!lengthSquared) return Math.hypot(point[0] - start[0], point[1] - start[1]);
        const amount = Math.max(0, Math.min(1, ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / lengthSquared));
        return Math.hypot(point[0] - (start[0] + amount * dx), point[1] - (start[1] + amount * dy));
    };

    const pointerDown = (event) => {
        if (!state.view) return;
        const pointer = pointerPosition(event);
        if (state.eyedropperActive) {
            if (!sampleEyedropperColor(pointer)) {
                setEyedropperActive(false);
            }
            return;
        }
        const points = state.points.map(imageToCanvas);
        let closest = null;
        let closestDistance = 18;
        points.forEach((point, index) => {
            const distance = Math.hypot(point[0] - pointer[0], point[1] - pointer[1]);
            if (distance < closestDistance) {
                closest = index;
                closestDistance = distance;
            }
        });
        if (closest !== null) {
            state.draggedHandle = { type: 'point', index: closest };
            state.lastTouchedHandle = state.draggedHandle;
            state.dragStartAspect = state.mode === 'rectangle'
                ? Math.abs(state.points[1][0] - state.points[0][0]) / Math.max(1, Math.abs(state.points[3][1] - state.points[0][1]))
                : null;
            sourceCanvas().setPointerCapture(event.pointerId);
            return;
        }
        const edges = [[0, 1], [1, 2], [2, 3], [3, 0]];
        let closestEdge = null;
        let closestEdgeDistance = 12;
        edges.forEach(([start, end], index) => {
            const distance = pointToSegmentDistance(pointer, points[start], points[end]);
            if (distance < closestEdgeDistance) {
                closestEdge = index;
                closestEdgeDistance = distance;
            }
        });
        if (closestEdge !== null) {
            state.lastTouchedHandle = { type: 'edge', index: closestEdge };
            if (state.mode === 'rectangle') {
                state.draggedHandle = { type: 'edge', index: closestEdge };
                state.dragStartAspect = null;
                sourceCanvas().setPointerCapture(event.pointerId);
            }
            renderSource();
        }
    };

    const setControlPoint = (index, point, preserveAspect = false, preservedRatio = null) => {
        if (state.mode === 'perspective') {
            state.points[index] = point;
            return;
        }
        let x = Math.max(0, Math.min(state.image.naturalWidth - 1, point[0]));
        let y = Math.max(0, Math.min(state.image.naturalHeight - 1, point[1]));
        if (preserveAspect) {
            const opposite = state.points[(index + 2) % 4];
            const ratio = Math.max(0.0001, preservedRatio ?? (Math.abs(state.points[1][0] - state.points[0][0]) / Math.max(1, Math.abs(state.points[3][1] - state.points[0][1]))));
            const directionX = index === 0 || index === 3 ? -1 : 1;
            const directionY = index === 0 || index === 1 ? -1 : 1;
            const desiredX = Math.abs(x - opposite[0]);
            const desiredY = Math.abs(y - opposite[1]);
            let distance = (ratio * desiredX + desiredY) / (ratio * ratio + 1);
            const maxXDistance = directionX < 0 ? opposite[0] / ratio : (state.image.naturalWidth - 1 - opposite[0]) / ratio;
            const maxYDistance = directionY < 0 ? opposite[1] : state.image.naturalHeight - 1 - opposite[1];
            distance = Math.max(1, Math.min(distance, maxXDistance, maxYDistance));
            x = opposite[0] + directionX * ratio * distance;
            y = opposite[1] + directionY * distance;
        }
        let [left, top] = state.points[0];
        let [right, bottom] = state.points[2];
        if (index === 0) {
            left = Math.min(x, right - 1);
            top = Math.min(y, bottom - 1);
        } else if (index === 1) {
            right = Math.max(x, left + 1);
            top = Math.min(y, bottom - 1);
        } else if (index === 2) {
            right = Math.max(x, left + 1);
            bottom = Math.max(y, top + 1);
        } else {
            left = Math.min(x, right - 1);
            bottom = Math.max(y, top + 1);
        }
        state.points = [[left, top], [right, top], [right, bottom], [left, bottom]];
    };

    const setEdge = (index, point) => {
        let [left, top] = state.points[0];
        let [right, bottom] = state.points[2];
        if (index === 0) top = Math.max(0, Math.min(bottom - 1, point[1]));
        if (index === 1) right = Math.max(left + 1, Math.min(state.image.naturalWidth - 1, point[0]));
        if (index === 2) bottom = Math.max(top + 1, Math.min(state.image.naturalHeight - 1, point[1]));
        if (index === 3) left = Math.max(0, Math.min(right - 1, point[0]));
        state.points = [[left, top], [right, top], [right, bottom], [left, bottom]];
    };

    const targetDownsamplingSize = (ratio) => [
        Math.max(1, Math.round(state.rectangleNaturalWidth * ratio)),
        Math.max(1, Math.round(state.rectangleNaturalHeight * ratio)),
    ];

    const syncDownsamplingOptions = () => {
        downsamplingSelect().querySelectorAll('option[data-ratio]').forEach((option) => {
            const [width, height] = targetDownsamplingSize(Number.parseFloat(option.value));
            option.textContent = `${option.dataset.label} - (${width}x${height})`;
        });
        downsamplingSelect().value = '';
    };

    const syncRectangleDimensions = () => {
        const width = Math.max(1, Math.round(state.points[1][0] - state.points[0][0] + 1));
        const height = Math.max(1, Math.round(state.points[3][1] - state.points[0][1] + 1));
        state.rectangleAspect = width / height;
        state.rectangleNaturalWidth = width;
        state.rectangleNaturalHeight = height;
        outputWidthInput().value = width;
        outputHeightInput().value = height;
        syncDownsamplingOptions();
    };

    const pointerMove = (event) => {
        if (state.draggedHandle === null || !state.view) return;
        const pointer = pointerPosition(event);
        const bounded = [
            Math.max(state.view.left - margin, Math.min(state.view.left + state.view.width + margin, pointer[0])),
            Math.max(state.view.top - margin, Math.min(state.view.top + state.view.height + margin, pointer[1])),
        ];
        const imagePoint = canvasToImage(bounded);
        if (state.draggedHandle.type === 'point') {
            setControlPoint(state.draggedHandle.index, imagePoint, event.shiftKey, state.dragStartAspect);
        } else {
            setEdge(state.draggedHandle.index, imagePoint);
        }
        if (state.mode === 'rectangle') syncRectangleDimensions();
        renderSource();
        bufferPreview();
    };

    const pointerUp = (event) => {
        if (state.draggedHandle !== null && sourceCanvas().hasPointerCapture(event.pointerId)) {
            sourceCanvas().releasePointerCapture(event.pointerId);
        }
        state.draggedHandle = null;
        state.dragStartAspect = null;
    };

    const nudgeLastPoint = (event) => {
        if (modal().hidden || state.lastTouchedHandle === null || !state.view) return;
        const offsets = {
            w: [0, -1],
            arrowup: [0, -1],
            a: [-1, 0],
            arrowleft: [-1, 0],
            s: [0, 1],
            arrowdown: [0, 1],
            d: [1, 0],
            arrowright: [1, 0],
        };
        const offset = offsets[event.key.toLowerCase()];
        if (!offset) return;
        event.preventDefault();
        const handle = state.lastTouchedHandle;
        const point = handle.type === 'point' ? state.points[handle.index] : state.points[handle.index];
        const overscan = margin / state.view.scale;
        if (handle.type === 'point') {
            setControlPoint(handle.index, [
                Math.max(-overscan, Math.min(state.image.naturalWidth + overscan, point[0] + offset[0])),
                Math.max(-overscan, Math.min(state.image.naturalHeight + overscan, point[1] + offset[1])),
            ], event.shiftKey);
        } else if (state.mode === 'perspective') {
            const [start, end] = [[0, 1], [1, 2], [2, 3], [3, 0]][handle.index];
            [start, end].forEach((index) => {
                state.points[index] = [
                    Math.max(-overscan, Math.min(state.image.naturalWidth + overscan, state.points[index][0] + offset[0])),
                    Math.max(-overscan, Math.min(state.image.naturalHeight + overscan, state.points[index][1] + offset[1])),
                ];
            });
        } else if ((handle.index === 0 || handle.index === 2) && offset[1]) {
            setEdge(handle.index, [point[0], point[1] + offset[1]]);
        } else if ((handle.index === 1 || handle.index === 3) && offset[0]) {
            setEdge(handle.index, [point[0] + offset[0], point[1]]);
        } else {
            return;
        }
        if (state.mode === 'rectangle') syncRectangleDimensions();
        renderSource();
        window.clearTimeout(state.previewTimer);
        renderPreview();
    };

    const close = () => {
        window.clearTimeout(state.previewTimer);
        setEyedropperActive(false);
        modal().hidden = true;
        document.body.classList.remove('crop-open');
        state.rowId = null;
        state.image = null;
        state.imageData = null;
        state.points = [];
        state.lastTouchedHandle = null;
        state.dragStartAspect = null;
        colorInput().value = '#ffffff';
    };

    const setMode = (mode) => {
        state.mode = mode;
        document.querySelector('#perspective-tools').hidden = mode !== 'perspective';
        document.querySelector('#rectangle-tools').hidden = mode !== 'rectangle';
        if (mode === 'rectangle' && state.image && state.points.length === 4) {
            const xs = state.points.map(([x]) => Math.max(0, Math.min(state.image.naturalWidth - 1, x)));
            const ys = state.points.map(([, y]) => Math.max(0, Math.min(state.image.naturalHeight - 1, y)));
            const left = Math.min(...xs);
            const right = Math.max(...xs);
            const top = Math.min(...ys);
            const bottom = Math.max(...ys);
            state.points = [[left, top], [right, top], [right, bottom], [left, bottom]];
            syncRectangleDimensions();
            state.lastTouchedHandle = null;
        }
        renderSource();
        renderPreview();
    };

    const open = async (rowId) => {
        state.rowId = rowId;
        setEyedropperActive(false);
        state.canvasBackgroundIndex = 2;
        setCanvasBackground();
        errorBox().textContent = '';
        selectAlgorithm(null);
        selectMode('rectangle');
        loading().hidden = false;
        modal().hidden = false;
        document.body.classList.add('crop-open');
        try {
            const card = document.querySelector(`#row-${rowId}`);
            document.querySelector('#crop-title').textContent = `Crop Tool - ${card.dataset.imageFilename} (${card.dataset.imageWidth}x${card.dataset.imageHeight})`;
            const cardImage = card.querySelector('img');
            const image = new Image();
            image.src = `${cardImage.src.split('?')[0]}?v=${Date.now()}`;
            await image.decode();
            state.image = image;
            const buffer = document.createElement('canvas');
            buffer.width = image.naturalWidth;
            buffer.height = image.naturalHeight;
            const bufferContext = buffer.getContext('2d', { willReadFrequently: true });
            bufferContext.drawImage(image, 0, 0);
            state.imageData = bufferContext.getImageData(0, 0, buffer.width, buffer.height);
            state.points = [
                [0, 0],
                [image.naturalWidth - 1, 0],
                [image.naturalWidth - 1, image.naturalHeight - 1],
                [0, image.naturalHeight - 1],
            ];
            const fullQuad = state.points.map((point) => [...point]);
            const normalizedQuad = card.dataset.imageQuad;
            let isFullQuad = true;
            if (normalizedQuad) {
                try {
                    const quad = JSON.parse(normalizedQuad);
                    if (!Array.isArray(quad) || quad.length !== 4 || quad.some((point) => !Array.isArray(point) || point.length !== 2 || point.some((coordinate) => !Number.isFinite(coordinate)))) {
                        throw new Error('Invalid quad.');
                    }
                    state.points = quad.map(([x, y]) => [
                        x * (image.naturalWidth - 1),
                        y * (image.naturalHeight - 1),
                    ]);
                    isFullQuad = state.points.every((point, index) => (
                        Math.abs(point[0] - fullQuad[index][0]) < 1e-6
                        && Math.abs(point[1] - fullQuad[index][1]) < 1e-6
                    ));
                } catch {
                    state.points = fullQuad;
                }
            }
            selectMode(isFullQuad ? 'rectangle' : 'perspective');
        } catch (error) {
            errorBox().textContent = error.message;
        } finally {
            loading().hidden = true;
        }
    };

    const apply = async () => {
        if (!state.rowId) return;
        const button = document.querySelector('#crop-apply');
        button.disabled = true;
        errorBox().textContent = '';
        try {
            const response = await fetch(`/row/${state.rowId}/clip`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    points: state.points,
                    background: colorInput().value,
                    mode: state.mode,
                    output_width: state.mode === 'rectangle' ? Number.parseInt(outputWidthInput().value, 10) : null,
                    output_height: state.mode === 'rectangle' ? Number.parseInt(outputHeightInput().value, 10) : null,
                    resampling: 'lanczos',
                }),
            });
            const result = await response.json();
            if (!response.ok) throw new Error(result.detail || 'Could not apply crop.');
            const card = document.querySelector(`#row-${state.rowId}`);
            const cardImage = card.querySelector('img');
            cardImage.src = `${result.image_src}?v=${Date.now()}`;
            card.dataset.imageWidth = result.width;
            card.dataset.imageHeight = result.height;
            card.dataset.imageQuad = JSON.stringify([[0, 0], [1, 0], [1, 1], [0, 1]]);
            card.querySelector('.image-dimensions').innerHTML = `<span${result.width > 2000 ? ' class="filename-mismatch"' : ''}>${result.width}</span>x<span${result.height > 2000 ? ' class="filename-mismatch"' : ''}>${result.height}</span>`;
            close();
        } catch (error) {
            errorBox().textContent = error.message;
        } finally {
            button.disabled = false;
        }
    };

    const detect = async (algorithm, button) => {
        if (!state.rowId) return;
        if ((algorithm === 'hough' || algorithm === 'contour' || algorithm === 'llm') && state.mode === 'rectangle') {
            selectMode('perspective');
        }
        if (algorithm === 'full') {
            if (!state.image) return;
            state.points = [
                [0, 0],
                [state.image.naturalWidth - 1, 0],
                [state.image.naturalWidth - 1, state.image.naturalHeight - 1],
                [0, state.image.naturalHeight - 1],
            ];
            state.lastTouchedHandle = null;
            if (state.mode === 'rectangle') {
                syncRectangleDimensions();
            }
            selectAlgorithm('full');
            renderSource();
            bufferPreview();
            return;
        }
        const detectorButtons = document.querySelectorAll('[data-crop-detector]');
        detectorButtons.forEach((detectorButton) => { detectorButton.disabled = true; });
        errorBox().textContent = '';
        loading().textContent = algorithm === 'llm' ? 'Asking the VLM...' : 'Detecting edges...';
        loading().hidden = false;
        try {
            const response = await fetch(`/row/${state.rowId}/clip?algorithm=${algorithm}`);
            const responseText = await response.text();
            let result;
            try {
                result = JSON.parse(responseText);
            } catch {
                throw new Error(response.ok ? 'The server returned an invalid detector response.' : responseText || 'Could not detect crop boundaries.');
            }
            if (!response.ok) throw new Error(result.detail || 'Could not detect crop boundaries.');
            state.points = result.points;
            state.lastTouchedHandle = null;
            if (result.background) colorInput().value = result.background;
            selectAlgorithm(algorithm);
            renderSource();
            bufferPreview();
        } catch (error) {
            errorBox().textContent = error.message;
        } finally {
            loading().hidden = true;
            detectorButtons.forEach((detectorButton) => { detectorButton.disabled = false; });
        }
    };

    const initialize = () => {
        sourceCanvas().addEventListener('pointerdown', pointerDown);
        sourceCanvas().addEventListener('pointermove', pointerMove);
        sourceCanvas().addEventListener('pointerup', pointerUp);
        sourceCanvas().addEventListener('pointercancel', pointerUp);
        colorInput().addEventListener('input', bufferPreview);
        eyedropperButton().addEventListener('click', () => {
            setEyedropperActive(!state.eyedropperActive);
        });
        document.querySelector('#crop-canvas-background').addEventListener('click', cycleCanvasBackground);
        downsamplingSelect().addEventListener('change', () => {
            if (!downsamplingSelect().value) return;
            const [width, height] = targetDownsamplingSize(Number.parseFloat(downsamplingSelect().value));
            outputWidthInput().value = width;
            outputHeightInput().value = height;
            renderPreview();
        });
        document.querySelectorAll('[data-crop-mode]').forEach((button) => {
            button.addEventListener('click', () => selectMode(button.dataset.cropMode));
        });
        outputWidthInput().addEventListener('blur', () => {
            const width = Math.max(1, Number.parseInt(outputWidthInput().value, 10) || 1);
            outputWidthInput().value = width;
            outputHeightInput().value = Math.max(1, Math.round(width / state.rectangleAspect));
            downsamplingSelect().value = '';
            renderPreview();
        });
        outputHeightInput().addEventListener('blur', () => {
            const height = Math.max(1, Number.parseInt(outputHeightInput().value, 10) || 1);
            outputHeightInput().value = height;
            outputWidthInput().value = Math.max(1, Math.round(height * state.rectangleAspect));
            downsamplingSelect().value = '';
            renderPreview();
        });
        document.querySelector('#crop-close').addEventListener('click', close);
        document.querySelector('#crop-cancel').addEventListener('click', close);
        document.querySelector('#crop-apply').addEventListener('click', apply);
        document.querySelectorAll('[data-crop-detector]').forEach((button) => {
            button.addEventListener('click', () => detect(button.dataset.cropDetector, button));
        });
        window.addEventListener('resize', renderSource);
        document.addEventListener('pointerdown', (event) => {
            // Outside interactions retain their normal behavior while dismissing the eyedropper.
            if (state.eyedropperActive
                && !sourceCanvas().contains(event.target)
                && !eyedropperButton().contains(event.target)) {
                setEyedropperActive(false);
            }
        });
        document.addEventListener('keydown', (event) => {
            if (event.key === 'Escape' && state.eyedropperActive) {
                setEyedropperActive(false);
                return;
            }
            if (event.key === 'Escape' && !modal().hidden) {
                close();
                return;
            }
            nudgeLastPoint(event);
        });
    };

    return { initialize, open };
})();

document.addEventListener('input', markDirty);
document.addEventListener('change', markDirty);
document.addEventListener('change', (event) => {
    if (event.target.matches('[data-auto-save]')) {
        event.target.closest('form').requestSubmit();
    }
});
document.addEventListener('click', (event) => {
    const cropButton = event.target.closest('[data-crop-row]');
    if (cropButton) {
        cropEditor.open(cropButton.dataset.cropRow);
        return;
    }
    const acceptButton = event.target.closest('[data-accept-filename]');
    if (!acceptButton) {
        return;
    }
    const form = acceptButton.closest('form');
    const cleanFilename = form.querySelector('[name="clean_filename"]');
    cleanFilename.value = acceptButton.dataset.acceptFilename;
    form.requestSubmit();
});
document.addEventListener('DOMContentLoaded', () => {
    updateReviewState();
    cropEditor.initialize();
});
document.addEventListener('htmx:afterSwap', updateReviewState);
document.addEventListener('htmx:responseError', (event) => {
    if (event.detail.requestConfig.path.startsWith('/row/')) {
        const message = event.detail.xhr.responseText.trim() || 'Could not save changes.';
        window.alert(message);
    }
});
