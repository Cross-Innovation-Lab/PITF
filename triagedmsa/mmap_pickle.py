from __future__ import annotations

import math
import pickle
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class _FileBlock:
    path: Path
    offset: int
    size: int

    def array(self, dtype, shape, order):
        dtype = np.dtype(dtype)
        if dtype.hasobject or math.prod(shape) * dtype.itemsize != self.size:
            raise pickle.UnpicklingError('Invalid numeric NumPy payload')
        return np.memmap(self.path, dtype=dtype, mode='r', offset=self.offset,
                         shape=shape, order=order)


class _ArrayProxy:
    def __setstate__(self, state):
        version, shape, dtype, fortran, raw = state
        if version != 1:
            raise pickle.UnpicklingError(f'Unsupported NumPy state version: {version}')
        if isinstance(raw, _FileBlock):
            self.value = raw.array(dtype, shape, 'F' if fortran else 'C')
        else:
            self.value = np.empty(0)
            self.value.__setstate__(state)


def _reconstruct(subtype, shape, dtype):
    if subtype is not np.ndarray:
        raise pickle.UnpicklingError('Only plain NumPy arrays are supported')
    return _ArrayProxy()


def _frombuffer(raw, dtype, shape, order):
    if isinstance(raw, _FileBlock):
        return raw.array(dtype, shape, order)
    return np.frombuffer(raw, dtype=dtype).reshape(shape, order=order)


class _MappedUnpickler(pickle._Unpickler):
    dispatch = pickle._Unpickler.dispatch.copy()

    def __init__(self, stream, path, threshold):
        super().__init__(stream)
        self.stream, self.path, self.threshold = stream, path, threshold
        self.file_size = path.stat().st_size
        self.frame_end = 0

    def find_class(self, module, name):
        if module in {'numpy.core.multiarray', 'numpy._core.multiarray'} and name == '_reconstruct':
            return _reconstruct
        if module in {'numpy.core.numeric', 'numpy._core.numeric'} and name == '_frombuffer':
            return _frombuffer
        return super().find_class(module, name)

    def load_frame(self):
        if self.stream.tell() - 1 < self.frame_end:
            raise pickle.UnpicklingError('New frame begins before the previous frame ends')
        size, = struct.unpack('<Q', self.read(8))
        self.frame_end = self.stream.tell() + size
        if self.frame_end > self.file_size:
            raise pickle.UnpicklingError('Truncated frame')

    def _payload(self, size, mutable=False):
        if self.stream.tell() + size > self.file_size:
            raise pickle.UnpicklingError('Truncated byte payload')
        if size >= self.threshold:
            block = _FileBlock(self.path, self.stream.tell(), size)
            self.stream.seek(size, 1)
            self.append(block)
        else:
            raw = self.read(size)
            self.append(bytearray(raw) if mutable else raw)

    def load_binbytes(self):
        self._payload(struct.unpack('<I', self.read(4))[0])

    def load_binbytes8(self):
        self._payload(struct.unpack('<Q', self.read(8))[0])

    def load_bytearray8(self):
        self._payload(struct.unpack('<Q', self.read(8))[0], mutable=True)

    dispatch[pickle.FRAME[0]] = load_frame
    dispatch[pickle.BINBYTES[0]] = load_binbytes
    dispatch[pickle.BINBYTES8[0]] = load_binbytes8
    dispatch[pickle.BYTEARRAY8[0]] = load_bytearray8


def _unwrap(value):
    if isinstance(value, _ArrayProxy):
        return value.value
    if isinstance(value, _FileBlock):
        raise pickle.UnpicklingError('Large non-array binary value is unsupported')
    if isinstance(value, dict):
        return {k: _unwrap(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_unwrap(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_unwrap(v) for v in value)
    return value


def load_mmap_pickle(path, threshold=65536):
    path = Path(path).resolve()
    before = path.stat()
    with path.open('rb') as stream:
        if stream.read(2) not in (b'\x80\x04', b'\x80\x05'):
            raise ValueError('Memory-mapped loading requires pickle protocol 4 or 5')
        stream.seek(0)
        data = _unwrap(_MappedUnpickler(stream, path, threshold).load())
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError('Dataset changed while its metadata was being read')
    return data
