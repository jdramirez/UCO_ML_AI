"""Utilidades de texto compartidas por los clasificadores."""
import re
import unicodedata

import numpy as np


def normalizar(texto):
    texto = str(texto).lower()
    texto = "".join(c for c in unicodedata.normalize("NFD", texto) if unicodedata.category(c) != "Mn")
    return re.sub(r"(.)\1{2,}", r"\1", texto)  # "muuuy" -> "muy"


def tokens(texto):
    return re.findall(r"[a-zñ0-9]+", normalizar(texto))


def softmax(z):
    z = np.asarray(z, dtype=float)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)
