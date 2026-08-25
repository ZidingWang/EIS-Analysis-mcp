import re

import numpy as np


_ELEMENT_RE = re.compile(r"^(CPE|R|C|L|W)([A-Za-z0-9_]*)$", re.IGNORECASE)


class CircuitSyntaxError(ValueError):
    """Raised when an ECM model string cannot be parsed."""


class Element(object):
    def __init__(self, token):
        match = _ELEMENT_RE.match(token)
        if not match:
            raise CircuitSyntaxError("Unsupported circuit element: %s" % token)
        self.kind = match.group(1).upper()
        suffix = match.group(2)
        self.name = self.kind + suffix

    def parameter_names(self):
        if self.kind == "CPE":
            return [self.name + "_Q", self.name + "_n"]
        return [self.name]

    def impedance(self, omega, params):
        omega = np.asarray(omega, dtype=float)
        jw = 1j * omega

        if self.kind == "R":
            return np.ones_like(omega, dtype=complex) * params[self.name]
        if self.kind == "C":
            c_value = max(float(params[self.name]), 1e-30)
            return 1.0 / (jw * c_value)
        if self.kind == "L":
            return jw * params[self.name]
        if self.kind == "W":
            # Semi-infinite Warburg: Z = sigma / sqrt(j * omega).
            return params[self.name] / np.sqrt(jw)
        if self.kind == "CPE":
            q_value = max(float(params[self.name + "_Q"]), 1e-30)
            n_value = float(params[self.name + "_n"])
            return 1.0 / (q_value * np.power(jw, n_value))

        raise CircuitSyntaxError("Unsupported element kind: %s" % self.kind)

    def collect_elements(self):
        return [self]


class Parallel(object):
    def __init__(self, children):
        if len(children) < 2:
            raise CircuitSyntaxError("Parallel block needs at least two elements")
        self.children = children

    def parameter_names(self):
        names = []
        for child in self.children:
            names.extend(child.parameter_names())
        return names

    def impedance(self, omega, params):
        admittance = np.zeros_like(np.asarray(omega, dtype=float), dtype=complex)
        for child in self.children:
            z_child = child.impedance(omega, params)
            admittance = admittance + 1.0 / z_child
        return 1.0 / admittance

    def collect_elements(self):
        elements = []
        for child in self.children:
            elements.extend(child.collect_elements())
        return elements


class Series(object):
    def __init__(self, children):
        if not children:
            raise CircuitSyntaxError("Circuit is empty")
        self.children = children

    def parameter_names(self):
        names = []
        for child in self.children:
            names.extend(child.parameter_names())
        return names

    def impedance(self, omega, params):
        total = np.zeros_like(np.asarray(omega, dtype=float), dtype=complex)
        for child in self.children:
            total = total + child.impedance(omega, params)
        return total

    def collect_elements(self):
        elements = []
        for child in self.children:
            elements.extend(child.collect_elements())
        return elements


class CircuitModel(object):
    """Parsed ECM model.

    Supported examples:
      R0
      R0-(R1||C1)
      R0-(R1||CPE1)-(R2||C2)-W1
      R0+p(R1,CPE1)+p(R2,C2)
    """

    def __init__(self, expression):
        self.expression = expression
        self.root = parse_circuit(expression)
        self.parameter_names = _unique(self.root.parameter_names())
        self.elements = self.root.collect_elements()

    def impedance(self, freq_hz, params):
        omega = 2.0 * np.pi * np.asarray(freq_hz, dtype=float)
        return self.root.impedance(omega, params)

    def guess_parameters(self, freq_hz, z):
        freq_hz = np.asarray(freq_hz, dtype=float)
        z = np.asarray(z, dtype=complex)
        real_z = np.real(z)
        positive_freq = freq_hz[freq_hz > 0]

        if positive_freq.size:
            f_mid = float(np.exp(np.mean(np.log(positive_freq))))
        else:
            f_mid = 1.0

        real_min = float(np.nanmin(real_z)) if real_z.size else 0.0
        real_max = float(np.nanmax(real_z)) if real_z.size else 1.0
        real_min = max(real_min, 0.0)
        real_range = max(real_max - real_min, 0.0)
        scale_floor = max(abs(real_max), abs(real_min), 1.0) * 1e-12
        span = max(real_range, 0.1 * abs(real_max), 0.1 * abs(real_min), scale_floor)

        r_params = [p for p in self.parameter_names if self.parameter_kind(p) == "R"]
        non_ohmic_count = max(len(r_params) - 1, 1)

        guesses = {}
        for name in self.parameter_names:
            kind = self.parameter_kind(name)
            if kind == "R":
                lower_name = name.lower()
                if lower_name in ("r0", "rs", "rinf", "r_infty"):
                    guesses[name] = max(real_min, 1e-6)
                else:
                    guesses[name] = max(span / non_ohmic_count, 1e-6)
            elif kind == "C":
                guesses[name] = 1.0 / (2.0 * np.pi * f_mid * max(span, 1e-6))
            elif kind == "L":
                guesses[name] = 1e-8
            elif kind == "W":
                guesses[name] = max(span * np.sqrt(2.0 * np.pi * f_mid), 1e-6)
            elif kind == "CPE_Q":
                guesses[name] = 1.0 / (2.0 * np.pi * f_mid * max(span, 1e-6))
            elif kind == "CPE_N":
                guesses[name] = 0.85
        return guesses

    def default_bounds(self, freq_hz, z):
        real_z = np.real(np.asarray(z, dtype=complex))
        real_max = float(np.nanmax(np.abs(real_z))) if real_z.size else 1.0
        scale = max(real_max, 1e-3)

        bounds = {}
        for name in self.parameter_names:
            kind = self.parameter_kind(name)
            if kind == "R":
                bounds[name] = (0.0, max(1e3 * scale, 1.0))
            elif kind == "C":
                bounds[name] = (1e-12, 1e4)
            elif kind == "L":
                bounds[name] = (0.0, 1e3)
            elif kind == "W":
                bounds[name] = (0.0, 1e6)
            elif kind == "CPE_Q":
                bounds[name] = (1e-12, 1e4)
            elif kind == "CPE_N":
                bounds[name] = (0.2, 1.0)
        return bounds

    def parameter_kind(self, name):
        if name.endswith("_Q"):
            return "CPE_Q"
        if name.endswith("_n"):
            return "CPE_N"
        for element in self.elements:
            if name == element.name:
                return element.kind
        raise KeyError("Unknown parameter: %s" % name)


def canonicalize_series_process_parameters(circuit, parameters):
    """Order interchangeable series RC/RQ processes from high to low frequency.

    Series branches commute, so optimizers may return the same physical
    processes under different R/CPE indices. Canonical ordering makes exported
    parameters directly comparable between runs and with ZView circuit labels.
    """
    original = dict(parameters)
    canonical = dict(parameters)
    root = circuit.root
    if not isinstance(root, Series):
        return canonical

    groups = {}
    for child in root.children:
        descriptor = _relaxation_branch_descriptor(child, original)
        if descriptor is not None:
            groups.setdefault(descriptor["kind"], []).append(descriptor)

    for branches in groups.values():
        if len(branches) < 2:
            continue
        sources = sorted(branches, key=lambda item: item["frequency_hz"], reverse=True)
        for target, source in zip(branches, sources):
            canonical[target["r_name"]] = original[source["r_name"]]
            canonical[target["reactive_names"][0]] = original[source["reactive_names"][0]]
            if len(target["reactive_names"]) == 2:
                canonical[target["reactive_names"][1]] = original[source["reactive_names"][1]]
    return canonical


def _relaxation_branch_descriptor(node, parameters):
    if not isinstance(node, Parallel) or len(node.children) != 2:
        return None
    if not all(isinstance(child, Element) for child in node.children):
        return None

    resistor = next((child for child in node.children if child.kind == "R"), None)
    reactive = next((child for child in node.children if child.kind in ("C", "CPE")), None)
    if resistor is None or reactive is None:
        return None

    resistance = float(parameters[resistor.name])
    if resistance <= 0:
        return None
    if reactive.kind == "C":
        reactive_names = [reactive.name]
        value = float(parameters[reactive.name])
        if value <= 0:
            return None
        frequency_hz = 1.0 / (2.0 * np.pi * resistance * value)
    else:
        reactive_names = [reactive.name + "_Q", reactive.name + "_n"]
        q_value = float(parameters[reactive_names[0]])
        n_value = float(parameters[reactive_names[1]])
        if q_value <= 0 or n_value <= 0:
            return None
        log_frequency = -np.log(resistance * q_value) / n_value - np.log(2.0 * np.pi)
        frequency_hz = float(np.exp(np.clip(log_frequency, -700.0, 700.0)))

    return {
        "kind": reactive.kind,
        "r_name": resistor.name,
        "reactive_names": reactive_names,
        "frequency_hz": float(frequency_hz),
    }


def parse_circuit(expression):
    text = (expression or "").replace(" ", "")
    if not text:
        raise CircuitSyntaxError("Circuit expression is empty")
    return _parse_series(text)


def _parse_series(text):
    tokens = _split_top_level(text, ["+", "-"])
    if len(tokens) == 1:
        return _parse_term(tokens[0])
    return Series([_parse_term(token) for token in tokens])


def _parse_term(text):
    if not text:
        raise CircuitSyntaxError("Empty circuit term")

    if _is_wrapped(text):
        inner = text[1:-1]
        parallel_tokens = _split_parallel(inner)
        if len(parallel_tokens) > 1:
            return Parallel([_parse_series(token) for token in parallel_tokens])
        return _parse_series(inner)

    if text.lower().startswith("p(") and text.endswith(")"):
        inner = text[2:-1]
        tokens = _split_top_level(inner, [","])
        return Parallel([_parse_series(token) for token in tokens])

    parallel_tokens = _split_parallel(text)
    if len(parallel_tokens) > 1:
        return Parallel([_parse_series(token) for token in parallel_tokens])

    return Element(text)


def _split_parallel(text):
    tokens = []
    depth = 0
    start = 0
    i = 0
    while i < len(text):
        char = text[i]
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth < 0:
                raise CircuitSyntaxError("Unmatched ')' in %s" % text)
        elif depth == 0 and text[i : i + 2] == "||":
            tokens.append(text[start:i])
            i += 2
            start = i
            continue
        i += 1

    if depth != 0:
        raise CircuitSyntaxError("Unmatched '(' in %s" % text)
    tokens.append(text[start:])
    return [token for token in tokens if token]


def _split_top_level(text, separators):
    tokens = []
    depth = 0
    start = 0
    for i, char in enumerate(text):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth < 0:
                raise CircuitSyntaxError("Unmatched ')' in %s" % text)
        elif depth == 0 and char in separators:
            tokens.append(text[start:i])
            start = i + 1
    if depth != 0:
        raise CircuitSyntaxError("Unmatched '(' in %s" % text)
    tokens.append(text[start:])
    return [token for token in tokens if token]


def _is_wrapped(text):
    if not (text.startswith("(") and text.endswith(")")):
        return False
    depth = 0
    for i, char in enumerate(text):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0 and i != len(text) - 1:
                return False
    return depth == 0


def _unique(values):
    seen = set()
    output = []
    for value in values:
        if value not in seen:
            output.append(value)
            seen.add(value)
    return output
