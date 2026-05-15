"""
ops.py

Defines the three order-53 binary operations used in the study, plus
exhaustive verification of their algebraic properties (associativity,
commutativity, Latin-square property). Every property claimed in the
paper about these operations is checked here computationally, not just
asserted -- run this file directly to reproduce the verification.

    Z53: a (+) b = (a + b) mod 53                  [associative, commutative]
    Q53: a (+) b = (a + sigma(b)) mod 53            [non-assoc, non-comm]
         where sigma swaps 51 and 52, fixes everything else
    C53: a (+) b = k * (a + b) mod 53               [non-assoc, COMMUTATIVE]
         for k not in {0, 1} -- this is the new operation that isolates
         associativity from commutativity (see Phase-1 Priority #1).
"""

import itertools
import numpy as np

N = 53


# ---------------------------------------------------------------------------
# Operation definitions
# ---------------------------------------------------------------------------

def op_Z53(a: int, b: int) -> int:
    """Standard modular addition. Associative, commutative group."""
    return (a + b) % N


def _sigma(x: int) -> int:
    """Transposition swapping 51 and 52, fixing everything else."""
    if x == 51:
        return 52
    if x == 52:
        return 51
    return x


def op_Q53(a: int, b: int) -> int:
    """a (+) b = a + sigma(b) mod 53. Non-associative, non-commutative."""
    return (a + _sigma(b)) % N


def make_op_C53(k: int):
    """
    Returns the operation a (+) b = k*(a+b) mod 53.

    This is commutative for any k (since a+b=b+a), a Latin square for any
    k != 0 mod 53 (since multiplication by k is then a bijection on Z_53,
    53 being prime), and non-associative for any k not in {0, 1}:

        (a*b)*c = k^2(a+b) + kc
        a*(b*c) = ka + k^2(b+c)

    Associativity for all a,b,c requires k^2 = k, i.e. k(k-1) = 0 mod 53.
    Since 53 is prime this forces k in {0, 1}. So any k in {2, ..., 51}
    gives a genuine commutative, non-associative Latin square of order 53.
    """
    assert 2 <= k <= N - 2, "k must avoid the associative cases {0, 1, ...} " \
                            "and their negatives; use 2 <= k <= 51"

    def op_C53(a: int, b: int) -> int:
        return (k * (a + b)) % N

    op_C53.k = k
    op_C53.__name__ = f"op_C53_k{k}"
    return op_C53


def _get_fixed_P(n: int = N, seed: int = 42):
    rng = np.random.RandomState(seed)
    return rng.permutation(n)

_P = _get_fixed_P(N)
_P_inv = np.zeros_like(_P)
for i in range(N):
    _P_inv[_P[i]] = i

def op_L53(a: int, b: int) -> int:
    """Random unstructured commutative Latin square (non-associative)."""
    return (_P[a] + _P[b]) % N

def op_L53_assoc(a: int, b: int) -> int:
    """Random unstructured commutative Latin square (associative, isomorphic to Z53)."""
    return _P_inv[(_P[a] + _P[b]) % N]


# ---------------------------------------------------------------------------
# Exhaustive verification
# ---------------------------------------------------------------------------

def is_latin_square(op, n: int = N) -> bool:
    """Every row and column of the operation table must be a permutation."""
    for a in range(n):
        row = {op(a, b) for b in range(n)}
        if len(row) != n:
            return False
    for b in range(n):
        col = {op(a, b) for a in range(n)}
        if len(col) != n:
            return False
    return True


def is_commutative(op, n: int = N):
    """Returns (bool, first_violation_or_None)."""
    for a in range(n):
        for b in range(a + 1, n):
            if op(a, b) != op(b, a):
                return False, (a, b)
    return True, None


def is_associative(op, n: int = N):
    """
    Exhaustive check over all n^3 ordered triples. Returns
    (bool, first_violation_or_None) where a violation is
    (a, b, c, lhs, rhs) with lhs = (a*b)*c, rhs = a*(b*c).
    """
    for a, b, c in itertools.product(range(n), repeat=3):
        lhs = op(op(a, b), c)
        rhs = op(a, op(b, c))
        if lhs != rhs:
            return False, (a, b, c, lhs, rhs)
    return True, None


def verify_operation(op, name: str, n: int = N, verbose: bool = True):
    """Runs the full exhaustive verification battery on one operation."""
    latin = is_latin_square(op, n)
    comm, comm_viol = is_commutative(op, n)
    assoc, assoc_viol = is_associative(op, n)

    report = {
        "name": name,
        "order": n,
        "latin_square": latin,
        "commutative": comm,
        "commutative_first_violation": comm_viol,
        "associative": assoc,
        "associative_first_violation": assoc_viol,
    }

    if verbose:
        print(f"--- {name} (order {n}) ---")
        print(f"  Latin square : {latin}")
        print(f"  Commutative  : {comm}"
              + ("" if comm else f"  (first violation at {comm_viol})"))
        print(f"  Associative  : {assoc}"
              + ("" if assoc else f"  (first violation at {assoc_viol})"))
    return report


def get_operation_table(op, n: int = N) -> np.ndarray:
    """Materialises the n x n Cayley table as a numpy array."""
    table = np.zeros((n, n), dtype=np.int64)
    for a in range(n):
        for b in range(n):
            table[a, b] = op(a, b)
    return table


# ---------------------------------------------------------------------------
# Registry of operations used in the confirmatory sweep
# ---------------------------------------------------------------------------

# k=2 is the default C53 twist: not close to the associative fixed points
# {0, 1} or to -1 = 52 (which would make the operation close to an
# "antipodal" involution-like structure). Any k in {2,...,51} works; k=2
# is simplest to explain and verify by hand (see docstring above).
C53_DEFAULT_K = 2

OPERATIONS = {
    "Z53": op_Z53,
    "Q53": op_Q53,
    "C53": make_op_C53(C53_DEFAULT_K),
}


if __name__ == "__main__":
    print(f"Exhaustive verification over all {N}^3 = {N**3} ordered triples "
          f"per operation.\n")
    reports = []
    for name, op in OPERATIONS.items():
        reports.append(verify_operation(op, name))
        print()

    # Sanity check: confirm the 2x2 design is what we intend.
    expected = {
        "Z53": dict(latin_square=True, commutative=True, associative=True),
        "Q53": dict(latin_square=True, commutative=False, associative=False),
        "C53": dict(latin_square=True, commutative=True, associative=False),
        "L53": dict(latin_square=True, commutative=True, associative=False),
        "L53_assoc": dict(latin_square=True, commutative=True, associative=True),
    }
    ok = True
    for r in reports:
        exp = expected[r["name"]]
        for key, val in exp.items():
            if r[key] != val:
                ok = False
                print(f"MISMATCH: {r['name']}.{key} expected {val}, "
                      f"got {r[key]}")
    print("ALL EXPECTED PROPERTIES CONFIRMED." if ok else "VERIFICATION FAILED.")
