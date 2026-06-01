"""
combinatorics.py
================
Generic combinatorial enumeration of charge-neutral transition-metal complexes.

This module provides the enumeration logic that was previously hard-coded in
generate_ni_complexes.py.  Everything here is metal- and ligand-agnostic; the
caller supplies pools of ligands, bridges, oxidation states and the library
does the rest.

Public API
----------
    enumerate_complexes(metal, ...)   -> iterator of (Molecule, metadata-dict)

The iterator yields every geometrically valid, charge-neutral complex across
the requested nuclearity (monomer / dimer / trimer) without any metal-specific
special-casing in the caller.

Design notes
------------
* Charge balance is purely arithmetic: oxidation state + sum(ligand charges) == 0.
* Geometry is auto-inferred from CN; callers may override via geometry_for_cn.
* Multi-bridge trimers use the pre-validated MULTI_BRIDGE_CASES table, which
  is also accessible and extensible by callers.
* All heavy combinatorial loops live here so generate_*.py files stay thin.
"""

from __future__ import annotations

from collections import Counter
from itertools import combinations_with_replacement
from pathlib import Path
from typing import (
    Any, Dict, Iterator, List, Optional, Sequence, Tuple
)

from molbuilder.api import build, build_isomers, dimer, trimer
from molbuilder.core.validation import validate as _validate_mol


def _passes_hard_validation(mol) -> bool:
    """Return True if mol has no hard-error geometry clashes."""
    try:
        result = _validate_mol(mol)
        return result.passed
    except Exception:
        return True   # don't drop on validator errors
from molbuilder.core.geometry import infer_geometry
from molbuilder.ligands.library import get_ligand

# -- ligand metadata helpers ---------------------------------------------------

def _ligand_charge(name: str) -> int:
    try:
        return get_ligand(name)["charge"]
    except KeyError:
        return 0


def _ligand_cn(name: str) -> int:
    try:
        return get_ligand(name)["denticity"]
    except KeyError:
        return 1


# -- geometry helpers ----------------------------------------------------------

_DEFAULT_GEOM_FOR_CN: Dict[int, str] = {
    3: "tp", 4: "tet", 5: "sqpy", 6: "oct", 7: "pbp",
}
_EXTRA_GEOM: Dict[int, List[str]] = {4: ["sqp"], 5: ["tbp"]}


def _geometries_for_cn(cn: int,
                        geometry_for_cn: Optional[Dict[int, str]] = None,
                        extra_geom: Optional[Dict[int, List[str]]] = None,
                        ) -> List[str]:
    gfc   = geometry_for_cn or _DEFAULT_GEOM_FOR_CN
    extra = extra_geom      or _EXTRA_GEOM
    base  = [gfc.get(cn, infer_geometry(cn))]
    return base + extra.get(cn, [])


# -- label helpers -------------------------------------------------------------

def combo_label(ligands: Sequence[str]) -> str:
    c = Counter(ligands)
    return "_".join(f"{l.replace(':', '')}{n}" for l, n in sorted(c.items()))


def safe(s: str) -> str:
    return s.replace(":", "-").replace("/", "-").replace(" ", "_")


def _ox_label(metal: str, ox: int) -> str:
    roman = {2: "II", 3: "III", 4: "IV", 1: "I", 5: "V"}
    return f"{metal}{roman.get(ox, str(ox))}"


# -- result row ----------------------------------------------------------------

def _make_row(mol, structure_type, metal, ox, cn, geom,
              lig_combo, bridge, n_bridges, arrangement, isomer_label,
              filename) -> Dict[str, Any]:
    return {
        "structure":    structure_type,
        "metal":        metal,
        "ox":           ox,
        "ox_label":     _ox_label(metal, ox),
        "cn":           cn,
        "geometry":     geom,
        "ligand_combo": lig_combo,
        "bridge":       bridge or "",
        "n_bridges":    n_bridges,
        "arrangement":  arrangement or "",
        "isomer":       isomer_label,
        "formula":      mol.formula,
        "charge":       mol.charge,
        "n_atoms":      mol.num_atoms(),
        "filename":     str(filename),
    }


# -- bridge pairs per topology --------------------------------------------------

_BRIDGE_PAIRS = {"linear": 2, "triangular": 3}


# -- MULTI_BRIDGE_CASES --------------------------------------------------------
# Each entry: (arrangement, n_bridges_per_pair, bridge, ox, terminal, label_suffix)
#
# Inclusion criteria
# ------------------
# 1. Charge-neutral: 3*ox + 3*tc + n_bridge_pairs*nbpp*bridge_charge == 0
# 2. Geometrically valid: the coordinated +/-alpha tilt scheme in trimer() keeps all
#    inter-edge O***O distances above the hard clash threshold of 1.98 Angstrom.
#
# Bridge-type-aware tilt angles (alpha, encoded in api.py _TRIANGULAR_DOUBLE_BRIDGE_ALPHA):
#   mu-HCOO : alpha ~= 35deg  ->  min inter-edge O***O ~= 2.20 Angstrom  (O-C-O bar geometry)
#   mu-OH   : alpha ~= 70deg  ->  min inter-edge O***O ~= 2.36 Angstrom  (single-atom bridge)
#
# Triangular nbpp=3 is EXCLUDED: packing 6 bridging donors per metal at
# Ni-Ni ~= 3.2-3.6 Angstrom produces irreducible O***O ~= 1.3 Angstrom regardless of tilt angle.

MULTI_BRIDGE_CASES: List[Tuple] = [
    # (arrangement,   nbpp, bridge,     ox,  terminal,   suffix)
    # -- linear, 3 bridges per pair, NiII --------------------------------------
    ("linear",      3, "mu-OH",   2, [],         "triplebridge"),
    ("linear",      3, "mu-HCOO", 2, [],         "triplebridge"),
    # -- triangular, 2 bridges per pair, NiII ----------------------------------
    # 3*Ni(II) + 3_pairs * 2_bridges * (-1) = 6 - 6 = 0 OK   CN=4 per Ni (tet)
    ("triangular",  2, "mu-HCOO", 2, [],         "doublebridge"),
    ("triangular",  2, "mu-OH",   2, [],         "doublebridge"),
]


# -- MONOMERS ------------------------------------------------------------------

def enumerate_monomers(
    metal: str,
    ox_states: Sequence[int],
    mono_ligands: Sequence[str],
    bi_ligands: Sequence[str] = (),
    cn_range: Tuple[int, int] = (3, 7),
    geometry_for_cn: Optional[Dict[int, str]] = None,
    extra_geom: Optional[Dict[int, List[str]]] = None,
    output_root: Optional[Path] = None,
    verbose: bool = True,
) -> Iterator[Tuple[Any, Dict[str, Any]]]:
    """Yield (Molecule, row_dict) for every charge-neutral monomer."""
    cn_min, cn_max = cn_range
    mono_charge = {l: _ligand_charge(l) for l in mono_ligands}
    bi_charge   = {l: _ligand_charge(l) for l in bi_ligands}
    bi_cn_map   = {l: _ligand_cn(l)     for l in bi_ligands}

    for ox in ox_states:
        oxl = _ox_label(metal, ox)

        # pure monodentate
        for cn in range(cn_min, cn_max + 1):
            for combo in combinations_with_replacement(mono_ligands, cn):
                if ox + sum(mono_charge[l] for l in combo) != 0:
                    continue
                for geom in _geometries_for_cn(cn, geometry_for_cn, extra_geom):
                    try:
                        iso_list = build_isomers(metal, ox=ox,
                                                 ligands=list(combo), geometry=geom)
                    except Exception as e:
                        if verbose:
                            print(f"  FAIL monomer {oxl} CN{cn} {geom} {combo}: {e}")
                        continue
                    cl = combo_label(combo)
                    for mol in iso_list:
                        label = getattr(mol, "label", "only")
                        path  = _poscar_path(output_root, "monomer", oxl, cn,
                                             f"{safe(cl)}_{geom}_{safe(label)}.POSCAR")
                        row = _make_row(mol, "monomer", metal, ox, cn, geom,
                                        cl, None, 0, None, label, path)
                        v_result = _passes_hard_validation(mol)
                        if verbose:
                            status = "OK" if v_result else "FAIL (clash)"
                            print(f"  {status} monomer {oxl} CN{cn} {geom:5s}  "
                                  f"{cl:35s}  {label:10s}  {mol.formula}")
                        if not v_result:
                            continue
                        yield mol, row

        # bidentate + monodentate mixes
        for bi_name in bi_ligands:
            b_cn  = bi_cn_map[bi_name]
            b_chg = bi_charge[bi_name]
            for n_bi in range(1, (cn_max // b_cn) + 1):
                for n_mono in range(0, cn_max - n_bi * b_cn + 1):
                    cn = n_bi * b_cn + n_mono
                    if not (cn_min <= cn <= cn_max):
                        continue
                    mono_iter = (combinations_with_replacement(mono_ligands, n_mono)
                                 if n_mono > 0 else [()])
                    for mono_combo in mono_iter:
                        lc = n_bi * b_chg + sum(mono_charge[l] for l in mono_combo)
                        if ox + lc != 0:
                            continue
                        full = [bi_name] * n_bi + list(mono_combo)
                        for geom in _geometries_for_cn(cn, geometry_for_cn, extra_geom):
                            try:
                                result = build(metal, ox=ox, ligands=full, geometry=geom)
                            except Exception as e:
                                if verbose:
                                    print(f"  FAIL monomer {oxl} CN{cn} {geom} {full}: {e}")
                                continue
                            mols = result if isinstance(result, list) else [result]
                            cl = combo_label(full)
                            for mol in mols:
                                label = getattr(mol, "label", "only")
                                path  = _poscar_path(output_root, "monomer", oxl, cn,
                                                     f"{safe(cl)}_{geom}_{safe(label)}.POSCAR")
                                row = _make_row(mol, "monomer", metal, ox, cn, geom,
                                                cl, None, 0, None, label, path)
                                if not _passes_hard_validation(mol):
                                    continue
                                if verbose:
                                    print(f"  OK monomer {oxl} CN{cn} {geom:5s}  "
                                          f"{cl:35s}  {label:10s}  {mol.formula}")
                                yield mol, row


# -- DIMERS --------------------------------------------------------------------

def enumerate_dimers(
    metal: str,
    ox_states: Sequence[int],
    mono_ligands: Sequence[str],
    bridge_ligands: Sequence[str],
    bi_ligands: Sequence[str] = (),
    cn_range: Tuple[int, int] = (3, 7),
    max_bridges: Optional[int] = None,
    geometry_for_cn: Optional[Dict[int, str]] = None,
    output_root: Optional[Path] = None,
    verbose: bool = True,
) -> Iterator[Tuple[Any, Dict[str, Any]]]:
    """Yield (Molecule, row_dict) for every charge-neutral dimer.

    max_bridges defaults to cn_range[1] so that fully-bridged dimers such as
    Ni2(mu-HCOO)4 (n=4, no terminals, CN=4) are always included.
    Pass an explicit integer to cap earlier.
    """
    cn_min, cn_max = cn_range
    _max_b        = max_bridges if max_bridges is not None else cn_max
    mono_charge   = {l: _ligand_charge(l) for l in mono_ligands}
    bridge_charge = {b: _ligand_charge(b) for b in bridge_ligands}
    bi_charge     = {l: _ligand_charge(l) for l in bi_ligands}
    bi_cn_map     = {l: _ligand_cn(l)     for l in bi_ligands}

    for ox in ox_states:
        oxl = _ox_label(metal, ox)
        for bridge in bridge_ligands:
            bc = bridge_charge[bridge]
            for nb in range(1, _max_b + 1):

                # pure monodentate terminals
                for n_term in range(0, cn_max - nb + 1):
                    mono_iter = (combinations_with_replacement(mono_ligands, n_term)
                                 if n_term > 0 else [()])
                    for terminal in mono_iter:
                        tc  = sum(mono_charge[l] for l in terminal)
                        if 2 * ox + 2 * tc + nb * bc != 0:
                            continue
                        cn = n_term + nb
                        if not (cn_min <= cn <= cn_max):
                            continue
                        try:
                            mol = dimer(metal, ox=ox, terminal=list(terminal),
                                        bridge=bridge, n=nb)
                        except Exception:
                            continue
                        if mol.charge != 0:
                            continue
                        geom = _DEFAULT_GEOM_FOR_CN.get(cn, infer_geometry(cn))
                        cl   = combo_label(list(terminal))
                        path = _poscar_path(output_root, "dimer", oxl, cn,
                                            f"{safe(cl)}_{nb}x{safe(bridge)}.POSCAR")
                        row = _make_row(mol, "dimer", metal, ox, cn, geom,
                                        cl, bridge, nb, None, "only", path)
                        if verbose:
                            print(f"  OK dimer  {oxl} CN{cn}  {nb}x{bridge:10s}  "
                                  f"term={list(terminal)}  {mol.formula}")
                        yield mol, row

                # bidentate terminals
                for bi_name in bi_ligands:
                    b_cn  = bi_cn_map[bi_name]
                    b_chg = bi_charge[bi_name]
                    for n_bi in range(1, 3):
                        for n_mono in range(0, cn_max - nb - n_bi * b_cn + 1):
                            cn = nb + n_bi * b_cn + n_mono
                            if not (cn_min <= cn <= cn_max):
                                continue
                            mono_iter = (combinations_with_replacement(
                                             mono_ligands, n_mono)
                                         if n_mono > 0 else [()])
                            for mono_combo in mono_iter:
                                tc = (n_bi * b_chg +
                                      sum(mono_charge[l] for l in mono_combo))
                                if 2 * ox + 2 * tc + nb * bc != 0:
                                    continue
                                terminal = [bi_name] * n_bi + list(mono_combo)
                                try:
                                    mol = dimer(metal, ox=ox, terminal=terminal,
                                                bridge=bridge, n=nb)
                                except Exception:
                                    continue
                                if mol.charge != 0:
                                    continue
                                geom = _DEFAULT_GEOM_FOR_CN.get(cn, infer_geometry(cn))
                                cl   = combo_label(terminal)
                                path = _poscar_path(output_root, "dimer", oxl, cn,
                                                    f"{safe(cl)}_{nb}x{safe(bridge)}.POSCAR")
                                row = _make_row(mol, "dimer", metal, ox, cn, geom,
                                                cl, bridge, nb, None, "only", path)
                                if verbose:
                                    print(f"  OK dimer  {oxl} CN{cn}  "
                                          f"{nb}x{bridge:10s}  "
                                          f"term={terminal}  {mol.formula}")
                                yield mol, row


# -- HETEROLEPTIC DIMERS -------------------------------------------------------

def enumerate_heteroleptic_dimers(
    metal: str,
    ox_states: Sequence[int],
    mono_ligands: Sequence[str],
    bridge_ligands: Sequence[str],
    bi_ligands: Sequence[str] = (),
    cn_range: Tuple[int, int] = (3, 7),
    max_bridges: Optional[int] = None,
    max_terminals_per_metal: Optional[int] = 2,
    geometry_for_cn: Optional[Dict[int, str]] = None,
    output_root: Optional[Path] = None,
    verbose: bool = True,
) -> Iterator[Tuple[Any, Dict[str, Any]]]:
    """
    Yield (Molecule, row_dict) for every charge-neutral *heteroleptic* dimer --
    i.e. dimers where the two metal centres carry *different* terminal ligand sets.

    The symmetric (homotopic) cases are deliberately excluded here; they are
    already covered by enumerate_dimers().

    A pair (t_m1, t_m2) is considered distinct from (t_m2, t_m1) only if the
    two multisets differ; the canonical form with t_m1 <= t_m2 (lexicographic on
    sorted tuples) is generated once.  The POSCAR filename encodes both sides:
    ``<m1_combo>__<m2_combo>_<nb>x<bridge>.POSCAR``.

    Parameters
    ----------
    max_terminals_per_metal : int or None
        Cap on the number of terminal ligands on *each* metal centre.
        Defaults to 2, which covers the most chemically relevant cases
        (e.g. one water on one side, bare or one formate on the other) and
        keeps runtime under ~5 s for the standard Ni ligand pool.
        Set to None to remove the cap (runtime scales steeply beyond 3).

    Performance
    -----------
    The inner loop is charge-first: for each t_m1 with charge tc1, only combos
    of t_m2 whose charge equals ``target - tc1`` are generated, making the
    enumeration roughly linear in the number of valid t_m1 rather than quadratic
    in the full cross-product.
    """
    cn_min, cn_max = cn_range
    _max_b   = max_bridges if max_bridges is not None else cn_max
    _max_t   = max_terminals_per_metal  # None means no cap
    all_charges = {l: _ligand_charge(l)
                   for l in list(mono_ligands) + list(bi_ligands)}
    bridge_charge = {b: _ligand_charge(b) for b in bridge_ligands}
    all_terminals = list(mono_ligands) + list(bi_ligands)

    # Pre-build a dict: required_tc2 -> list of (sorted_tuple, t2_list) combos.
    # This is built once per (ox, bridge, nb, n2) and reused across all t1.
    def _combos_by_charge(n: int) -> Dict[int, List[tuple]]:
        """Map charge -> list of sorted-tuple combos of length n."""
        result: Dict[int, List[tuple]] = {}
        for combo in (combinations_with_replacement(all_terminals, n)
                      if n > 0 else [()]):
            tc = sum(all_charges[l] for l in combo)
            result.setdefault(tc, []).append(tuple(sorted(combo)))
        return result

    # Deduplication set
    _seen: set = set()

    for ox in ox_states:
        oxl = _ox_label(metal, ox)
        for bridge in bridge_ligands:
            bc = bridge_charge[bridge]
            for nb in range(1, _max_b + 1):
                # charge target: 2*ox + tc1 + tc2 + nb*bc = 0
                # -> tc1 + tc2 = -(2*ox + nb*bc)
                charge_target = -(2 * ox + nb * bc)

                # Build charge->combos index for each valid n2 once
                # n_max caps terminals per metal (None = no cap)
                n_max = (cn_max - nb) if _max_t is None else min(cn_max - nb, _max_t)

                n2_index: Dict[int, Dict[int, List[tuple]]] = {}
                for n2 in range(0, n_max + 1):
                    if not (cn_min <= n2 + nb <= cn_max):
                        continue
                    n2_index[n2] = _combos_by_charge(n2)

                for n1 in range(0, n_max + 1):
                    if not (cn_min <= n1 + nb <= cn_max):
                        continue

                    for t1 in (combinations_with_replacement(all_terminals, n1)
                               if n1 > 0 else [()]):
                        tc1 = sum(all_charges[l] for l in t1)
                        need_tc2 = charge_target - tc1
                        key_t1 = tuple(sorted(t1))

                        for n2, by_charge in n2_index.items():
                            if need_tc2 not in by_charge:
                                continue
                            for key_t2 in by_charge[need_tc2]:
                                # Skip homotopic
                                if key_t1 == key_t2:
                                    continue
                                # Canonical order
                                k1, k2 = (key_t1, key_t2) if key_t1 <= key_t2 \
                                         else (key_t2, key_t1)
                                dedup_key = (ox, bridge, nb, k1, k2)
                                if dedup_key in _seen:
                                    continue
                                _seen.add(dedup_key)

                                t1_list = list(k1)
                                t2_list = list(k2)
                                cn1 = len(t1_list) + nb
                                cn2 = len(t2_list) + nb
                                if not (cn_min <= cn1 <= cn_max):
                                    continue
                                if not (cn_min <= cn2 <= cn_max):
                                    continue

                                try:
                                    mol = dimer(metal, ox=ox,
                                                terminal_m1=t1_list,
                                                terminal_m2=t2_list,
                                                bridge=bridge, n=nb)
                                except Exception:
                                    continue
                                if mol.charge != 0:
                                    continue

                                cn_rep = max(cn1, cn2)
                                geom   = _DEFAULT_GEOM_FOR_CN.get(
                                    cn_rep, infer_geometry(cn_rep))
                                cl1  = combo_label(t1_list) or "bare"
                                cl2  = combo_label(t2_list) or "bare"
                                fname = (f"{safe(cl1)}__{safe(cl2)}_"
                                         f"{nb}x{safe(bridge)}.POSCAR")
                                path = _poscar_path(output_root, "dimer", oxl,
                                                    cn_rep, fname)
                                row = _make_row(mol, "dimer_hetero", metal, ox,
                                                cn_rep, geom,
                                                f"{cl1}__{cl2}",
                                                bridge, nb, None, "hetero", path)
                                if verbose:
                                    print(f"  OK dimer  {oxl} CN{cn1}/{cn2}  "
                                          f"{nb}x{bridge:10s}  "
                                          f"m1={t1_list}  m2={t2_list}  "
                                          f"{mol.formula}")
                                yield mol, row


# -- TRIMERS -------------------------------------------------------------------

def enumerate_trimers(
    metal: str,
    ox_states: Sequence[int],
    mono_ligands: Sequence[str],
    bridge_ligands: Sequence[str],
    bi_ligands: Sequence[str] = (),
    cn_range: Tuple[int, int] = (3, 7),
    arrangements: Sequence[str] = ("linear", "triangular"),
    multi_bridge_cases: Optional[List] = None,
    geometry_for_cn: Optional[Dict[int, str]] = None,
    output_root: Optional[Path] = None,
    verbose: bool = True,
) -> Iterator[Tuple[Any, Dict[str, Any]]]:
    """Yield (Molecule, row_dict) for every charge-neutral trimer."""
    cn_min, cn_max = cn_range
    mono_charge   = {l: _ligand_charge(l) for l in mono_ligands}
    bridge_charge = {b: _ligand_charge(b) for b in bridge_ligands}
    bi_charge     = {l: _ligand_charge(l) for l in bi_ligands}
    bi_cn_map     = {l: _ligand_cn(l)     for l in bi_ligands}
    mb_cases      = multi_bridge_cases if multi_bridge_cases is not None else MULTI_BRIDGE_CASES

    for ox in ox_states:
        oxl = _ox_label(metal, ox)
        for bridge in bridge_ligands:
            bc = bridge_charge[bridge]

            # single bridge per pair
            for n_term in range(0, cn_max):
                mono_iter = (combinations_with_replacement(mono_ligands, n_term)
                             if n_term > 0 else [()])
                for terminal in mono_iter:
                    tc  = sum(mono_charge[l] for l in terminal)
                    cn  = n_term + 2
                    if not (cn_min <= cn <= cn_max):
                        continue
                    for arrangement in arrangements:
                        n_bp = _BRIDGE_PAIRS[arrangement]
                        if 3 * ox + 3 * tc + n_bp * bc != 0:
                            continue
                        try:
                            mol = trimer(metal, ox=ox,
                                         terminal=list(terminal),
                                         bridge=bridge,
                                         arrangement=arrangement)
                        except Exception:
                            continue
                        if mol.charge != 0:
                            continue
                        geom = _DEFAULT_GEOM_FOR_CN.get(cn, infer_geometry(cn))
                        cl   = combo_label(list(terminal))
                        path = _poscar_path(output_root, "trimer", oxl, cn,
                                            f"{safe(cl)}_{safe(bridge)}_{arrangement}.POSCAR")
                        row = _make_row(mol, f"trimer_{arrangement}", metal, ox, cn,
                                        geom, cl, bridge, n_bp, arrangement, "only", path)
                        if verbose:
                            print(f"  OK trimer {oxl} CN{cn}  {bridge:10s}  "
                                  f"term={list(terminal)}  "
                                  f"{arrangement:12s}  {mol.formula}")
                        yield mol, row

                # bidentate terminals
                for bi_name in bi_ligands:
                    b_cn  = bi_cn_map[bi_name]
                    b_chg = bi_charge[bi_name]
                    cn_bi = 2 + b_cn + n_term
                    if not (cn_min <= cn_bi <= cn_max):
                        continue
                    mono_iter2 = (combinations_with_replacement(mono_ligands, n_term)
                                  if n_term > 0 else [()])
                    for mono_combo in mono_iter2:
                        tc2 = b_chg + sum(mono_charge[l] for l in mono_combo)
                        terminal_bi = [bi_name] + list(mono_combo)
                        for arrangement in arrangements:
                            n_bp = _BRIDGE_PAIRS[arrangement]
                            if 3 * ox + 3 * tc2 + n_bp * bc != 0:
                                continue
                            try:
                                mol = trimer(metal, ox=ox,
                                             terminal=terminal_bi,
                                             bridge=bridge,
                                             arrangement=arrangement)
                            except Exception:
                                continue
                            if mol.charge != 0:
                                continue
                            geom = _DEFAULT_GEOM_FOR_CN.get(cn_bi, infer_geometry(cn_bi))
                            cl   = combo_label(terminal_bi)
                            path = _poscar_path(output_root, "trimer", oxl, cn_bi,
                                                f"{safe(cl)}_{safe(bridge)}_{arrangement}.POSCAR")
                            row = _make_row(mol, f"trimer_{arrangement}", metal, ox, cn_bi,
                                            geom, cl, bridge, n_bp, arrangement, "only", path)
                            if verbose:
                                print(f"  OK trimer {oxl} CN{cn_bi}  "
                                      f"{bridge:10s}  term={terminal_bi}  "
                                      f"{arrangement}  {mol.formula}")
                            yield mol, row

    # multi-bridge trimers from the cases table
    for (arr, nbpp, bridge, ox, terminal, suffix) in mb_cases:
        if bridge not in bridge_ligands:
            continue
        if ox not in ox_states:
            continue
        oxl  = _ox_label(metal, ox)
        bc   = bridge_charge.get(bridge, _ligand_charge(bridge))
        n_bp = _BRIDGE_PAIRS[arr]
        tc   = sum(mono_charge.get(l, _ligand_charge(l)) for l in terminal)
        if 3 * ox + 3 * tc + n_bp * nbpp * bc != 0:
            continue
        cn = len(terminal) + 2 * nbpp
        if not (cn_range[0] <= cn <= cn_range[1]):
            continue
        try:
            mol = trimer(metal, ox=ox,
                         terminal=list(terminal),
                         bridge=bridge,
                         arrangement=arr,
                         n_bridges_per_pair=nbpp)
        except Exception as e:
            if verbose:
                print(f"  FAIL trimer {oxl} {arr} {nbpp}x{bridge} "
                      f"term={terminal}: {e}")
            continue
        if mol.charge != 0:
            continue
        geom = _DEFAULT_GEOM_FOR_CN.get(cn, infer_geometry(cn))
        cl   = combo_label(list(terminal))
        stem = (f"{safe(cl)}_{nbpp}x{safe(bridge)}_{arr}"
                if cl else f"{nbpp}x{safe(bridge)}_{arr}")
        path = _poscar_path(output_root, "trimer", oxl, cn, f"{stem}.POSCAR")
        row  = _make_row(mol, f"trimer_{arr}", metal, ox, cn, geom,
                         cl, bridge, n_bp * nbpp, arr, "only", path)
        if verbose:
            print(f"  OK trimer {oxl} CN{cn}  {nbpp}x{bridge:10s}  "
                  f"term={list(terminal)}  {arr:12s}  {mol.formula}")
        yield mol, row


# -- HETEROLEPTIC TRIMERS ------------------------------------------------------

def enumerate_heteroleptic_trimers(
    metal: str,
    ox_states: Sequence[int],
    mono_ligands: Sequence[str],
    bridge_ligands: Sequence[str],
    bi_ligands: Sequence[str] = (),
    cn_range: Tuple[int, int] = (3, 7),
    arrangements: Sequence[str] = ("linear", "triangular"),
    multi_bridge_cases: Optional[List] = None,
    max_terminals_per_metal: Optional[int] = 2,
    geometry_for_cn: Optional[Dict[int, str]] = None,
    output_root: Optional[Path] = None,
    verbose: bool = True,
) -> Iterator[Tuple[Any, Dict[str, Any]]]:
    """
    Yield (Molecule, row_dict) for heteroleptic trimers where the three metal
    centres carry *different* sets of terminal ligands.

    For each entry in MULTI_BRIDGE_CASES (and the single-bridge-per-pair trimer
    enumeration), this function generates all unique assignments of terminal
    ligands to the three metals where *at least one* metal differs.

    The canonical ordering is the sorted tuple of per-metal ligand multisets,
    so (m0=[], m1=[H2O], m2=[]) and (m0=[H2O], m1=[], m2=[]) are the same
    structure and generated only once.

    Parameters
    ----------
    max_terminals_per_metal : int or None
        Cap on terminals per metal site (default 2). Controls the combinatorial
        explosion: 1->fast, 2->~10s for the standard Ni pool, 3->slow.
    """
    cn_min, cn_max = cn_range
    all_charges   = {l: _ligand_charge(l)
                     for l in list(mono_ligands) + list(bi_ligands)}
    bridge_charge = {b: _ligand_charge(b) for b in bridge_ligands}
    all_terminals = list(mono_ligands) + list(bi_ligands)
    mb_cases      = multi_bridge_cases if multi_bridge_cases is not None else MULTI_BRIDGE_CASES
    _max_t        = max_terminals_per_metal

    # Helper: all terminal combos of length n
    def _combos(n: int):
        return (list(combinations_with_replacement(all_terminals, n))
                if n > 0 else [()])

    def _combos_by_charge_cache(n_max: int) -> Dict[int, Dict[int, List[tuple]]]:
        out: Dict[int, Dict[int, List[tuple]]] = {}
        for n in range(0, n_max + 1):
            by_c: Dict[int, List[tuple]] = {}
            for combo in _combos(n):
                tc = sum(all_charges[l] for l in combo)
                by_c.setdefault(tc, []).append(tuple(sorted(combo)))
            out[n] = by_c
        return out

    # Iterate over MULTI_BRIDGE_CASES entries only (the interesting multi-bridge trimers)
    for (arr, nbpp, bridge, ox, base_terminal, suffix) in mb_cases:
        if bridge not in bridge_ligands:
            continue
        if ox not in ox_states:
            continue
        bc   = bridge_charge.get(bridge, _ligand_charge(bridge))
        n_bp = _BRIDGE_PAIRS[arr]
        oxl  = _ox_label(metal, ox)

        # Base charge contributed by bridges
        bridge_charge_total = n_bp * nbpp * bc
        # We need: 3*ox + sum(tc_k for k in 0,1,2) + bridge_charge_total = 0
        # -> tc0 + tc1 + tc2 = -(3*ox + bridge_charge_total)
        total_tc_needed = -(3 * ox + bridge_charge_total)

        # Max terminals per metal (CN constraint)
        n_max = (_max_t if _max_t is not None
                 else cn_max - 2 * nbpp)
        n_max = max(0, min(n_max, cn_max - 2 * nbpp))

        # Cache combos-by-charge for each length
        cbc = _combos_by_charge_cache(n_max)

        # Enumerate all 3-tuples (t0, t1, t2) that are charge-neutral
        # and where at least one differs from base_terminal
        base_key = tuple(sorted(base_terminal))
        _seen: set = set()

        for n0 in range(0, n_max + 1):
            if not (cn_min <= n0 + 2 * nbpp <= cn_max):
                continue
            for tc0, t0_list in cbc[n0].items():
                for t0 in t0_list:
                    remain_tc = total_tc_needed - tc0

                    for n1 in range(0, n_max + 1):
                        if not (cn_min <= n1 + 2 * nbpp <= cn_max):
                            continue
                        if n1 not in cbc:
                            continue
                        for tc1, t1_list in cbc[n1].items():
                            if tc1 > remain_tc:
                                continue
                            tc2_needed = remain_tc - tc1
                            for t1 in t1_list:

                                for n2 in range(0, n_max + 1):
                                    if not (cn_min <= n2 + 2 * nbpp <= cn_max):
                                        continue
                                    if tc2_needed not in cbc.get(n2, {}):
                                        continue
                                    for t2 in cbc[n2][tc2_needed]:

                                        # Skip if all three metals same as base
                                        if t0 == base_key and t1 == base_key and t2 == base_key:
                                            continue
                                        # Skip if all three identical (covered by symmetric trimer)
                                        if t0 == t1 == t2:
                                            continue

                                        # Canonical form: sort the 3-tuple
                                        canon = tuple(sorted([t0, t1, t2]))
                                        dedup_key = (ox, bridge, arr, nbpp, canon)
                                        if dedup_key in _seen:
                                            continue
                                        _seen.add(dedup_key)

                                        try:
                                            mol = trimer(
                                                metal, ox=ox,
                                                bridge=bridge,
                                                arrangement=arr,
                                                n_bridges_per_pair=nbpp,
                                                terminals_per_metal=[
                                                    list(t0), list(t1), list(t2)
                                                ],
                                            )
                                        except Exception:
                                            continue
                                        if mol.charge != 0:
                                            continue
                                        if not _passes_hard_validation(mol):
                                            continue

                                        cn_rep = max(n0, n1, n2) + 2 * nbpp
                                        geom   = _DEFAULT_GEOM_FOR_CN.get(
                                            cn_rep, infer_geometry(cn_rep))
                                        cl0    = combo_label(list(t0)) or "bare"
                                        cl1    = combo_label(list(t1)) or "bare"
                                        cl2    = combo_label(list(t2)) or "bare"
                                        stem   = (f"{safe(cl0)}__{safe(cl1)}__{safe(cl2)}_"
                                                  f"{nbpp}x{safe(bridge)}_{arr}")
                                        path   = _poscar_path(output_root, "trimer", oxl,
                                                              cn_rep, f"{stem}.POSCAR")
                                        lig_combo = f"{cl0}__{cl1}__{cl2}"
                                        row = _make_row(mol, f"trimer_{arr}_hetero", metal, ox,
                                                        cn_rep, geom, lig_combo,
                                                        bridge, n_bp * nbpp,
                                                        arr, "hetero", path)
                                        if verbose:
                                            print(f"  OK trimer {oxl} CN{cn_rep}  "
                                                  f"{nbpp}x{bridge:10s}  "
                                                  f"m0={list(t0)} m1={list(t1)} m2={list(t2)}  "
                                                  f"{arr}  {mol.formula}")
                                        yield mol, row


# -- top-level convenience -----------------------------------------------------

def enumerate_complexes(
    metal: str,
    ox_states: Sequence[int],
    ligand_pool: Sequence[str],
    bridge_pool: Sequence[str] = (),
    bi_ligands: Sequence[str] = (),
    nuclearity: Sequence[int] = (1, 2, 3),
    arrangements: Sequence[str] = ("linear", "triangular"),
    cn_range: Tuple[int, int] = (3, 7),
    max_bridges_per_pair: Optional[int] = None,
    multi_bridge_cases: Optional[List] = None,
    geometry_for_cn: Optional[Dict[int, str]] = None,
    include_heteroleptic: bool = False,
    include_heteroleptic_trimers: bool = False,
    max_terminals_per_metal: Optional[int] = 2,
    output_root: Optional[Path] = None,
    verbose: bool = True,
) -> Iterator[Tuple[Any, Dict[str, Any]]]:
    """
    Yield (Molecule, metadata_dict) for every charge-neutral complex.

    Parameters
    ----------
    metal                        : Element symbol, e.g. "Ni", "Co", "Fe".
    ox_states                    : Oxidation states to enumerate, e.g. [2, 3].
    ligand_pool                  : Terminal monodentate ligand names.
    bridge_pool                  : Bridging ligand names for di/trinuclear complexes.
    bi_ligands                   : Bidentate chelating terminal ligand names.
    nuclearity                   : Which nuclearities: 1=monomer, 2=dimer, 3=trimer.
    arrangements                 : Trimer topologies: "linear", "triangular".
    cn_range                     : (min_CN, max_CN) inclusive, per metal centre.
    max_bridges_per_pair         : Upper limit on bridges per metal pair in dimers.
    multi_bridge_cases           : Override the default MULTI_BRIDGE_CASES table.
    geometry_for_cn              : Override default CN -> geometry mapping.
    include_heteroleptic         : Enumerate heteroleptic dimers (different terminals
                                   on each metal). Disabled by default.
    include_heteroleptic_trimers : Enumerate heteroleptic trimers (different terminals
                                   per metal site). Disabled by default. When enabled,
                                   generates e.g. Ni3(mu-HCOO)6 + H2O on one Ni only.
    max_terminals_per_metal      : Cap on terminals per metal in heteroleptic cases.
                                   Default 2. Pass None to remove the cap.
    output_root                  : Root directory for POSCAR path metadata.
    verbose                      : Print each structure as generated.

    Yields
    ------
    (mol, row_dict)
    """
    if 1 in nuclearity:
        if verbose:
            print("\n" + "=" * 60)
            print(f"  MONOMERS  ({metal})")
            print("=" * 60)
        yield from enumerate_monomers(
            metal, ox_states, ligand_pool, bi_ligands,
            cn_range, geometry_for_cn, None, output_root, verbose,
        )

    if 2 in nuclearity and bridge_pool:
        if verbose:
            print("\n" + "=" * 60)
            print(f"  DIMERS  ({metal})")
            print("=" * 60)
        yield from enumerate_dimers(
            metal, ox_states, ligand_pool, list(bridge_pool), bi_ligands,
            cn_range, max_bridges_per_pair, geometry_for_cn, output_root, verbose,
        )
        if include_heteroleptic:
            if verbose:
                print("\n" + "-" * 60)
                print(f"  DIMERS - heteroleptic  ({metal})")
                print("-" * 60)
            yield from enumerate_heteroleptic_dimers(
                metal, ox_states, ligand_pool, list(bridge_pool), bi_ligands,
                cn_range, max_bridges_per_pair, max_terminals_per_metal,
                geometry_for_cn, output_root, verbose,
            )

    if 3 in nuclearity and bridge_pool:
        if verbose:
            print("\n" + "=" * 60)
            print(f"  TRIMERS  ({metal})")
            print("=" * 60)
        yield from enumerate_trimers(
            metal, ox_states, ligand_pool, list(bridge_pool), bi_ligands,
            cn_range, list(arrangements), multi_bridge_cases,
            geometry_for_cn, output_root, verbose,
        )
        if include_heteroleptic_trimers:
            if verbose:
                print("\n" + "-" * 60)
                print(f"  TRIMERS - heteroleptic  ({metal})")
                print("-" * 60)
            yield from enumerate_heteroleptic_trimers(
                metal, ox_states, ligand_pool, list(bridge_pool), bi_ligands,
                cn_range, list(arrangements), multi_bridge_cases,
                max_terminals_per_metal, geometry_for_cn, output_root, verbose,
            )


# -- internal path helper ------------------------------------------------------

def _poscar_path(output_root: Optional[Path],
                 structure_type: str,
                 ox_label: str,
                 cn: int,
                 filename: str) -> Path:
    if output_root is None:
        return Path(filename)
    return output_root / structure_type / ox_label / f"CN{cn}" / filename
