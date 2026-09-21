"""
pls_pole_xml_parser.py

Independent parser for PLS-POLE XML export files, built for the pole
design optimization tool. Not related to / does not reuse PoleAgent's
pls_parser.py.

Structure reference: see pls-pole-xml-reader skill (built from a real
export, PLS-POLE Version 21.01).

Key parsing rules implemented here:
- File encoding is windows-1252.
- Content is a sequence of <table plsname=... tagname=... nrows=...
  titledetail=...> blocks, each containing <tagname rownum='N'> row
  elements with field sub-elements (optionally carrying units='...').
- `titledetail` marks a table as having multiple sub-instances, but its
  MEANING depends on the table (confirmed against a real file — do not
  assume "non-empty titledetail" always means "load case"):
    - For the tables in LOAD_CASE_TAGNAMES below, titledetail IS the
      load case description (no load_case field exists on the rows
      themselves for these tables).
    - For other tables (e.g. intermediate_joints, base_plate,
      relative_attachment_labels_for), titledetail identifies a
      different kind of sub-instance (davit property, pole property,
      etc.) — NOT a load case. Don't key these by load case.
- Tables with titledetail='' are either one-time model-definition
  tables, or summary/aggregate tables that already carry their own
  load_case field per row.
"""

from lxml import etree
from dataclasses import dataclass, field
from typing import Optional
from collections import defaultdict

# Tables confirmed (against a real export) to use titledetail as the
# load case description. Only these should be treated as "per load case."
LOAD_CASE_TAGNAMES = {
    'point_loads',
    'detailed_pole_loading_data',
    'equilibrium_joint_positions_and_rotations',
    'joint_support_reactions',
    'detailed_steel_pole_usages',
    'detailed_tubular_davit_arm_usages',
    'summary_of_clamp_capacities_and_usages',
}


def parse_pls_pole_xml(filepath: str) -> dict:
    """
    Parse a PLS-POLE XML export into a structured dict.

    Returns a dict with two top-level keys:
      - 'tables': dict of tagname -> list of table-instances, where each
        table-instance is {'titledetail': str, 'rows': list[dict]}
        (list, because per-load-case tables repeat under the same tagname)
      - 'creator': dict of metadata from the <creator> element
    """
    parser = etree.XMLParser(encoding='windows-1252', recover=False)
    tree = etree.parse(filepath, parser=parser)
    root = tree.getroot()

    result = {'tables': defaultdict(list), 'creator': {}}

    creator_el = root.find('creator')
    if creator_el is not None:
        result['creator'] = dict(creator_el.attrib)

    for table_el in root.findall('table'):
        tagname = table_el.get('tagname')
        titledetail = table_el.get('titledetail', '')
        plsname = table_el.get('plsname')
        nrows_declared = int(table_el.get('nrows', 0))

        rows = []
        for row_el in table_el.findall(tagname):
            row_data = {}
            for field_el in row_el:
                field_name = field_el.tag
                units = field_el.get('units')
                value_text = field_el.text
                value = value_text.strip() if value_text else None
                if units:
                    row_data[field_name] = {'value': value, 'units': units}
                else:
                    row_data[field_name] = value
            rows.append(row_data)

        result['tables'][tagname].append({
            'plsname': plsname,
            'titledetail': titledetail,
            'nrows_declared': nrows_declared,
            'nrows_actual': len(rows),
            'rows': rows,
        })

    result['tables'] = dict(result['tables'])
    return result


def get_field(row: dict, field_name: str, default=None):
    """Extract a field's numeric value from a parsed row, handling both
    the {'value':..,'units':..} dict form and plain string form."""
    v = row.get(field_name, default)
    if isinstance(v, dict):
        v = v.get('value')
    if v is None:
        return default
    try:
        return float(v)
    except (ValueError, TypeError):
        return v  # non-numeric field (labels, descriptions)


def get_load_case_instances(parsed: dict, tagname: str) -> dict:
    """
    For a CONFIRMED per-load-case table (tagname must be in
    LOAD_CASE_TAGNAMES), return a dict keyed by load case description
    (from titledetail) -> rows.
    """
    if tagname not in LOAD_CASE_TAGNAMES:
        raise ValueError(
            f"'{tagname}' is not a confirmed per-load-case table. Its "
            f"titledetail attribute (if any) means something else — check "
            f"the pls-pole-xml-reader skill / table-schema.md before "
            f"assuming this table is keyed by load case."
        )
    instances = parsed['tables'].get(tagname, [])
    out = {}
    for inst in instances:
        key = inst['titledetail']
        if key:
            out[key] = inst['rows']
    return out


def get_single_table(parsed: dict, tagname: str) -> list:
    """For a one-time (non-repeating) table, return its rows directly."""
    instances = parsed['tables'].get(tagname, [])
    if not instances:
        return []
    if len(instances) > 1:
        raise ValueError(
            f"Expected a single instance of table '{tagname}' but found "
            f"{len(instances)} — this table may actually be per-load-case; "
            f"use get_load_case_instances() instead."
        )
    return instances[0]['rows']


if __name__ == '__main__':
    import sys
    filepath = sys.argv[1] if len(sys.argv) > 1 else \
        '/mnt/user-data/uploads/001_T316-S-01_105FT_FDN.xml'

    parsed = parse_pls_pole_xml(filepath)

    print("=== Creator metadata ===")
    for k, v in parsed['creator'].items():
        print(f"  {k}: {v}")

    print("\n=== Table inventory ===")
    for tagname, instances in sorted(parsed['tables'].items()):
        n_instances = len(instances)
        has_titledetail = any(inst['titledetail'] for inst in instances)
        if tagname in LOAD_CASE_TAGNAMES:
            kind = "PER-LOAD-CASE"
        elif has_titledetail:
            kind = "repeated (non-load-case sub-instance)"
        else:
            kind = "single/summary"
        print(f"  {tagname}: {n_instances} instance(s) [{kind}]")
