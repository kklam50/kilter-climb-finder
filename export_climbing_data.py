"""
LLM Training Data Extraction Pipeline
Converts climbing layout data into grid-based training records.

Usage:
    python export_climbing_data.py

Outputs:
    - climbs_training_set.jsonl (final training data)
    - validation_report.csv (QC metrics)
    - data_dictionary.md (field descriptions)
"""

import sqlite3
import json
import re
import math
from pathlib import Path


def extract_climbs(db_path: str, layout_id: int = 1, limit: int | None = None) -> list[dict]:
    """Extract filtered climbs with supporting tables.

    Args:
        db_path: Path to SQLite database
        layout_id: Filter climbs by layout_id (default: 1)
        limit: Optional limit on number of climbs (None for all rows)
    """
    conn = sqlite3.connect(db_path)
    # Use Row factory for proper dict-like access
    conn.row_factory = sqlite3.Row

    cursor = conn.cursor()
    query = """
        SELECT c.uuid, c.layout_id, c.name, c.description, c.frames, c.created_at,
               SUM(cs.ascensionist_count) as ascensionist_count
        FROM climbs c
        JOIN climb_stats cs ON c.uuid = cs.climb_uuid
        WHERE c.angle IS NOT NULL AND c.layout_id = ?
          AND cs.ascensionist_count > 5
        GROUP BY c.uuid, c.layout_id, c.name, c.description, c.frames, c.created_at
        ORDER BY c.name
    """
    args = [layout_id]
    if limit is not None:
        query += " LIMIT ?"
        args.append(limit)

    cursor.execute(query, tuple(args))
    rows = cursor.fetchall()

    climb_records = []
    for row in rows:
        # Build dict by accessing column names directly
        record = {
            "uuid": row["uuid"],
            "layout_id": row["layout_id"],
            "name": row["name"],
            "description": row["description"],
            "frames": row["frames"],
            "created_at": row["created_at"],
            "ascensionist_count": row["ascensionist_count"],
        }
        # Parse frames string later using holes/placement_roles data
        climb_records.append(record)

    conn.close()
    return climb_records


def build_holes_lookup(db_path: str, layout_id: int = 1) -> dict[int, dict]:
    """Build dict: hole_id -> {id, product_id, x, y, name}"""
    conn = sqlite3.connect(db_path)
    # Use Row factory for proper dict-like access
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Get all hole_ids referenced by placements for this layout
    cursor.execute("""
        SELECT DISTINCT p.id, h.product_id, h.x, h.y, h.name
        FROM holes h
        JOIN placements p ON h.id = p.hole_id
        WHERE p.layout_id = ?
        ORDER BY x DESC
    """, (layout_id,))

    rows = cursor.fetchall()
    # Build dict by accessing row values directly (Row objects are dict-like)
    holes_lookup = {}
    for row in rows:
        hole_id = row["id"]
        holes_lookup[hole_id] = {
            "id": row["id"],
            "product_id": row["product_id"],
            "x": row["x"],
            "y": row["y"],
            "name": row["name"],
        }

    conn.close()
    return holes_lookup


def build_placement_roles_lookup(db_path: str, product_ids: set[int]) -> dict[tuple[int, int], dict]:
    """Build dict: (product_id, role_id) -> {full_name, name, ascii_symbol, led_color}"""
    conn = sqlite3.connect(db_path)
    # Use Row factory for proper dict-like access
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Build IN clause with placeholders for each product_id
    ids_str = ",".join("?" * len(product_ids))
    query = f"""
        SELECT id, product_id, position, full_name,
               CASE
                   WHEN full_name = 'Start' THEN 'S'
                   WHEN full_name = 'Middle' THEN 'M'
                   WHEN full_name = 'Finish' THEN 'F'
                   WHEN full_name = 'Foot Only' THEN 'X'
                   ELSE '.'
               END as ascii_symbol,
               led_color
        FROM placement_roles
        WHERE product_id IN ({ids_str})
        ORDER BY product_id, id
    """
    cursor.execute(query, tuple(sorted(product_ids)))
    rows = cursor.fetchall()

    roles_lookup = {}
    for row in rows:
        key = (row["product_id"], row["id"])  # (product_id, role_id)
        roles_lookup[key] = {
            "id": row["id"],
            "position": row["position"],
            "full_name": row["full_name"],
            "ascii_symbol": row["ascii_symbol"],
            "led_color": row["led_color"],
        }

    conn.close()
    return roles_lookup


def frames_to_ascii_grid(frames_str: str, holes_table: dict[int, dict],
                         placement_roles_table: dict[tuple[int, int], dict]) -> dict:
    """Convert frames string to ASCII art grid representation.

    Args:
        frames_str: Raw frames string (e.g., "p1145r12p1146r12...")
        holes_table: Dict mapping hole_id -> {id, product_id, x, y}
        placement_roles_table: Dict mapping (product_id, id) -> {ascii_symbol, full_name, ...}

    Returns:
        Dict with:
          - "ascii_grid": 2D string grid (rows of characters like ".FSM")
          - "grid_min_x", "grid_max_x": horizontal bounds
          - "grid_min_y", "grid_max_y": vertical bounds
          - "piece_map": list of hold positions for reference
    """
    if not frames_str:
        return {
            "ascii_grid": "",
            "grid_min_x": 0,
            "grid_max_x": 0,
            "grid_min_y": 0,
            "grid_max_y": 0,
            "piece_map": []
        }

    # Parse each segment: format "p<hole_id>r<piece_type_id>"
    # Pattern captures "p<number>r<number>", extract just the numbers
    segments = re.findall(r'p(\d+)r(\d+)', frames_str)

    if not segments:
        return {
            "ascii_grid": "",
            "grid_min_x": 0,
            "grid_max_x": 0,
            "grid_min_y": 0,
            "grid_max_y": 0,
            "piece_map": []
        }

    # Collect all unique (x, y) coordinates and build piece_map with symbols
    coords = set()
    piece_map = []  # List of {x, y, type_id, symbol} for reference

    for hole_id_str, piece_type_id_str in segments:
        hole_id = int(hole_id_str)
        piece_type_id = int(piece_type_id_str)

        # Get hole coordinates (hole_id is the primary key in holes table)
        if hole_id not in holes_table:
            continue  # Skip invalid hole references
        hole_data = holes_table[hole_id] # this needs to reference placement_roles_table first
        x, y = int(hole_data["x"]), int(hole_data["y"])
        product_id = hole_data["product_id"]
        coords.add((x, y))

        # Build composite key for roles lookup: (product_id, role_id)
        role_key = (product_id, piece_type_id)
        symbol = "."
        if role_key in placement_roles_table:
            symbol = placement_roles_table[role_key].get("ascii_symbol", ".")

        piece_map.append({
            "x": x,
            "y": y,
            "type_id": piece_type_id,
            "symbol": symbol,
        }) 

    if not coords:
        return {
            "ascii_grid": "",
            "grid_min_x": 0,
            "grid_max_x": 0,
            "grid_min_y": 0,
            "grid_max_y": 0,
            "piece_map": piece_map
        }

    min_x, max_x = min(c[0] for c in coords), max(c[0] for c in coords)
    min_y, max_y = min(c[1] for c in coords), max(c[1] for c in coords)
    # Build the ASCII grid: each row is a string of characters
    # Grid columns represent x values from min_x to max_x
    # Grid rows are ordered by y value (top to bottom as we display)
    ascii_grid_lines = []

    for y in range(max_y, - 1, -1):  # Display top (high y) to bottom (low y)
        line_chars = []
        for x in range(0, max_x + 1):  # Left to right
            hold_piece = None
            for p in piece_map:
                if p["x"] == x and p["y"] == y:
                    hold_piece = p
                    break

            if hold_piece:
                symbol = hold_piece.get("symbol", ".")
                line_chars.append(symbol)
            else:
                line_chars.append(".")

        ascii_grid_lines.append("".join(line_chars))

    return {
        "ascii_grid": "\n".join(ascii_grid_lines),
        "grid_min_x": min_x,
        "grid_max_x": max_x,
        "grid_min_y": min_y,
        "grid_max_y": max_y,
        "piece_map": piece_map
    }


def compute_hold_sparsity(piece_map: list[dict]) -> float:
    """Calculate hold sparsity (holds per unit area)."""
    if not piece_map:
        return 0.0

    min_x = min(p["x"] for p in piece_map)
    max_x = max(p["x"] for p in piece_map)
    min_y = min(p["y"] for p in piece_map)
    max_y = max(p["y"] for p in piece_map)

    area = (max_x - min_x + 1) * (max_y - min_y + 1)
    if area == 0:
        return 0.0

    return len(piece_map) / area


def compute_max_reach(piece_map: list[dict]) -> float:
    """Calculate maximum reach distance between consecutive hold pairs."""
    if len(piece_map) < 2:
        return 0.0

    # Sort by position to get natural climb order (use piece type position as heuristic)
    sorted_pieces = sorted(piece_map, key=lambda p: p.get("type_id", 0))

    max_distance = 0.0
    for i in range(len(sorted_pieces) - 1):
        x1, y1 = sorted_pieces[i]["x"], sorted_pieces[i]["y"]
        x2, y2 = sorted_pieces[i + 1]["x"], sorted_pieces[i + 1]["y"]
        distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        max_distance = max(max_distance, distance)

    return round(max_distance, 2)


def process_climb(climb: dict, holes_table: dict[int, dict],
                  placement_roles_table: dict[tuple[int, int], dict]) -> dict:
    """Process a single climb record into training data."""
    # Parse frames string to ASCII grid
    grid_result = frames_to_ascii_grid(
        climb.get("frames", ""),
        holes_table,
        placement_roles_table
    )

    if not grid_result["ascii_grid"]:
        return None  # Skip climbs with no valid layout

    piece_map = grid_result["piece_map"]
    num_holds = len(piece_map)

    return {
        "climb_id": climb["uuid"],
        "climb_name": climb["name"],
        "description": climb.get("description", ""),
        "layout_grid": grid_result,
        "num_holds": num_holds,
        "hold_sparsity": round(compute_hold_sparsity(piece_map), 4),
        "max_reach": compute_max_reach(piece_map),
        "created_at": climb.get("created_at", ""),
    }


def export_climbing_data(db_path: str, layout_id: int = 1, output_dir: str = ".",
                         limit: int | None = None) -> dict:
    """Main extraction and transformation pipeline.

    Args:
        db_path: Path to SQLite database
        layout_id: Filter climbs by layout_id (default: 1)
        output_dir: Directory for output files
        limit: Optional limit on number of climbs (None for all rows, default: None)

    Returns:
        Dictionary with paths to generated files and metrics
    """
    # Step 1: Extract climb records
    print(f"Extracting climbs from {db_path} (layout_id={layout_id})...")
    if limit is not None:
        print(f"  Limiting output to {limit} climbs for validation")
    climbs = extract_climbs(db_path, layout_id, limit)
    print(f"  Found {len(climbs)} climbs")

    if not climbs:
        print("No climbs found. Exiting.")
        return {}

    # Step 2: Build lookup tables
    holes_table = build_holes_lookup(db_path, layout_id)
    product_ids = set(h["product_id"] for h in holes_table.values())
    placement_roles_table = build_placement_roles_lookup(db_path, product_ids)

    print(f"  Loaded {len(holes_table)} hole types across {len(product_ids)} products")
    print(f"  Loaded {len(placement_roles_table)} placement role definitions")

    # Step 3: Process each climb into training record
    records = []
    for climb in climbs:
        try:
            record = process_climb(climb, holes_table, placement_roles_table)
            if record:
                records.append(record)
        except Exception as e:
            # Handle unicode names by encoding them for display
            try:
                name_display = climb["name"]
            except UnicodeEncodeError:
                name_display = str(climb.get("name", ""))[:50]  # Truncate long names
            print(f"  Warning: Could not process climb {name_display}: {e}")

    valid_records = [r for r in records if r is not None]
    print(f"  Generated {len(valid_records)} training records")

    # Step 4: Write JSONL output
    jsonl_path = Path(output_dir) / "training_data/climbs_training_set.jsonl"
    with open(jsonl_path, "w") as f:
        for record in valid_records:
            f.write(json.dumps(record) + "\n")

    print(f"  Written training data to {jsonl_path}")

    # Step 5: Generate validation report
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Count total placements for layout
    cursor.execute("SELECT COUNT(*) FROM placements WHERE layout_id = ?", (layout_id,))
    total_placements = cursor.fetchone()[0]

    # Check for orphaned placement->hole references
    cursor.execute("""
        SELECT COUNT(DISTINCT hole_id) FROM placements
        WHERE layout_id = ? AND hole_id NOT IN (SELECT id FROM holes)
    """, (layout_id,))
    orphaned_placements = cursor.fetchone()[0]

    # Count unique piece types used
    conn.close()

    # Calculate grid bounds from the last processed climb (or first)
    grid_width = None
    grid_height = None
    if valid_records:
        grid_result = frames_to_ascii_grid(
            valid_records[-1].get("layout_grid", {}).get("ascii_grid", ""),
            holes_table,
            placement_roles_table
        )
        grid_width = grid_result.get("grid_max_x", 0) - grid_result.get("grid_min_x", 0) + 1
        grid_height = grid_result.get("grid_max_y", 0) - grid_result.get("grid_min_y", 0) + 1

    validation_report = {
        "total_climbs_processed": len(climbs),
        "valid_training_records": len(valid_records),
        "climbs_with_layouts": len(valid_records),
        "climbs_without_layouts": len(climbs) - len(valid_records),
        "total_placements_in_database": total_placements,
        "orphaned_placement_references": orphaned_placements,
        "grid_width_range": f"{grid_width}" if grid_width else "N/A",
        "grid_height_range": f"{grid_height}" if grid_height else "N/A",
    }

    # Find min/max x/y from all grids
    if valid_records:
        global_min_x = min(r["layout_grid"]["grid_min_x"] for r in valid_records)
        global_max_x = max(r["layout_grid"]["grid_max_x"] for r in valid_records)
        global_min_y = min(r["layout_grid"]["grid_min_y"] for r in valid_records)
        global_max_y = max(r["layout_grid"]["grid_max_y"] for r in valid_records)
        validation_report["global_min_x"] = global_min_x
        validation_report["global_max_x"] = global_max_x
        validation_report["global_min_y"] = global_min_y
        validation_report["global_max_y"] = global_max_y

    # Write CSV report
    csv_path = Path(output_dir) / "training_data/validation_report.csv"
    with open(csv_path, "w") as f:
        f.write("metric,value\n")
        for key, value in validation_report.items():
            if isinstance(value, float):
                f.write(f"{key},{value:.4f}\n")
            else:
                f.write(f"{key},{value}\n")

    print(f"  Written validation report to {csv_path}")

    # Step 6: Generate data dictionary
    dict_path = Path(output_dir) / "training_data/data_dictionary.md"
    with open(dict_path, "w") as f:
        f.write("# LLM Training Data Field Dictionary\n\n")
        f.write("## Overview\n")
        f.write("This document describes the fields in `climbs_training_set.jsonl` for downstream use.\n\n")

        f.write("### Core Fields\n\n")
        f.write("| Field | Type | Description |\n")
        f.write("|-------|------|-------------|\n")
        f.write(f"| climb_id | str | Unique identifier (UUID) of the climb route |\n")
        f.write(f"| climb_name | str | Human-readable name of the climb |\n")
        f.write(f"| description | str | Optional description text for context |\n")
        f.write("|---\n\n")

        f.write("### Layout Grid\n\n")
        f.write("| Field | Type | Description |\n")
        f.write("|-------|------|-------------|\n")
        f.write("| ascii_grid | str | Multi-line ASCII representation of hold positions |\n")
        f.write("| grid_min_x | int | Leftmost x-coordinate of any hold |\n")
        f.write("| grid_max_x | int | Rightmost x-coordinate of any hold |\n")
        f.write("| grid_min_y | int | Bottom-most y-coordinate of any hold |\n")
        f.write("| grid_max_y | int | Top-most y-coordinate of any hold |\n")
        f.write("|---\n\n")

        f.write("**ASCII Grid Legend**:\n\n")
        f.write("- `S` = Start piece ( foothold at beginning)\n")
        f.write("- `M` = Middle piece ( foothold in middle section)\n")
        f.write("- `F` = Finish piece ( foothold near end)\n")
        f.write("- `.` = Empty space (no hold)\n\n")

        f.write("### Derived Metrics\n\n")
        f.write("| Field | Type | Description |\n")
        f.write("|-------|------|-------------|\n")
        f.write("| num_holds | int | Total number of holds in layout |\n")
        f.write("| hold_sparsity | float | Holds per unit grid area (e.g., 0.15 = 15%) |\n")
        f.write("| max_reach | float | Maximum reach distance between consecutive holds |\n")
        f.write("| created_at | str | ISO timestamp when data was exported |\n")

    print(f"  Written data dictionary to {dict_path}")

    return {
        "jsonl_path": jsonl_path,
        "csv_path": csv_path,
        "dict_path": dict_path,
        "records_written": len(valid_records),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract climbing training data")
    parser.add_argument("--db", default="db/db.sqlite", help="Path to SQLite database")
    parser.add_argument("--layout-id", type=int, default=1, help="Filter by layout_id")
    parser.add_argument("--output-dir", default=".", help="Output directory")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of climbs for validation")

    args = parser.parse_args()

    DB_PATH = args.db
    OUTPUT_DIR = args.output_dir
    LAYOUT_ID = args.layout_id
    LIMIT = args.limit

    results = export_climbing_data(DB_PATH, LAYOUT_ID, OUTPUT_DIR, LIMIT)

    print("\n=== Summary ===")
    for key, value in results.items():
        print(f"{key}: {value}")
