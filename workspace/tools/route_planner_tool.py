"""Route Planner Tool - Geocode addresses, calculate distances, find shortest route."""

import requests
import math
import time
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field
from itertools import combinations


class RoutePlannerArgs(BaseModel):
    addresses: List[str] = Field(
        ..., description="List of addresses or place names to route between (2-20 locations)"
    )
    return_to_start: bool = Field(
        False, description="Whether the route should return to the starting point (round trip)"
    )


class RoutePlannerTool:
    """
    Route planner: geocode addresses, compute pairwise distances, find shortest route.
    
    Features:
    - Geocode any address/place name to lat/lon (via OpenStreetMap Nominatim)
    - Haversine distance between all location pairs
    - Shortest route via nearest-neighbor TSP heuristic
    - Optional round-trip mode
    - Distance matrix output
    
    No API key required. Uses free OpenStreetMap Nominatim API.
    Rate-limited to 1 request/second per Nominatim policy.
    """
    
    name = "route_planner"
    description = """Geocode addresses to coordinates, calculate distances between all pairs, 
    and find the shortest route. Supports 2-20 locations. Free, no API key needed."""
    args_schema = RoutePlannerArgs

    GEOCODE_URL = "https://nominatim.openstreetmap.org/search"
    HEADERS = {"User-Agent": "AgenticSystem-RoutePlanner/1.0"}

    def run(self, addresses: List[str], return_to_start: bool = False) -> Dict[str, Any]:
        """Geocode addresses, compute distances, find shortest route."""

        if len(addresses) < 2:
            return {"success": False, "error": "Need at least 2 addresses."}
        if len(addresses) > 20:
            return {"success": False, "error": "Maximum 20 addresses supported."}

        # Step 1: Geocode all addresses
        locations = []
        failed = []
        for addr in addresses:
            coords = self._geocode(addr)
            if coords:
                locations.append({
                    "address": addr,
                    "resolved": coords["display_name"],
                    "lat": coords["lat"],
                    "lon": coords["lon"],
                })
            else:
                failed.append(addr)
            time.sleep(1.1)  # Nominatim rate limit: 1 req/sec

        if failed:
            return {
                "success": False,
                "error": f"Could not geocode: {', '.join(failed)}",
                "geocoded": locations,
            }

        n = len(locations)

        # Step 2: Distance matrix (haversine)
        dist_matrix = [[0.0] * n for _ in range(n)]
        pair_distances = []

        for i, j in combinations(range(n), 2):
            d = self._haversine(
                locations[i]["lat"], locations[i]["lon"],
                locations[j]["lat"], locations[j]["lon"],
            )
            dist_matrix[i][j] = d
            dist_matrix[j][i] = d
            pair_distances.append({
                "from": locations[i]["address"],
                "to": locations[j]["address"],
                "distance_km": round(d, 2),
            })

        pair_distances.sort(key=lambda x: x["distance_km"])

        # Step 3: Shortest route (nearest-neighbor TSP)
        route_indices, route_distance = self._nearest_neighbor_tsp(dist_matrix, return_to_start)

        route_steps = []
        for idx, ri in enumerate(route_indices):
            step = {
                "order": idx + 1,
                "address": locations[ri]["address"],
                "resolved": locations[ri]["resolved"],
                "lat": locations[ri]["lat"],
                "lon": locations[ri]["lon"],
            }
            if idx > 0:
                prev = route_indices[idx - 1]
                step["distance_from_prev_km"] = round(dist_matrix[prev][ri], 2)
            route_steps.append(step)

        # Build readable summary
        lines = [
            "=== Route Planner Results ===",
            f"Locations: {n}",
            f"Mode: {'Round trip' if return_to_start else 'One way'}",
            "",
            "--- Coordinates ---",
        ]
        for loc in locations:
            lines.append(f"  {loc['address']}: {loc['lat']:.5f}, {loc['lon']:.5f}")

        lines.append("")
        lines.append("--- All Pairwise Distances ---")
        for p in pair_distances:
            lines.append(f"  {p['from']} <-> {p['to']}: {p['distance_km']} km")

        lines.append("")
        lines.append(f"--- Shortest Route (total: {round(route_distance, 2)} km) ---")
        for step in route_steps:
            dist_str = f" [{step['distance_from_prev_km']} km]" if "distance_from_prev_km" in step else " [START]"
            lines.append(f"  {step['order']}. {step['address']}{dist_str}")

        closest = pair_distances[0] if pair_distances else None
        farthest = pair_distances[-1] if pair_distances else None
        if closest:
            lines.append("")
            lines.append(f"Closest pair: {closest['from']} <-> {closest['to']} ({closest['distance_km']} km)")
        if farthest:
            lines.append(f"Farthest pair: {farthest['from']} <-> {farthest['to']} ({farthest['distance_km']} km)")

        return {
            "success": True,
            "locations": locations,
            "distance_matrix_km": [[round(d, 2) for d in row] for row in dist_matrix],
            "pair_distances": pair_distances,
            "shortest_route": route_steps,
            "total_route_distance_km": round(route_distance, 2),
            "closest_pair": closest,
            "farthest_pair": farthest,
            "summary": "\n".join(lines),
        }

    def _geocode(self, address: str) -> Optional[Dict]:
        """Geocode an address using Nominatim."""
        try:
            resp = requests.get(
                self.GEOCODE_URL,
                params={"q": address, "format": "json", "limit": 1},
                headers=self.HEADERS,
                timeout=10,
            )
            resp.raise_for_status()
            results = resp.json()
            if not results:
                return None
            r = results[0]
            return {
                "lat": float(r["lat"]),
                "lon": float(r["lon"]),
                "display_name": r.get("display_name", address),
            }
        except Exception:
            return None

    @staticmethod
    def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate great-circle distance in km between two points."""
        R = 6371.0  # Earth radius in km
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        return R * 2 * math.asin(math.sqrt(a))

    @staticmethod
    def _nearest_neighbor_tsp(dist_matrix: List[List[float]], return_to_start: bool) -> Tuple[List[int], float]:
        """Nearest-neighbor heuristic for TSP. Tries all starting points, picks best."""
        n = len(dist_matrix)
        best_route = None
        best_dist = float("inf")

        for start in range(n):
            visited = [start]
            unvisited = set(range(n)) - {start}
            total = 0.0

            while unvisited:
                current = visited[-1]
                nearest = min(unvisited, key=lambda x: dist_matrix[current][x])
                total += dist_matrix[current][nearest]
                visited.append(nearest)
                unvisited.remove(nearest)

            if return_to_start:
                total += dist_matrix[visited[-1]][visited[0]]
                visited.append(visited[0])

            if total < best_dist:
                best_dist = total
                best_route = visited

        return best_route, best_dist


def test_tool():
    tool = RoutePlannerTool()
    result = tool.run([
        "Mumbai, India",
        "Pune, India",
        "Nashik, India",
        "Goa, India",
        "Bangalore, India",
    ])
    if result.get("summary"):
        print(result["summary"])
    else:
        print(result)


if __name__ == "__main__":
    test_tool()
