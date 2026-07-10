import re

import geopandas as gpd
import pandas as pd

from ra2ce.analysis.damages.shape_to_integrate_object.to_Integrate_shaper_protocol import (
    ToIntegrateShaperProtocol,
)


class ManToIntegrateShaper(ToIntegrateShaperProtocol):
    gdf: gpd.GeoDataFrame

    def __init__(self, gdf):
        self.gdf = gdf

    @staticmethod
    def _extract_columns_by_pattern(
        pattern_text: str, gdf: gpd.GeoDataFrame
    ) -> set[str]:
        """
        Extract column names based on the provided regex pattern.

        Args:
            pattern_text (Pattern[str]): The compiled regex pattern to match the column names.
            df (pd.DataFrame): The DataFrame from which to extract the RP values.

        Returns:
            Set[str]: A set of RP values extracted from the column names.
        """
        pattern = re.compile(pattern_text)
        columns = {pattern.search(c).group(1) for c in gdf.columns if pattern.search(c)}
        return columns

    def get_return_periods(self) -> list:
        # Extract the RP values from the columns
        rp_values = ManToIntegrateShaper._extract_columns_by_pattern(
            pattern_text=r"RP(\d+)", gdf=self.gdf
        )
        if not rp_values:
            raise ValueError("No damage column with RP found")

        return sorted([float(rp) for rp in rp_values])

    def shape_to_integrate_object(self, return_periods: list) -> dict[str, gpd.GeoDataFrame]:
        """
        Build per-vulnerability DataFrames for EAD integration from columns like:
        dam_EV1_ro, dam_EVTF12_ro, dam_EV1_ra, ...

        Parameters
        ----------
        return_periods : list
            Expected to be a list of HazardEvent-like objects carrying:
            - event_id
            - event_probability
            Optionally supports:
            - list of (event_id, return_period) tuples
            - dict[event_id] = return_period

        Returns
        -------
        dict[str, GeoDataFrame]
            Key = vulnerability suffix (e.g. "ro"), value = dataframe with
            columns renamed to numeric return periods, sorted descending.
        """
        from collections import defaultdict
        from numbers import Number

        if self.gdf is None or self.gdf.empty:
            return {}

        # 1) Parse damage columns
        pattern = re.compile(r"^dam_EV(?P<event_id>[^_]+)_(?P<suffix>.+)$")
        by_suffix: dict[str, list[tuple[str, str]]] = defaultdict(list)
        ordered_event_ids: list[str] = []

        for col in self.gdf.columns:
            m = pattern.match(str(col))
            if not m:
                continue
            event_id = m.group("event_id")
            suffix = m.group("suffix")
            by_suffix[suffix].append((event_id, str(col)))
            if event_id not in ordered_event_ids:
                ordered_event_ids.append(event_id)

        if not by_suffix:
            return {}

        # 2) Build event_id -> return_period map
        event_to_rp: dict[str, float] = {}

        def _add_event_mapping(event_id, rp_value):
            if event_id is None or rp_value is None:
                return
            try:
                rp = float(rp_value)
            except (TypeError, ValueError):
                return
            if rp > 0:
                event_to_rp[str(event_id)] = rp

        # 2a) Preferred: self.hazard_events
        hazard_events = getattr(self, "hazard_events", None) or []
        for ev in hazard_events:
            event_id = getattr(ev, "event_id", None)
            prob = getattr(ev, "event_probability", None)
            if event_id is None or prob in (None, 0):
                continue
            try:
                rp = 1.0 / float(prob)
            except (TypeError, ValueError, ZeroDivisionError):
                continue
            _add_event_mapping(event_id, rp)

        # 2b) Fallback: return_periods input
        if not event_to_rp and return_periods:
            first = return_periods[0]

            # HazardEvent-like list
            if hasattr(first, "event_id") and hasattr(first, "event_probability"):
                for ev in return_periods:
                    event_id = getattr(ev, "event_id", None)
                    prob = getattr(ev, "event_probability", None)
                    if event_id is None or prob in (None, 0):
                        continue
                    try:
                        rp = 1.0 / float(prob)
                    except (TypeError, ValueError, ZeroDivisionError):
                        continue
                    _add_event_mapping(event_id, rp)

            # Numeric list aligned with event ids detected in columns
            elif all(isinstance(x, Number) for x in return_periods):
                vals = [float(x) for x in return_periods if float(x) > 0]
                if len(vals) != len(ordered_event_ids):
                    raise ValueError(
                        "Cannot map return_periods to event ids: "
                        f"{len(vals)} values for {len(ordered_event_ids)} event ids "
                        f"{ordered_event_ids}."
                    )

                # If all <= 1, interpret as probabilities; else as return periods
                treat_as_prob = all(v <= 1.0 for v in vals)
                for eid, v in zip(ordered_event_ids, vals):
                    rp = (1.0 / v) if treat_as_prob else v
                    _add_event_mapping(eid, rp)

        # Ensure every event id in columns has a mapping
        missing = [eid for eid in ordered_event_ids if eid not in event_to_rp]
        if missing:
            raise ValueError(
                "Missing return-period mapping for event ids "
                f"{missing}. Provide hazard_events with event_id/event_probability "
                "or provide numeric return_periods aligned with column event order."
            )

        # 3) Build output dict per vulnerability suffix
        out: dict[str, gpd.GeoDataFrame] = {}

        for suffix, event_cols in by_suffix.items():
            selected_cols = []
            rp_cols = []

            for event_id, col in event_cols:
                rp = event_to_rp[event_id]
                selected_cols.append(col)
                rp_cols.append(float(rp))

            df = self.gdf[selected_cols].copy()
            df.columns = rp_cols

            # Collapse duplicate RP columns if present
            if df.columns.duplicated().any():
                df = df.T.groupby(level=0).max().T

            # Optional cleanup: remove rows where all mapped event damages are NaN
            df = df.dropna(axis=0, how="all")

            out[suffix] = df.sort_index(axis="columns", ascending=False)

        return out