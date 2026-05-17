"""
Task 5 — Energy Optimization Recommendations
==============================================
Rule engine that analyses appliance contributions and generates
actionable, prioritised recommendations.
Works at inference time — no model training required.
"""
from __future__ import annotations

from src.config import HIGH_IMPACT_PCT, MODERATE_PCT, LOW_MEDIUM_PCT, HIGH_TOTAL_KW, ROOM_DISPLAY


def generate_recommendations(
    appliance_readings: dict,        # {canonical_room: kW_or_kwh_value}
    predicted_total_kw: float | None = None,
    house_meta: dict | None = None,  # optional metadata for context
) -> list[str]:
    """
    Generate actionable energy-saving recommendations.

    Parameters
    ----------
    appliance_readings  : {room_name: consumption_value}
    predicted_total_kw  : model-predicted total (uses sum if None)
    house_meta          : optional dict with n_acs, area_sqft, n_people etc.

    Returns
    -------
    List of human-readable recommendation strings, sorted by impact.
    """
    total = predicted_total_kw or sum(float(v) for v in appliance_readings.values())
    if total <= 0:
        return ["No energy data available — cannot generate recommendations."]

    recs: list[str] = []

    for room, val in sorted(appliance_readings.items(), key=lambda x: x[1], reverse=True):
        val   = float(val)
        pct   = val / total * 100
        label = ROOM_DISPLAY.get(room, room.replace("_", " ").title())

        if pct >= HIGH_IMPACT_PCT:
            recs.append(
                f"🔴 [HIGH IMPACT] {label} accounts for {pct:.1f}% of total energy "
                f"({val:.2f} kW). "
                f"→ Install a smart timer / programmable thermostat, upgrade to an "
                f"energy-efficient model (5-star rating), or reduce daily run-time by 20%."
            )
        elif pct >= MODERATE_PCT:
            recs.append(
                f"🟡 [MODERATE] {label} uses {pct:.1f}% of total energy ({val:.2f} kW). "
                f"→ Optimise the usage schedule to off-peak hours and consider a "
                f"programmable controller."
            )
        elif pct >= LOW_MEDIUM_PCT:
            recs.append(
                f"🟢 [LOW-MEDIUM] {label} uses {pct:.1f}% ({val:.2f} kW). "
                f"→ Maintain current usage; consider upgrading if the appliance is >10 years old."
            )

    # Total consumption caution
    if total > HIGH_TOTAL_KW:
        recs.append(
            f"⚡ [CAUTION] Total instantaneous load is high ({total:.1f} kW). "
            f"→ Consider a home energy audit, shift heavy appliances to off-peak hours "
            f"(10 PM – 6 AM), and assess solar panel feasibility."
        )

    # Metadata-driven tips
    if house_meta:
        n_acs = int(house_meta.get("n_acs", 0))
        n_ppl = int(house_meta.get("n_people", 0))
        if n_acs >= 3:
            recs.append(
                f"❄ [TIP] {n_acs} AC units detected. Ensure all are serviced annually "
                f"and set thermostats to 24–26 °C for optimal efficiency."
            )
        if n_ppl >= 6:
            recs.append(
                f"👥 [TIP] {n_ppl} occupants — consider a time-of-use tariff plan to "
                f"reduce peak-hour costs."
            )

    if not recs:
        recs.append(
            "✅ Energy distribution looks balanced. No high-impact recommendations at this time."
        )

    return recs
