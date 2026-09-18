"""Request schema (Problem Statement §07). Invalid requests are answered with 400."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, Field, StringConstraints, field_validator, model_validator

Finite = Annotated[float, Field(allow_inf_nan=False)]
NonNeg = Annotated[float, Field(ge=0, allow_inf_nan=False)]
Note = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)]


class HourEntry(BaseModel):
    hour: int = Field(ge=0, le=23)
    demand_kwh: NonNeg
    solar_kwh: NonNeg
    tariff_bdt_per_kwh: Finite


class Battery(BaseModel):
    capacity_kwh: NonNeg
    initial_energy_kwh: NonNeg
    minimum_energy_kwh: NonNeg
    max_charge_kwh_per_hour: NonNeg
    max_discharge_kwh_per_hour: NonNeg

    @model_validator(mode="after")
    def _within_capacity(self):
        if self.minimum_energy_kwh > self.capacity_kwh:
            raise ValueError("minimum_energy_kwh exceeds capacity_kwh")
        if self.initial_energy_kwh > self.capacity_kwh:
            raise ValueError("initial_energy_kwh exceeds capacity_kwh")
        return self


class OptimizeRequest(BaseModel):
    scenario_id: Annotated[str, StringConstraints(min_length=1, max_length=200)]
    operator_notes: list[Note] = Field(min_length=1, max_length=3)
    hours: list[HourEntry] = Field(min_length=24, max_length=24)
    battery: Battery

    @field_validator("hours")
    @classmethod
    def _all_hours(cls, v: list[HourEntry]):
        if sorted(h.hour for h in v) != list(range(24)):
            raise ValueError("hours must contain each hour 0-23 exactly once")
        return sorted(v, key=lambda h: h.hour)
