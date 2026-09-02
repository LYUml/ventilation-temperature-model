"""Reusable RDF-informed baseline indoor-temperature models.

The public API deliberately keeps weather and indoor-temperature history as
separate tables.  A fitted model never reads the repository's bundled CSVs and
does not write evaluation artifacts, so it can be imported by another project.
"""

from __future__ import annotations

import math
import pickle
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .common.rdf import extract_building

Method = Literal["rdf_rc_narx_ridge", "rdf_kernel_narx_ridge"]
SUPPORTED_METHODS: tuple[Method, ...] = (
    "rdf_rc_narx_ridge",
    "rdf_kernel_narx_ridge",
)
DEFAULT_HORIZON = 24
LAGS = (1, 2, 3, 6, 12, 24)
RIDGE_ALPHAS = (0.1, 1.0, 10.0, 100.0, 1000.0)
_DATE_PARTS = {"year", "mon", "month", "day", "hour"}
_OUTDOOR_CANDIDATES = ("Ta", "TA", "TEM", "OUTDOOR", "outdoor_temperature")


def _timestamped(frame: pd.DataFrame, name: str, timestamp_column: str) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame")
    result = frame.copy()
    if timestamp_column in result:
        timestamp = pd.to_datetime(result[timestamp_column], errors="raise")
    elif isinstance(result.index, pd.DatetimeIndex):
        timestamp = pd.Series(result.index, index=result.index)
    elif {"year", "mon", "day", "hour"}.issubset(result.columns):
        timestamp = pd.to_datetime(
            {
                "year": result["year"],
                "month": result["mon"],
                "day": result["day"],
                "hour": result["hour"],
            },
            errors="raise",
        )
    else:
        raise ValueError(
            f"{name} needs a {timestamp_column!r} column, a DatetimeIndex, "
            "or year/mon/day/hour columns"
        )
    result["_base_ta_timestamp"] = np.asarray(timestamp)
    if result["_base_ta_timestamp"].duplicated().any():
        raise ValueError(f"{name} contains duplicate timestamps")
    if not result["_base_ta_timestamp"].is_monotonic_increasing:
        raise ValueError(f"{name} timestamps must be increasing")
    gaps = result["_base_ta_timestamp"].diff().dropna()
    if len(gaps) and not (gaps == pd.Timedelta(hours=1)).all():
        raise ValueError(f"{name} must be a continuous hourly series")
    return result.reset_index(drop=True)


def _finite_numeric(
    frame: pd.DataFrame, columns: Sequence[str], name: str
) -> np.ndarray:
    try:
        values = frame[list(columns)].to_numpy(dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} columns must be numeric: {list(columns)}") from error
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains missing or non-finite values")
    return values


def _cyclic_hour(timestamp: pd.Timestamp) -> tuple[float, float]:
    angle = 2.0 * math.pi * timestamp.hour / 24.0
    return math.sin(angle), math.cos(angle)


def _ewma(values: np.ndarray, alpha: float) -> np.ndarray:
    result = np.empty(len(values), dtype=float)
    result[0] = values[0]
    for index in range(1, len(values)):
        result[index] = alpha * result[index - 1] + (1.0 - alpha) * values[index]
    return result


class BaseTaModel:
    """Train and reuse one of the two RDF-informed baseline models."""

    def __init__(
        self,
        method: Method = "rdf_kernel_narx_ridge",
        *,
        horizon: int = DEFAULT_HORIZON,
        timestamp_column: str = "Timestamp",
        outdoor_column: str | None = None,
    ) -> None:
        if method not in SUPPORTED_METHODS:
            raise ValueError(
                f"method must be one of {SUPPORTED_METHODS}, got {method!r}"
            )
        if horizon < 1:
            raise ValueError("horizon must be positive")
        self.method = method
        self.horizon = int(horizon)
        self.timestamp_column = timestamp_column
        self.outdoor_column = outdoor_column
        self.is_fitted = False

    def fit(
        self,
        inputRdf: str | Path,
        weatherHistory: pd.DataFrame,
        indoorHistory: pd.DataFrame,
        targetSpaces: Sequence[str] | None = None,
    ) -> BaseTaModel:
        """Fit using aligned hourly weather and measured indoor temperatures."""
        weather = _timestamped(weatherHistory, "weatherHistory", self.timestamp_column)
        indoor = _timestamped(indoorHistory, "indoorHistory", self.timestamp_column)
        if not weather["_base_ta_timestamp"].equals(indoor["_base_ta_timestamp"]):
            raise ValueError(
                "weatherHistory and indoorHistory timestamps must match exactly"
            )

        building = extract_building(Path(inputRdf), target_names=[])
        rdf_spaces = building["spaces"]
        state_spaces = [name for name in indoor.columns if name in rdf_spaces]
        if not state_spaces:
            raise ValueError(
                "indoorHistory has no column whose name matches an RDF space"
            )
        targets = list(targetSpaces) if targetSpaces is not None else list(state_spaces)
        if not targets:
            raise ValueError("targetSpaces cannot be empty")
        missing = [name for name in targets if name not in state_spaces]
        if missing:
            raise ValueError(
                "Every target space must exist in the RDF and indoorHistory; "
                f"missing {missing}"
            )

        excluded = {self.timestamp_column, "_base_ta_timestamp", *_DATE_PARTS}
        weather_columns = [
            name
            for name in weather.columns
            if name not in excluded and pd.api.types.is_numeric_dtype(weather[name])
        ]
        if not weather_columns:
            raise ValueError("weatherHistory has no numeric weather columns")
        outdoor_name = self.outdoor_column
        if outdoor_name is None:
            outdoor_name = next(
                (name for name in _OUTDOOR_CANDIDATES if name in weather_columns), None
            )
        if outdoor_name not in weather_columns:
            raise ValueError(
                "Cannot identify outdoor temperature; pass outdoor_column to BaseTaModel"
            )

        weather_values = _finite_numeric(weather, weather_columns, "weatherHistory")
        indoor_values = _finite_numeric(indoor, state_spaces, "indoorHistory")
        minimum = max(5 * self.horizon, 2 * max(LAGS) + 2)
        if len(weather) < minimum:
            raise ValueError(
                f"At least {minimum} aligned hourly history rows are required; got {len(weather)}"
            )

        self.input_rdf = str(Path(inputRdf).resolve())
        self.weather_columns = weather_columns
        self.outdoor_column_ = outdoor_name
        self.outdoor_index_ = weather_columns.index(outdoor_name)
        self.state_spaces = state_spaces
        self.target_spaces = targets
        self.space_index_ = {name: index for index, name in enumerate(state_spaces)}
        self.static_features_, self.neighbours_ = self._rdf_features(rdf_spaces)
        self.history_timestamps_ = weather["_base_ta_timestamp"].tolist()
        self.weather_history_ = weather_values
        self.indoor_history_ = indoor_values

        if self.method == "rdf_kernel_narx_ridge":
            self._fit_kernel_model()
        else:
            self._fit_rc_model()
        self.is_fitted = True
        return self

    def predict(self, forecastWeather: pd.DataFrame) -> dict[str, list[float]]:
        """Predict target-space temperatures for 1..horizon future hours."""
        self._require_fitted()
        weather = _timestamped(
            forecastWeather, "forecastWeather", self.timestamp_column
        )
        if not 1 <= len(weather) <= self.horizon:
            raise ValueError(
                f"forecastWeather must contain between 1 and {self.horizon} rows"
            )
        expected_start = self.history_timestamps_[-1] + pd.Timedelta(hours=1)
        if weather["_base_ta_timestamp"].iloc[0] != expected_start:
            raise ValueError(
                f"forecastWeather must start at {expected_start.isoformat()}"
            )
        missing = [name for name in self.weather_columns if name not in weather]
        if missing:
            raise ValueError(
                f"forecastWeather is missing fitted weather columns: {missing}"
            )
        future = _finite_numeric(weather, self.weather_columns, "forecastWeather")
        timestamps = weather["_base_ta_timestamp"].tolist()
        if self.method == "rdf_kernel_narx_ridge":
            prediction = self._predict_kernel(future, timestamps)
        else:
            prediction = self._predict_rc(future, timestamps)
        return {
            name: prediction[:, self.space_index_[name]].astype(float).tolist()
            for name in self.target_spaces
        }

    def save(self, path: str | Path) -> Path:
        """Save a fitted model. Only load model files from trusted sources."""
        self._require_fitted()
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("wb") as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return destination

    @classmethod
    def load(cls, path: str | Path) -> BaseTaModel:
        """Load a model saved by :meth:`save` from a trusted source."""
        with Path(path).open("rb") as handle:
            model = pickle.load(handle)
        if not isinstance(model, cls):
            raise TypeError("The model file does not contain a BaseTaModel")
        model._require_fitted()
        return model

    def _require_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError(
                "BaseTaModel.fit must be called before prediction or saving"
            )

    def _rdf_features(
        self, rdf_spaces: dict[str, dict[str, Any]]
    ) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
        static = []
        neighbours: list[tuple[np.ndarray, np.ndarray]] = []
        for name in self.state_spaces:
            space = rdf_spaces[name]
            static.append(
                [
                    float(space.get("floor_area_m2") or 0.0),
                    float(space.get("volume_m3") or 0.0),
                    float(space.get("base_height_m") or 0.0),
                    float(space.get("exterior_ua_w_k") or 0.0),
                    float(space.get("operable_window_area_m2") or 0.0),
                ]
            )
            weights: dict[int, float] = {}
            for edge in space.get("adjacent_spaces", []):
                if edge["space"] not in self.space_index_:
                    continue
                index = self.space_index_[edge["space"]]
                weight = float(edge.get("ua_w_k") or 0.0)
                weights[index] = weights.get(index, 0.0) + max(weight, 0.0)
            if weights:
                indices = np.asarray(sorted(weights), dtype=int)
                values = np.asarray([weights[index] for index in indices], dtype=float)
                if values.sum() == 0.0:
                    values[:] = 1.0
                values /= values.sum()
            else:
                indices = np.asarray([self.space_index_[name]], dtype=int)
                values = np.asarray([1.0])
            neighbours.append((indices, values))
        return np.asarray(static, dtype=float), neighbours

    def _one_hot(self, space: int) -> np.ndarray:
        result = np.zeros(len(self.state_spaces), dtype=float)
        result[space] = 1.0
        return result

    def _neighbour_value(self, values: np.ndarray, time: int, space: int) -> float:
        indices, weights = self.neighbours_[space]
        return float(values[time, indices] @ weights)

    def _select_kernel_alpha(
        self, outdoor: np.ndarray, indoor: np.ndarray, end: int
    ) -> float:
        inner = max(2, int(end * 0.75))
        best = (float("inf"), 0.9)
        for alpha in np.linspace(0.5, 0.995, 100):
            state = _ewma(outdoor, float(alpha))
            errors = []
            for space in range(indoor.shape[1]):
                coefficient = np.linalg.lstsq(
                    np.c_[np.ones(inner), state[:inner]],
                    indoor[:inner, space],
                    rcond=None,
                )[0]
                predicted = coefficient[0] + coefficient[1] * state[inner:end]
                errors.append(np.mean((predicted - indoor[inner:end, space]) ** 2))
            score = float(np.sqrt(np.mean(errors)))
            if score < best[0]:
                best = (score, float(alpha))
        return best[1]

    def _kernel_rows(
        self,
        residual: np.ndarray,
        weather: np.ndarray,
        timestamps: Sequence[pd.Timestamp],
        starts: range,
        lag: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        rows, targets = [], []
        for time in starts:
            sin_hour, cos_hour = _cyclic_hour(timestamps[time])
            for space in range(len(self.state_spaces)):
                self_lags = [
                    residual[time - offset, space] for offset in range(1, lag + 1)
                ]
                neighbour_lags = [
                    self._neighbour_value(residual, time - offset, space)
                    for offset in range(1, lag + 1)
                ]
                rows.append(
                    np.r_[
                        self_lags,
                        neighbour_lags,
                        weather[time],
                        sin_hour,
                        cos_hour,
                        self.static_features_[space],
                        self._one_hot(space),
                    ]
                )
                targets.append(residual[time, space])
        return np.asarray(rows), np.asarray(targets)

    def _fit_kernel_model(self) -> None:
        weather, indoor = self.weather_history_, self.indoor_history_
        outdoor = weather[:, self.outdoor_index_]
        train_end = int(len(indoor) * 0.80)
        self.kernel_alpha_ = self._select_kernel_alpha(outdoor, indoor, train_end)
        state = _ewma(outdoor, self.kernel_alpha_)
        selection_coefficients = np.asarray(
            [
                np.linalg.lstsq(
                    np.c_[np.ones(train_end), state[:train_end]],
                    indoor[:train_end, space],
                    rcond=None,
                )[0]
                for space in range(indoor.shape[1])
            ]
        )
        selection_base = selection_coefficients[:, 0] + np.outer(
            state, selection_coefficients[:, 1]
        )
        selection_residual = indoor - selection_base
        candidates = []
        for lag in (1, 2, 3, 6, 12, 24):
            train_x, train_y = self._kernel_rows(
                selection_residual,
                weather,
                self.history_timestamps_,
                range(lag, train_end),
                lag,
            )
            val_x, val_y = self._kernel_rows(
                selection_residual,
                weather,
                self.history_timestamps_,
                range(train_end, len(indoor)),
                lag,
            )
            for alpha in RIDGE_ALPHAS:
                model = make_pipeline(StandardScaler(), Ridge(alpha=alpha)).fit(
                    train_x, train_y
                )
                score = float(np.sqrt(np.mean((model.predict(val_x) - val_y) ** 2)))
                candidates.append((score, lag, alpha))
        _, self.lag_, self.ridge_alpha_ = min(candidates, key=lambda item: item[0])

        self.kernel_coefficients_ = np.asarray(
            [
                np.linalg.lstsq(
                    np.c_[np.ones(len(state)), state], indoor[:, space], rcond=None
                )[0]
                for space in range(indoor.shape[1])
            ]
        )
        base = self.kernel_coefficients_[:, 0] + np.outer(
            state, self.kernel_coefficients_[:, 1]
        )
        residual = indoor - base
        x, y = self._kernel_rows(
            residual,
            weather,
            self.history_timestamps_,
            range(self.lag_, len(indoor)),
            self.lag_,
        )
        self.residual_model_ = make_pipeline(
            StandardScaler(), Ridge(alpha=self.ridge_alpha_)
        ).fit(x, y)
        self.kernel_state_last_ = float(state[-1])
        self.residual_history_ = residual[-self.lag_ :].copy()

    def _predict_kernel(
        self, future: np.ndarray, timestamps: Sequence[pd.Timestamp]
    ) -> np.ndarray:
        residual_history = [row.copy() for row in self.residual_history_]
        kernel_state = self.kernel_state_last_
        result = []
        for step, weather_row in enumerate(future):
            kernel_state = (
                self.kernel_alpha_ * kernel_state
                + (1.0 - self.kernel_alpha_) * weather_row[self.outdoor_index_]
            )
            base = (
                self.kernel_coefficients_[:, 0]
                + self.kernel_coefficients_[:, 1] * kernel_state
            )
            sin_hour, cos_hour = _cyclic_hour(timestamps[step])
            residual_row = np.empty(len(self.state_spaces), dtype=float)
            history_array = np.asarray(residual_history)
            for space in range(len(self.state_spaces)):
                self_lags = [
                    history_array[-offset, space] for offset in range(1, self.lag_ + 1)
                ]
                indices, weights = self.neighbours_[space]
                neighbour_lags = [
                    float(history_array[-offset, indices] @ weights)
                    for offset in range(1, self.lag_ + 1)
                ]
                features = np.r_[
                    self_lags,
                    neighbour_lags,
                    weather_row,
                    sin_hour,
                    cos_hour,
                    self.static_features_[space],
                    self._one_hot(space),
                ]
                residual_row[space] = self.residual_model_.predict(features[None])[0]
            residual_history.append(residual_row)
            result.append(base + residual_row)
        return np.asarray(result)

    @staticmethod
    def _fit_rc_parameters(
        indoor: np.ndarray, outdoor: np.ndarray, end: int
    ) -> np.ndarray:
        parameters = []
        for space in range(indoor.shape[1]):
            delta = indoor[1:end, space] - indoor[: end - 1, space]
            gap = outdoor[: end - 1] - indoor[: end - 1, space]
            gain, response = np.linalg.lstsq(
                np.c_[np.ones(len(delta)), gap], delta, rcond=None
            )[0]
            parameters.append([float(gain), float(np.clip(response, 1.0e-5, 0.3))])
        return np.asarray(parameters)

    @staticmethod
    def _rc_rollout(
        parameters: np.ndarray, initial: float, outdoor: np.ndarray
    ) -> np.ndarray:
        gain, response = parameters
        current = float(initial)
        result = []
        for outside in outdoor:
            current += gain + response * (outside - current)
            result.append(current)
        return np.asarray(result)

    def _rc_residuals(self, parameters: np.ndarray) -> np.ndarray:
        indoor = self.indoor_history_
        outdoor = self.weather_history_[:, self.outdoor_index_]
        result = np.zeros_like(indoor)
        for space in range(indoor.shape[1]):
            one_step = (
                indoor[:-1, space]
                + parameters[space, 0]
                + parameters[space, 1] * (outdoor[1:] - indoor[:-1, space])
            )
            result[1:, space] = indoor[1:, space] - one_step
        return result

    def _padded_weather(self, weather: np.ndarray) -> np.ndarray:
        if len(weather) == self.horizon:
            return weather
        return np.vstack(
            [weather, np.repeat(weather[-1][None], self.horizon - len(weather), axis=0)]
        )

    def _rc_feature(
        self,
        origin: int,
        space: int,
        weather_window: np.ndarray,
        timestamp: pd.Timestamp,
        residual: np.ndarray,
        indoor: np.ndarray,
    ) -> np.ndarray:
        sin_hour, cos_hour = _cyclic_hour(timestamp)
        residual_lags = [residual[origin - lag, space] for lag in LAGS]
        neighbour_lags = [
            self._neighbour_value(indoor, origin - lag, space) for lag in LAGS
        ]
        return np.r_[
            weather_window.reshape(-1),
            sin_hour,
            cos_hour,
            residual_lags,
            neighbour_lags,
            self.static_features_[space],
            self._one_hot(space),
        ]

    def _rc_dataset(
        self,
        starts: Sequence[int],
        parameters: np.ndarray,
        residual: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        rows, targets = [], []
        weather = self.weather_history_
        indoor = self.indoor_history_
        outdoor = weather[:, self.outdoor_index_]
        for origin in starts:
            weather_window = weather[origin : origin + self.horizon]
            for space in range(indoor.shape[1]):
                base = self._rc_rollout(
                    parameters[space],
                    indoor[origin - 1, space],
                    outdoor[origin : origin + self.horizon],
                )
                rows.append(
                    self._rc_feature(
                        origin,
                        space,
                        weather_window,
                        self.history_timestamps_[origin],
                        residual,
                        indoor,
                    )
                )
                targets.append(indoor[origin : origin + self.horizon, space] - base)
        return np.asarray(rows), np.asarray(targets)

    def _fit_rc_model(self) -> None:
        indoor = self.indoor_history_
        outdoor = self.weather_history_[:, self.outdoor_index_]
        train_end = int(len(indoor) * 0.80)
        selection_parameters = self._fit_rc_parameters(indoor, outdoor, train_end)
        selection_residual = self._rc_residuals(selection_parameters)
        train_starts = list(range(max(LAGS), train_end - self.horizon + 1))
        val_starts = list(range(train_end, len(indoor) - self.horizon + 1))
        train_x, train_y = self._rc_dataset(
            train_starts, selection_parameters, selection_residual
        )
        val_x, val_y = self._rc_dataset(
            val_starts, selection_parameters, selection_residual
        )
        candidates = []
        for alpha in RIDGE_ALPHAS:
            model = make_pipeline(StandardScaler(), Ridge(alpha=alpha)).fit(
                train_x, train_y
            )
            score = float(np.sqrt(np.mean((model.predict(val_x) - val_y) ** 2)))
            candidates.append((score, alpha))
        _, self.ridge_alpha_ = min(candidates, key=lambda item: item[0])

        self.rc_parameters_ = self._fit_rc_parameters(indoor, outdoor, len(indoor))
        residual = self._rc_residuals(self.rc_parameters_)
        all_starts = list(range(max(LAGS), len(indoor) - self.horizon + 1))
        x, y = self._rc_dataset(all_starts, self.rc_parameters_, residual)
        self.residual_model_ = make_pipeline(
            StandardScaler(), Ridge(alpha=self.ridge_alpha_)
        ).fit(x, y)
        self.rc_residual_history_ = residual

    def _predict_rc(
        self, future: np.ndarray, timestamps: Sequence[pd.Timestamp]
    ) -> np.ndarray:
        count = len(future)
        padded = self._padded_weather(future)
        outdoor = padded[:, self.outdoor_index_]
        result = np.empty((count, len(self.state_spaces)), dtype=float)
        origin = len(self.indoor_history_)
        for space in range(len(self.state_spaces)):
            base = self._rc_rollout(
                self.rc_parameters_[space], self.indoor_history_[-1, space], outdoor
            )
            features = self._rc_feature(
                origin,
                space,
                padded,
                timestamps[0],
                self.rc_residual_history_,
                self.indoor_history_,
            )
            correction = self.residual_model_.predict(features[None])[0]
            result[:, space] = (base + correction)[:count]
        return result


def calculateBaseTa(
    inputRdf: str | Path,
    weatherHistory: pd.DataFrame,
    indoorHistory: pd.DataFrame,
    forecastWeather: pd.DataFrame,
    targetSpaces: Sequence[str] | None = None,
    method: Method = "rdf_kernel_narx_ridge",
    *,
    horizon: int = DEFAULT_HORIZON,
    timestamp_column: str = "Timestamp",
    outdoor_column: str | None = None,
) -> dict[str, list[float]]:
    """Fit and predict in one call; use :class:`BaseTaModel` for repeated calls."""
    model = BaseTaModel(
        method=method,
        horizon=horizon,
        timestamp_column=timestamp_column,
        outdoor_column=outdoor_column,
    )
    model.fit(inputRdf, weatherHistory, indoorHistory, targetSpaces)
    return model.predict(forecastWeather)


calculate_base_ta = calculateBaseTa


__all__ = [
    "SUPPORTED_METHODS",
    "BaseTaModel",
    "Method",
    "calculateBaseTa",
    "calculate_base_ta",
]
