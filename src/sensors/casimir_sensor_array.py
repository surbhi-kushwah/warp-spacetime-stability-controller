"""
Casimir-Enhanced Sensor Arrays

This module implements ultra-high precision Casimir force sensor arrays for
real-time spacetime curvature detection and warp field stability monitoring,
leveraging 847× metamaterial amplification.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from scipy.optimize import minimize
from scipy.signal import butter, filtfilt
import logging
from dataclasses import dataclass
import time

@dataclass
class SensorConfig:
    """Configuration for Casimir sensor arrays."""
    sensor_spacing: float = 1e-6      # Sensor spacing (m)
    plate_separation: float = 1e-9    # Casimir plate separation (m)
    sensor_area: float = 1e-6         # Sensor area (m²)
    temperature_target: float = 0.01  # Target temperature (K)
    pressure_target: float = 1e-6     # Target pressure (Pa)
    response_time_target: float = 1e-3  # Target response time (s)

@dataclass
class SensorReading:
    """Individual sensor reading with metadata."""
    sensor_id: int
    position: Tuple[float, float, float]
    casimir_force: float
    temperature: float
    pressure: float
    timestamp: float
    quality_factor: float

class CasimirSensorArray:
    """
    Ultra-high precision Casimir force sensor arrays for warp field monitoring.
    
    Key Features:
    - ±0.01 K temperature precision
    - ≤10⁻⁶ Pa vacuum monitoring
    - <1ms synchronization
    - Sub-nanometer spacetime perturbation sensing
    - 847× metamaterial amplification
    """
    
    def __init__(self,
                 config: SensorConfig,
                 array_size: Tuple[int, int, int] = (10, 10, 1),
                 metamaterial_amplification: float = 847.0,
                 polymer_parameter: float = 0.1):
        """
        Initialize Casimir sensor array.
        
        Args:
            config: Sensor configuration parameters
            array_size: Number of sensors in (x, y, z) directions
            metamaterial_amplification: Metamaterial enhancement factor
            polymer_parameter: Polymer correction parameter μ
        """
        self.config = config
        self.array_size = array_size
        self.metamaterial_amp = metamaterial_amplification
        self.mu = polymer_parameter
        
        # Physical constants
        self.hbar = 1.054571817e-34  # Reduced Planck constant (J·s)
        self.c = 299792458.0         # Speed of light (m/s)
        self.k_B = 1.380649e-23      # Boltzmann constant (J/K)
        self.epsilon_0 = 8.8541878128e-12  # Vacuum permittivity (F/m)
        
        # Initialize sensor array
        self.sensors = self._initialize_sensor_positions()
        self.n_sensors = len(self.sensors)
        
        # Data storage
        self.sensor_readings = []
        self.stability_metrics = {}
        
        # Filter for noise reduction
        self.filter_order = 4
        self.cutoff_frequency = 1000.0  # Hz
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initialized {self.n_sensors} Casimir sensors with {self.metamaterial_amp}× amplification")
        
    def _initialize_sensor_positions(self) -> List[Tuple[float, float, float]]:
        """Initialize sensor positions in 3D array."""
        sensors = []
        nx, ny, nz = self.array_size
        
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    x = i * self.config.sensor_spacing
                    y = j * self.config.sensor_spacing
                    z = k * self.config.sensor_spacing
                    sensors.append((x, y, z))
        
        return sensors
    
    def theoretical_casimir_force(self,
                                plate_separation: float,
                                plate_area: float,
                                temperature: float = 0.0) -> float:
        """
        Compute theoretical Casimir force between parallel plates.
        
        F = -π²ℏc A / (240 d⁴) for T=0
        With temperature corrections and metamaterial enhancement.
        
        Args:
            plate_separation: Distance between plates (m)
            plate_area: Area of plates (m²)
            temperature: Temperature (K)
            
        Returns:
            Casimir force (N), negative for attractive
        """
        # Zero-temperature Casimir force
        F_0 = -(np.pi**2 * self.hbar * self.c * plate_area) / (240 * plate_separation**4)
        
        # Temperature correction (simplified)
        if temperature > 0:
            thermal_length = self.hbar * self.c / (self.k_B * temperature)
            if plate_separation > thermal_length:
                temp_factor = 1.0 - (plate_separation / thermal_length)**2
                F_0 *= temp_factor
        
        # Metamaterial enhancement with polymer correction
        polymer_factor = self._sinc(np.pi * self.mu)
        enhanced_force = F_0 * self.metamaterial_amp * polymer_factor
        
        return enhanced_force
    
    def spacetime_curvature_coupling(self,
                                   metric_perturbation: np.ndarray,
                                   sensor_position: Tuple[float, float, float]) -> float:
        """
        Compute coupling between spacetime curvature and Casimir force.
        
        δF/F₀ = α_coupling · h_μν · f_coupling(x,y,z)
        
        Args:
            metric_perturbation: 4×4 metric perturbation tensor h_μν
            sensor_position: Sensor location (x, y, z)
            
        Returns:
            Fractional change in Casimir force
        """
        # Ensure proper metric shape
        if metric_perturbation.shape != (4, 4):
            metric_perturbation = np.zeros((4, 4))
        
        # Coupling strength (dimensionless)
        alpha_coupling = 1e-3  # Typical gravitational coupling
        
        # Spatial coupling function
        x, y, z = sensor_position
        spatial_factor = np.exp(-((x**2 + y**2 + z**2) / (1e-6)**2))
        
        # Trace of spatial part of metric perturbation
        h_spatial_trace = np.trace(metric_perturbation[1:, 1:])
        
        # Fractional force change
        delta_F_over_F = alpha_coupling * h_spatial_trace * spatial_factor
        
        # Polymer enhancement
        delta_F_over_F *= self._sinc(np.pi * self.mu)
        
        return delta_F_over_F
    
    def acquire_sensor_data(self,
                          warp_field_state: Optional[Dict[str, np.ndarray]] = None,
                          noise_level: float = 1e-15) -> List[SensorReading]:
        """
        Acquire data from all sensors with realistic noise and field coupling.
        
        Args:
            warp_field_state: Current warp field configuration
            noise_level: Sensor noise level (N)
            
        Returns:
            List of sensor readings
        """
        start_time = time.time()
        current_timestamp = start_time
        
        readings = []
        
        for sensor_id, position in enumerate(self.sensors):
            # Theoretical Casimir force
            F_theoretical = self.theoretical_casimir_force(
                self.config.plate_separation,
                self.config.sensor_area,
                self.config.temperature_target
            )
            
            # Warp field coupling
            if warp_field_state is not None:
                metric_perturbation = warp_field_state.get('metric_perturbation', np.zeros((4, 4)))
                curvature_coupling = self.spacetime_curvature_coupling(metric_perturbation, position)
                F_coupled = F_theoretical * (1.0 + curvature_coupling)
            else:
                F_coupled = F_theoretical
            
            # Add realistic noise
            noise = np.random.normal(0, noise_level)
            F_measured = F_coupled + noise
            
            # Environmental parameters with realistic variations
            temperature = self.config.temperature_target + np.random.normal(0, 0.005)  # ±5mK
            pressure = self.config.pressure_target + np.random.normal(0, 1e-7)  # ±0.1μPa
            
            # Quality factor (signal-to-noise ratio)
            quality_factor = abs(F_measured) / noise_level if noise_level > 0 else 1e6
            
            reading = SensorReading(
                sensor_id=sensor_id,
                position=position,
                casimir_force=F_measured,
                temperature=temperature,
                pressure=pressure,
                timestamp=current_timestamp,
                quality_factor=quality_factor
            )
            
            readings.append(reading)
        
        # Update data storage
        self.sensor_readings.extend(readings)
        
        # Performance check
        acquisition_time = time.time() - start_time
        meets_timing = acquisition_time < self.config.response_time_target
        
        self.logger.debug(f"Acquired {len(readings)} sensor readings in {acquisition_time*1000:.2f}ms")
        
        if not meets_timing:
            self.logger.warning(f"Acquisition time {acquisition_time*1000:.2f}ms exceeds target {self.config.response_time_target*1000:.2f}ms")
        
        return readings
    
    def real_time_stability_monitoring(self,
                                     readings: List[SensorReading],
                                     stability_threshold: float = 1e-12) -> Dict[str, any]:
        """
        Real-time stability monitoring with <1ms response time.
        
        Args:
            readings: Current sensor readings
            stability_threshold: Threshold for stability violations (N)
            
        Returns:
            Dictionary with stability analysis
        """
        start_time = time.time()
        
        if not readings:
            return {'status': 'no_data', 'processing_time_ms': 0.0}
        
        # Extract force measurements
        forces = np.array([r.casimir_force for r in readings])
        temperatures = np.array([r.temperature for r in readings])
        pressures = np.array([r.pressure for r in readings])
        
        # Statistical analysis
        force_mean = np.mean(forces)
        force_std = np.std(forces)
        force_range = np.max(forces) - np.min(forces)
        
        # Temperature stability
        temp_deviation = np.max(np.abs(temperatures - self.config.temperature_target))
        temp_meets_spec = temp_deviation <= 0.01  # ±0.01 K requirement
        
        # Pressure stability  
        pressure_deviation = np.max(np.abs(pressures - self.config.pressure_target))
        pressure_meets_spec = pressure_deviation <= 1e-6  # ≤10⁻⁶ Pa requirement
        
        # Stability violation detection
        stability_violations = np.abs(forces - force_mean) > stability_threshold
        n_violations = np.sum(stability_violations)
        violation_fraction = n_violations / len(forces)
        
        # Spatial correlation analysis
        positions = np.array([r.position for r in readings])
        spatial_gradient = self._compute_spatial_gradient(forces, positions)
        
        # Quality factor statistics
        quality_factors = np.array([r.quality_factor for r in readings])
        min_quality = np.min(quality_factors)
        
        # Overall stability score
        stability_score = 1.0 - violation_fraction
        if not temp_meets_spec:
            stability_score *= 0.8
        if not pressure_meets_spec:
            stability_score *= 0.8
        if min_quality < 100:
            stability_score *= 0.9
        
        # Processing time check
        processing_time = time.time() - start_time
        meets_realtime = processing_time < self.config.response_time_target
        
        results = {
            'stability_score': stability_score,
            'force_statistics': {
                'mean': force_mean,
                'std': force_std,
                'range': force_range
            },
            'environmental_status': {
                'temperature_deviation_K': temp_deviation,
                'pressure_deviation_Pa': pressure_deviation,
                'temperature_meets_spec': temp_meets_spec,
                'pressure_meets_spec': pressure_meets_spec
            },
            'stability_violations': {
                'count': n_violations,
                'fraction': violation_fraction,
                'threshold': stability_threshold
            },
            'spatial_analysis': {
                'gradient_magnitude': np.linalg.norm(spatial_gradient),
                'max_gradient_component': np.max(np.abs(spatial_gradient))
            },
            'quality_metrics': {
                'min_quality_factor': min_quality,
                'mean_quality_factor': np.mean(quality_factors)
            },
            'performance': {
                'processing_time_ms': processing_time * 1000,
                'meets_realtime_requirement': meets_realtime,
                'n_sensors': len(readings)
            }
        }
        
        self.stability_metrics[time.time()] = results
        
        if not meets_realtime:
            self.logger.warning(f"Stability monitoring exceeded 1ms: {processing_time*1000:.2f}ms")
        
        return results
    
    def _compute_spatial_gradient(self,
                                forces: np.ndarray,
                                positions: np.ndarray) -> np.ndarray:
        """Compute spatial gradient of force field."""
        if len(forces) < 4:
            return np.zeros(3)
        
        # Simple finite difference approximation
        gradient = np.zeros(3)
        
        for i in range(3):  # x, y, z components
            if positions.shape[1] > i:
                # Sort by position component
                sort_indices = np.argsort(positions[:, i])
                sorted_forces = forces[sort_indices]
                sorted_positions = positions[sort_indices, i]
                
                # Compute gradient
                if len(sorted_positions) > 1:
                    gradient[i] = np.gradient(sorted_forces, sorted_positions)[len(sorted_forces)//2]
        
        return gradient
    
    def adaptive_filter_design(self,
                             signal_data: np.ndarray,
                             sampling_rate: float = 1000.0) -> np.ndarray:
        """
        Design adaptive filter for noise reduction.
        
        Args:
            signal_data: Input signal data
            sampling_rate: Sampling rate (Hz)
            
        Returns:
            Filtered signal data
        """
        if len(signal_data) < 10:
            return signal_data
        
        # Design Butterworth filter
        nyquist = sampling_rate / 2
        normalized_cutoff = self.cutoff_frequency / nyquist
        
        # Prevent cutoff frequency from being too high
        normalized_cutoff = min(normalized_cutoff, 0.95)
        
        try:
            b, a = butter(self.filter_order, normalized_cutoff, btype='low')
            filtered_data = filtfilt(b, a, signal_data)
        except Exception as e:
            self.logger.warning(f"Filter design failed: {e}")
            filtered_data = signal_data
        
        return filtered_data
    
    def _sinc(self, x: float) -> float:
        """Normalized sinc function: sinc(x) = sin(x)/x for x≠0, 1 for x=0"""
        if abs(x) < 1e-10:
            return 1.0
        return np.sin(x) / x
    
    def calibration_sequence(self) -> Dict[str, any]:
        """
        Perform sensor array calibration sequence.
        
        Returns:
            Calibration results and corrections
        """
        self.logger.info("Starting sensor calibration sequence...")
        
        # Acquire baseline readings (no external fields)
        baseline_readings = self.acquire_sensor_data(noise_level=1e-16)  # Low noise for calibration
        
        # Compute calibration corrections
        baseline_forces = [r.casimir_force for r in baseline_readings]
        theoretical_force = self.theoretical_casimir_force(
            self.config.plate_separation,
            self.config.sensor_area,
            self.config.temperature_target
        )
        
        # Sensor-by-sensor corrections
        corrections = {}
        for i, (reading, sensor_pos) in enumerate(zip(baseline_readings, self.sensors)):
            correction_factor = theoretical_force / reading.casimir_force if reading.casimir_force != 0 else 1.0
            corrections[i] = {
                'force_correction': correction_factor,
                'position': sensor_pos,
                'baseline_force': reading.casimir_force,
                'theoretical_force': theoretical_force
            }
        
        # Overall calibration metrics
        force_uniformity = np.std(baseline_forces) / np.mean(baseline_forces) if baseline_forces else 0.0
        
        calibration_results = {
            'calibration_timestamp': time.time(),
            'sensor_corrections': corrections,
            'force_uniformity': force_uniformity,
            'baseline_statistics': {
                'mean_force': np.mean(baseline_forces),
                'std_force': np.std(baseline_forces),
                'n_sensors': len(baseline_readings)
            },
            'theoretical_reference': theoretical_force,
            'metamaterial_amplification': self.metamaterial_amp,
            'calibration_quality': 'good' if force_uniformity < 0.1 else 'needs_adjustment'
        }
        
        self.logger.info(f"Calibration complete: uniformity = {force_uniformity:.3f}")
        return calibration_results
    
    def sensor_array_summary(self) -> Dict[str, any]:
        """
        Generate comprehensive summary of sensor array performance.
        
        Returns:
            Dictionary with sensor array metrics
        """
        # Recent performance analysis
        recent_readings = self.sensor_readings[-100:] if len(self.sensor_readings) >= 100 else self.sensor_readings
        
        if not recent_readings:
            return {'status': 'no_data_available'}
        
        # Performance metrics
        response_times = []
        force_values = []
        temperatures = []
        pressures = []
        
        for reading in recent_readings:
            force_values.append(reading.casimir_force)
            temperatures.append(reading.temperature)
            pressures.append(reading.pressure)
        
        # Compute summary statistics
        summary = {
            'array_configuration': {
                'array_size': self.array_size,
                'n_sensors': self.n_sensors,
                'sensor_spacing_um': self.config.sensor_spacing * 1e6,
                'plate_separation_nm': self.config.plate_separation * 1e9
            },
            'performance_specs': {
                'metamaterial_amplification': self.metamaterial_amp,
                'temperature_precision_mK': 10,  # ±0.01 K = ±10 mK
                'pressure_precision_uPa': 1.0,   # ≤10⁻⁶ Pa = ≤1 μPa
                'response_time_target_ms': self.config.response_time_target * 1000
            },
            'recent_performance': {
                'force_statistics': {
                    'mean_pN': np.mean(force_values) * 1e12,  # Convert to pN
                    'std_pN': np.std(force_values) * 1e12,
                    'range_pN': (np.max(force_values) - np.min(force_values)) * 1e12
                },
                'environmental_stability': {
                    'temp_mean_K': np.mean(temperatures),
                    'temp_std_mK': np.std(temperatures) * 1000,
                    'pressure_mean_uPa': np.mean(pressures) * 1e6,
                    'pressure_std_nPa': np.std(pressures) * 1e9
                }
            },
            'data_summary': {
                'total_readings': len(self.sensor_readings),
                'recent_readings': len(recent_readings),
                'polymer_parameter': self.mu
            }
        }
        
        self.logger.info(f"Sensor array: {self.n_sensors} sensors, {len(self.sensor_readings)} total readings")
        return summary
