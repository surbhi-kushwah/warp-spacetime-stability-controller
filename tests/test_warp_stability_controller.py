"""
Comprehensive test suite for the Warp Spacetime Stability Controller.

This module tests all major components of the stability controller including:
- Enhanced gauge coupling matrices
- Polymer-corrected control systems
- Field algebra operations
- Stability analysis
- Causality preservation
- Sensor array operations
"""

import numpy as np
import pytest
import logging
import time
from unittest.mock import Mock, patch

# Import controller components
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from warp_stability_controller import WarpSpacetimeStabilityController, WarpStabilityConfig
from enhanced_gauge_coupling import EnhancedGaugeCouplingMatrix
from polymer_corrected_controller import PolymerCorrectedController, ControllerParameters
from field_algebra import WarpFieldAlgebra
from hybrid_stability_analyzer import HybridStabilityAnalyzer
from causality_preservation import CausalityPreservationFramework
from casimir_sensor_array import CasimirSensorArray, SensorConfig

class TestEnhancedGaugeCoupling:
    """Test enhanced gauge coupling matrix functionality."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.coupling_matrix = EnhancedGaugeCouplingMatrix(
            coupling_strength=1e-3,
            polymer_parameter=0.1,
            stability_threshold=1e6
        )
    
    def test_gauge_coupling_matrix_generation(self):
        """Test enhanced gauge coupling matrix generation."""
        C_enhanced = self.coupling_matrix.enhanced_gauge_coupling_matrix('SU3xSU2xU1')
        
        # Check matrix properties
        assert C_enhanced.shape[0] == C_enhanced.shape[1]  # Square matrix
        assert C_enhanced.shape[0] >= 4  # At least base 4×4
        assert not np.any(np.isnan(C_enhanced))  # No NaN values
        assert not np.any(np.isinf(C_enhanced))  # No infinite values
        
        # Check diagonal elements
        diagonal = np.diag(C_enhanced)
        assert np.all(diagonal > 0)  # Positive diagonal elements
        
    def test_gell_mann_matrices(self):
        """Test Gell-Mann matrix generation."""
        gell_mann = self.coupling_matrix._generate_gell_mann_matrices()
        
        assert gell_mann.shape == (8, 3, 3)  # 8 matrices, each 3×3
        
        # Check tracelessness (except λ₈)
        for i in range(7):
            assert abs(np.trace(gell_mann[i])) < 1e-10
        
        # Check Hermiticity
        for i in range(8):
            assert np.allclose(gell_mann[i], gell_mann[i].conj().T)
    
    def test_pauli_matrices(self):
        """Test Pauli matrix generation."""
        pauli = self.coupling_matrix._generate_pauli_matrices()
        
        assert pauli.shape == (3, 2, 2)  # 3 matrices, each 2×2
        
        # Check properties
        for i in range(3):
            # Traceless
            assert abs(np.trace(pauli[i])) < 1e-10
            # Hermitian
            assert np.allclose(pauli[i], pauli[i].conj().T)
            # Unit determinant (±1)
            det = np.linalg.det(pauli[i])
            assert abs(abs(det) - 1) < 1e-10
    
    def test_stability_eigenvalues(self):
        """Test stability eigenvalue computation."""
        C_matrix = np.array([[1.0, 0.1], [0.1, 1.0]])
        eigenvals, is_stable = self.coupling_matrix.compute_stability_eigenvalues(C_matrix)
        
        assert len(eigenvals) == 2
        assert isinstance(is_stable, bool)
        
    def test_warp_field_coupling_coefficients(self):
        """Test warp field coupling coefficient computation."""
        coefficients = self.coupling_matrix.warp_field_coupling_coefficients()
        
        required_keys = ['theta_gravity', 'epsilon_exotic', 'gamma_spacetime']
        for key in required_keys:
            assert key in coefficients
            assert isinstance(coefficients[key], (int, float))
            assert not np.isnan(coefficients[key])

class TestPolymerCorrectedController:
    """Test polymer-corrected control system."""
    
    def setup_method(self):
        """Setup for each test method."""
        Kp = np.eye(3) * 10.0
        Ki = np.eye(3) * 1.0  
        Kd = np.eye(3) * 0.1
        
        self.params = ControllerParameters(
            Kp=Kp, Ki=Ki, Kd=Kd,
            mu=0.1, dt=1e-4, stability_threshold=1e6
        )
        
        coupling_matrix = np.eye(10)  # Simplified coupling matrix
        
        self.controller = PolymerCorrectedController(
            controller_params=self.params,
            coupling_matrix=coupling_matrix,
            field_dimensions=3
        )
    
    def test_polymer_corrected_gains(self):
        """Test polymer-corrected gain computation."""
        Kp_corr, Ki_corr, Kd_corr = self.controller.compute_polymer_corrected_gains()
        
        # Check shapes
        assert Kp_corr.shape == (3, 3)
        assert Ki_corr.shape == (3, 3)
        assert Kd_corr.shape == (3, 3)
        
        # Check for finite values
        assert np.all(np.isfinite(Kp_corr))
        assert np.all(np.isfinite(Ki_corr))
        assert np.all(np.isfinite(Kd_corr))
    
    def test_control_update(self):
        """Test control update computation."""
        error = np.array([0.1, -0.2, 0.05])
        current_time = time.time()
        
        control_output = self.controller.control_update(error, current_time)
        
        assert len(control_output) == 3
        assert np.all(np.isfinite(control_output))
    
    def test_stability_analysis(self):
        """Test stability analysis."""
        system_matrix = np.array([[-1.0, 0.1], [0.1, -2.0]])
        
        results = self.controller.stability_analysis(system_matrix)
        
        assert 'is_stable' in results
        assert 'stability_margin' in results
        assert 'settling_time_s' in results
        assert isinstance(results['is_stable'], bool)

class TestWarpFieldAlgebra:
    """Test warp field algebra operations."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.field_algebra = WarpFieldAlgebra(
            spacetime_dim=4,
            gauge_group='SU3',
            polymer_parameter=0.1
        )
    
    def test_enhanced_field_commutator(self):
        """Test enhanced field commutator computation."""
        commutator = self.field_algebra.enhanced_field_commutator(0, 1, 0, 0)
        
        assert isinstance(commutator, complex)
        assert np.isfinite(commutator.real)
        assert np.isfinite(commutator.imag)
    
    def test_gauge_field_commutators(self):
        """Test gauge field commutator computation."""
        aa_comm, api_comm = self.field_algebra.gauge_field_commutators(0, 1, 0, 0)
        
        # A-A commutator should be zero
        assert abs(aa_comm) < 1e-10
        
        # A-Pi commutator should be non-zero
        assert isinstance(api_comm, complex)
    
    def test_field_strength_computation(self):
        """Test enhanced field strength computation."""
        field_config = {
            'coupling_constant': 0.1,
            'gauge_fields': np.random.random((4, 8)) * 0.1,
            'field_derivatives': {}
        }
        
        field_strength = self.field_algebra.enhanced_field_strength(0, 1, 0, field_config)
        
        assert isinstance(field_strength, complex)
        assert np.isfinite(field_strength.real)
    
    def test_algebra_consistency(self):
        """Test field algebra consistency."""
        consistency = self.field_algebra.field_algebra_consistency_check()
        
        assert 'overall_consistent' in consistency
        assert 'commutator_antisymmetry' in consistency
        assert isinstance(consistency['overall_consistent'], bool)

class TestHybridStabilityAnalyzer:
    """Test hybrid stability analyzer."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.analyzer = HybridStabilityAnalyzer(
            n_gaussians=3,
            polymer_parameter=0.1,
            stability_threshold=1e6,
            spatial_extent=5.0
        )
    
    def test_multi_gaussian_profile(self):
        """Test multi-Gaussian profile computation."""
        r_grid = np.linspace(-5, 5, 50)
        profile = self.analyzer.multi_gaussian_profile(r_grid)
        
        assert len(profile) == len(r_grid)
        assert np.all(np.isfinite(profile))
        assert np.all(profile >= 0)  # Gaussian profiles are non-negative
    
    def test_stability_hamiltonian(self):
        """Test stability Hamiltonian computation."""
        r_grid = np.linspace(-5, 5, 50)
        
        # Test parameters: [A₁, A₂, A₃, r₁, r₂, r₃, σ₁, σ₂, σ₃]
        params = np.array([1.0, 0.8, 1.2, -1.0, 0.0, 1.0, 1.0, 1.5, 0.8])
        
        hamiltonian = self.analyzer.stability_hamiltonian(params, r_grid)
        
        assert isinstance(hamiltonian, (int, float))
        assert np.isfinite(hamiltonian)
    
    def test_stability_condition_check(self):
        """Test stability condition checking."""
        eigenvalues = np.array([-1000, -2000, -500])  # Stable eigenvalues
        
        results = self.analyzer.stability_condition_check(eigenvalues)
        
        assert 'is_stable' in results
        assert 'response_time_s' in results
        assert 'meets_1ms_requirement' in results
        assert results['is_stable']  # Should be stable

class TestCausalityPreservation:
    """Test causality preservation framework."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.causality = CausalityPreservationFramework(
            spacetime_dim=4,
            polymer_parameter=0.1,
            causality_threshold=1e-6,
            emergency_response_time=1e-3
        )
    
    def test_israel_darmois_conditions(self):
        """Test enhanced Israel-Darmois junction conditions."""
        K_plus = np.random.random((3, 3)) * 0.1
        K_minus = np.random.random((3, 3)) * 0.1
        extrinsic_curvature = {'plus': K_plus, 'minus': K_minus}
        
        metric_induced = np.eye(4)
        matter_content = {'warp_field': np.random.random(4) * 0.1}
        
        S_enhanced = self.causality.enhanced_israel_darmois_conditions(
            extrinsic_curvature, metric_induced, matter_content
        )
        
        assert S_enhanced.shape == (3, 3)
        assert np.all(np.isfinite(S_enhanced))
    
    def test_ctc_detection(self):
        """Test closed timelike curve detection."""
        metric_tensor = np.diag([-1, 1, 1, 1])  # Minkowski metric
        coordinates = (0.0, 0.0, 0.0, 0.0)
        velocity_field = np.array([1.0, 0.1, 0.0, 0.0])
        
        results = self.causality.closed_timelike_curve_detection(
            metric_tensor, coordinates, velocity_field
        )
        
        assert 'ctc_detected' in results
        assert 'metamaterial_amplification' in results
        assert isinstance(results['ctc_detected'], bool)
    
    def test_energy_momentum_conservation(self):
        """Test energy-momentum conservation checking."""
        shape = (4, 4)
        matter_tensor = np.random.random(shape) * 0.1
        warp_tensor = np.random.random(shape) * 0.1
        gauge_tensor = np.random.random(shape) * 0.1
        stability_tensor = np.random.random(shape) * 0.1
        
        results = self.causality.energy_momentum_conservation(
            matter_tensor, warp_tensor, gauge_tensor, stability_tensor
        )
        
        assert 'is_conserved' in results
        assert 'total_tensor' in results
        assert results['total_tensor'].shape == shape

class TestCasimirSensorArray:
    """Test Casimir sensor array functionality."""
    
    def setup_method(self):
        """Setup for each test method."""
        sensor_config = SensorConfig(
            sensor_spacing=1e-6,
            plate_separation=1e-9,
            sensor_area=1e-6,
            temperature_target=0.01,
            pressure_target=1e-6,
            response_time_target=1e-3
        )
        
        self.sensor_array = CasimirSensorArray(
            config=sensor_config,
            array_size=(3, 3, 1),
            metamaterial_amplification=847.0,
            polymer_parameter=0.1
        )
    
    def test_theoretical_casimir_force(self):
        """Test theoretical Casimir force computation."""
        force = self.sensor_array.theoretical_casimir_force(
            plate_separation=1e-9,
            plate_area=1e-6,
            temperature=0.01
        )
        
        assert isinstance(force, (int, float))
        assert force < 0  # Casimir force is attractive
        assert np.isfinite(force)
    
    def test_sensor_data_acquisition(self):
        """Test sensor data acquisition."""
        readings = self.sensor_array.acquire_sensor_data(noise_level=1e-15)
        
        assert len(readings) == 9  # 3×3×1 array
        
        for reading in readings:
            assert hasattr(reading, 'sensor_id')
            assert hasattr(reading, 'casimir_force')
            assert hasattr(reading, 'temperature')
            assert hasattr(reading, 'pressure')
            assert np.isfinite(reading.casimir_force)
    
    def test_stability_monitoring(self):
        """Test real-time stability monitoring."""
        readings = self.sensor_array.acquire_sensor_data()
        results = self.sensor_array.real_time_stability_monitoring(readings)
        
        assert 'stability_score' in results
        assert 'force_statistics' in results
        assert 'environmental_status' in results
        assert 'performance' in results
        
        assert 0 <= results['stability_score'] <= 1

class TestWarpStabilityController:
    """Test integrated warp stability controller."""
    
    def setup_method(self):
        """Setup for each test method."""
        config = WarpStabilityConfig(
            polymer_parameter=0.1,
            stability_threshold=1e6,
            emergency_response_time=1e-3,
            field_dimensions=3,
            sensor_array_size=(2, 2, 1)
        )
        
        self.controller = WarpSpacetimeStabilityController(config)
    
    def test_controller_initialization(self):
        """Test controller initialization."""
        assert self.controller.config is not None
        assert self.controller.coupling_matrix is not None
        assert self.controller.controller is not None
        assert self.controller.field_algebra is not None
        assert self.controller.stability_analyzer is not None
        assert self.controller.causality_framework is not None
        assert self.controller.sensor_array is not None
    
    def test_field_state_update(self):
        """Test field state update."""
        field_values = np.array([0.1, -0.2, 0.05])
        field_derivatives = np.array([0.01, 0.02, -0.01])
        metric_perturbation = np.random.random((4, 4)) * 0.01
        
        state = self.controller.update_field_state(
            field_values, field_derivatives, metric_perturbation
        )
        
        assert state is not None
        assert hasattr(state, 'field_values')
        assert hasattr(state, 'energy_density')
        assert hasattr(state, 'stability_score')
        assert np.array_equal(state.field_values, field_values)
    
    def test_real_time_control(self):
        """Test real-time stability control."""
        target_state = np.array([0.0, 0.0, 0.0])
        current_measurements = {
            'field_values': np.array([0.1, -0.1, 0.05]),
            'field_derivatives': np.array([0.01, 0.01, -0.005]),
            'metric_perturbation': np.zeros((4, 4))
        }
        
        result = self.controller.real_time_stability_control(
            target_state, current_measurements
        )
        
        assert 'control_output' in result
        assert 'field_state' in result
        assert 'processing_time_ms' in result
        assert 'stability_analysis' in result
        
        # Check processing time requirement
        assert result['processing_time_ms'] < 10.0  # Should be much less than 10ms
    
    def test_calibration_sequence(self):
        """Test system calibration."""
        calibration_results = self.controller.calibrate_system()
        
        assert 'sensor_array' in calibration_results
        assert 'field_algebra' in calibration_results
        assert 'coupling_matrix' in calibration_results
        assert 'overall_status' in calibration_results
        
        assert calibration_results['overall_status'] in ['success', 'needs_adjustment']
    
    def test_system_report_generation(self):
        """Test system report generation."""
        # Generate some control history first
        for i in range(5):
            target_state = np.zeros(3)
            measurements = {
                'field_values': np.random.random(3) * 0.1,
                'field_derivatives': np.random.random(3) * 0.01,
                'metric_perturbation': np.zeros((4, 4))
            }
            self.controller.real_time_stability_control(target_state, measurements)
        
        report = self.controller.generate_system_report()
        
        assert 'system_configuration' in report
        assert 'performance_summary' in report
        assert 'component_status' in report
        assert 'system_health' in report

class TestPerformanceRequirements:
    """Test that performance requirements are met."""
    
    def setup_method(self):
        """Setup for performance tests."""
        config = WarpStabilityConfig(
            polymer_parameter=0.1,
            stability_threshold=1e6,
            emergency_response_time=1e-3,
            field_dimensions=4
        )
        
        self.controller = WarpSpacetimeStabilityController(config)
    
    def test_sub_millisecond_response(self):
        """Test sub-millisecond response requirement."""
        target_state = np.zeros(4)
        measurements = {
            'field_values': np.random.random(4) * 0.1,
            'field_derivatives': np.random.random(4) * 0.01,
            'metric_perturbation': np.zeros((4, 4))
        }
        
        start_time = time.time()
        result = self.controller.real_time_stability_control(target_state, measurements)
        processing_time = time.time() - start_time
        
        # Sub-millisecond requirement
        assert processing_time < 1e-3, f"Processing time {processing_time*1000:.2f}ms exceeds 1ms requirement"
        
        # Verify reported processing time is consistent
        assert abs(result['processing_time_ms'] - processing_time * 1000) < 10  # Within 10ms tolerance
    
    def test_stability_threshold_compliance(self):
        """Test compliance with stability threshold requirements."""
        # Test with stable eigenvalues
        stable_eigenvals = np.array([-2000, -1500, -3000, -1000])  # All > 1000 s⁻¹ magnitude
        
        stability_results = self.controller.stability_analyzer.stability_condition_check(stable_eigenvals)
        
        assert stability_results['is_stable']
        assert stability_results['meets_1ms_requirement']
        assert stability_results['actual_response_rate'] >= 1000  # Above threshold
    
    def test_sensor_precision_requirements(self):
        """Test sensor precision requirements."""
        sensor_config = self.controller.sensor_array.config
        
        # Temperature precision: ±0.01 K
        assert sensor_config.temperature_target == 0.01
        
        # Pressure precision: ≤10⁻⁶ Pa
        assert sensor_config.pressure_target <= 1e-6
        
        # Response time: <1ms
        assert sensor_config.response_time_target < 1e-3
    
    def test_metamaterial_amplification(self):
        """Test metamaterial amplification factor."""
        amplification = self.controller.coupling_matrix.metamaterial_amplification_factor()
        
        # Should achieve 847× amplification
        assert amplification >= 800  # Allow some tolerance
        assert amplification <= 900

if __name__ == '__main__':
    # Configure logging for tests
    logging.basicConfig(level=logging.INFO)
    
    # Run tests
    pytest.main([__file__, '-v'])
