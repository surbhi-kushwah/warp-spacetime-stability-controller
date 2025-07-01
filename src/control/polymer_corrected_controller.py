"""
Real-Time Polymer-Corrected Control System

This module implements enhanced PID control with polymer corrections and multi-domain
coupling compensation for warp field stability control, building on the control
architecture from polymerized-lqg-matter-transporter.
"""

import numpy as np
from typing import Tuple, Dict, List, Optional, Callable
import logging
from dataclasses import dataclass
from scipy.integrate import solve_ivp
import time

@dataclass
class ControllerParameters:
    """Parameters for polymer-corrected PID controller."""
    Kp: np.ndarray  # Proportional gains
    Ki: np.ndarray  # Integral gains  
    Kd: np.ndarray  # Derivative gains
    mu: float       # Polymer parameter
    dt: float       # Control timestep (target <1ms)
    stability_threshold: float  # Minimum response rate

class PolymerCorrectedController:
    """
    Enhanced PID control with polymer modifications for warp field stability.
    
    Mathematical Framework:
    u(t) = Kp·e(t) + Ki·∫e(τ)sinc(πμ)dτ + Kd·de/dt
    
    Cross-coupling compensation:
    Kp^polymer = Kp · [polymer_coupling_matrix]
    
    Stability condition:
    Re(λᵢ) < -γ·sinc(πμ_stability) ∀i
    """
    
    def __init__(self, 
                 controller_params: ControllerParameters,
                 coupling_matrix: np.ndarray,
                 field_dimensions: int = 4):
        """
        Initialize polymer-corrected controller.
        
        Args:
            controller_params: PID parameters with polymer corrections
            coupling_matrix: Multi-physics coupling matrix from enhanced_gauge_coupling
            field_dimensions: Number of controlled field variables
        """
        self.params = controller_params
        self.coupling_matrix = coupling_matrix
        self.n_dims = field_dimensions
        
        # State variables for integral and derivative terms
        self.integral_error = np.zeros(self.n_dims)
        self.previous_error = np.zeros(self.n_dims)
        self.previous_time = None
        
        # Performance monitoring
        self.response_times = []
        self.stability_violations = 0
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initialized polymer controller for {self.n_dims}D field")
        
    def compute_polymer_corrected_gains(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute polymer-corrected control gains with cross-coupling compensation.
        
        Mathematical formulation:
        Kp^polymer = Kp · [1 + α_ij·sinc(πμ_ij)]
        Ki^polymer = Ki · [1 + β_ij·sinc(πμ_ij)]  
        Kd^polymer = Kd · [1 + γ_ij·sinc(πμ_ij)]
        
        Returns:
            Tuple of (Kp_corrected, Ki_corrected, Kd_corrected)
        """
        # Cross-coupling coefficients
        alpha_12, alpha_13 = 0.1, 0.05
        alpha_21, alpha_23 = 0.08, 0.12
        alpha_31, alpha_32 = 0.06, 0.09
        
        # Polymer modification parameters for each coupling
        mu_12, mu_13 = self.params.mu * 1.1, self.params.mu * 0.9
        mu_21, mu_23 = self.params.mu * 1.2, self.params.mu * 0.8
        mu_31, mu_32 = self.params.mu * 1.0, self.params.mu * 1.1
        
        # Cross-coupling matrix with polymer corrections
        if self.n_dims >= 3:
            coupling_correction = np.array([
                [1.0, alpha_12 * self._sinc(np.pi * mu_12), alpha_13 * self._sinc(np.pi * mu_13)],
                [alpha_21 * self._sinc(np.pi * mu_21), 1.0, alpha_23 * self._sinc(np.pi * mu_23)],
                [alpha_31 * self._sinc(np.pi * mu_31), alpha_32 * self._sinc(np.pi * mu_32), 1.0]
            ])
        else:
            # For smaller systems, use identity with diagonal polymer correction
            coupling_correction = np.eye(self.n_dims) * self._sinc(np.pi * self.params.mu)
        
        # Extend coupling correction to match field dimensions
        if coupling_correction.shape[0] < self.n_dims:
            extended_correction = np.eye(self.n_dims)
            extended_correction[:coupling_correction.shape[0], :coupling_correction.shape[1]] = coupling_correction
            coupling_correction = extended_correction
        elif coupling_correction.shape[0] > self.n_dims:
            coupling_correction = coupling_correction[:self.n_dims, :self.n_dims]
        
        # Apply polymer corrections to gains
        Kp_corrected = self.params.Kp * coupling_correction if self.params.Kp.ndim == 2 else np.diag(self.params.Kp) @ coupling_correction
        Ki_corrected = self.params.Ki * coupling_correction if self.params.Ki.ndim == 2 else np.diag(self.params.Ki) @ coupling_correction  
        Kd_corrected = self.params.Kd * coupling_correction if self.params.Kd.ndim == 2 else np.diag(self.params.Kd) @ coupling_correction
        
        self.logger.debug(f"Applied polymer corrections with μ={self.params.mu}")
        return Kp_corrected, Ki_corrected, Kd_corrected
        
    def _sinc(self, x: float) -> float:
        """Normalized sinc function: sinc(x) = sin(x)/x for x≠0, 1 for x=0"""
        if abs(x) < 1e-10:
            return 1.0
        return np.sin(x) / x
    
    def control_update(self, 
                      error: np.ndarray, 
                      current_time: float,
                      field_state: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute control output with polymer corrections and coupling compensation.
        
        Enhanced PID control equation:
        u(t) = Kp·e(t) + Ki·∫e(τ)sinc(πμ)dτ + Kd·de/dt + K_coupling·C_warp·x(t)
        
        Args:
            error: Current error vector
            current_time: Current time (s)
            field_state: Current field state for coupling compensation
            
        Returns:
            Control output vector
        """
        start_time = time.time()
        
        # Ensure error is proper shape
        if error.shape[0] != self.n_dims:
            error = error[:self.n_dims] if len(error) > self.n_dims else np.pad(error, (0, self.n_dims - len(error)))
        
        # Time delta calculation
        if self.previous_time is None:
            dt = self.params.dt
        else:
            dt = current_time - self.previous_time
            
        # Polymer-corrected gains
        Kp_corrected, Ki_corrected, Kd_corrected = self.compute_polymer_corrected_gains()
        
        # Proportional term
        proportional = Kp_corrected @ error
        
        # Integral term with polymer modification
        self.integral_error += error * dt * self._sinc(np.pi * self.params.mu)
        integral = Ki_corrected @ self.integral_error
        
        # Derivative term
        if self.previous_time is not None:
            derivative_error = (error - self.previous_error) / dt
        else:
            derivative_error = np.zeros_like(error)
        derivative = Kd_corrected @ derivative_error
        
        # Coupling compensation term
        coupling_compensation = np.zeros(self.n_dims)
        if field_state is not None and self.coupling_matrix is not None:
            # Extract relevant coupling submatrix
            n_coupling = min(self.coupling_matrix.shape[0], self.n_dims)
            coupling_sub = self.coupling_matrix[:n_coupling, :n_coupling]
            state_sub = field_state[:n_coupling] if len(field_state) >= n_coupling else np.pad(field_state, (0, n_coupling - len(field_state)))
            
            K_coupling = 0.01  # Coupling gain factor
            coupling_compensation[:n_coupling] = K_coupling * coupling_sub @ state_sub
        
        # Total control output
        control_output = proportional + integral + derivative + coupling_compensation
        
        # Update state
        self.previous_error = error.copy()
        self.previous_time = current_time
        
        # Performance monitoring
        response_time = time.time() - start_time
        self.response_times.append(response_time)
        
        # Check sub-millisecond requirement
        if response_time > 1e-3:
            self.stability_violations += 1
            self.logger.warning(f"Control update exceeded 1ms: {response_time*1000:.2f}ms")
        
        self.logger.debug(f"Control update: {response_time*1000:.3f}ms")
        return control_output
    
    def stability_analysis(self, system_matrix: np.ndarray) -> Dict[str, any]:
        """
        Analyze closed-loop stability with polymer corrections.
        
        Stability condition: Re(λᵢ) < -γ·sinc(πμ_stability) ∀i
        
        Args:
            system_matrix: Closed-loop system matrix A_cl
            
        Returns:
            Dictionary with stability analysis results
        """
        # Compute eigenvalues
        eigenvals = np.linalg.eigvals(system_matrix)
        
        # Stability threshold with polymer correction
        gamma_stability = self.params.stability_threshold
        stability_threshold = -gamma_stability * self._sinc(np.pi * self.params.mu)
        
        # Stability check
        real_parts = np.real(eigenvals)
        is_stable = np.all(real_parts < stability_threshold)
        stability_margin = stability_threshold - np.max(real_parts)
        
        # Response time estimation (dominant pole)
        dominant_pole = np.max(real_parts)
        settling_time = 4.0 / abs(dominant_pole) if dominant_pole < 0 else np.inf
        
        results = {
            'is_stable': is_stable,
            'eigenvalues': eigenvals,
            'stability_margin': stability_margin,
            'settling_time_s': settling_time,
            'meets_1ms_requirement': settling_time < 1e-3,
            'stability_threshold': stability_threshold,
            'polymer_correction_factor': self._sinc(np.pi * self.params.mu)
        }
        
        self.logger.info(f"Stability analysis: stable={is_stable}, margin={stability_margin:.2e}")
        return results
    
    def adaptive_gain_tuning(self, 
                           performance_metrics: Dict[str, float],
                           target_response_time: float = 1e-3) -> None:
        """
        Adaptive tuning of controller gains based on performance metrics.
        
        Args:
            performance_metrics: Dictionary with overshoot, settling_time, etc.
            target_response_time: Target response time (default 1ms)
        """
        overshoot = performance_metrics.get('overshoot', 0.0)
        settling_time = performance_metrics.get('settling_time', 0.0)
        steady_state_error = performance_metrics.get('steady_state_error', 0.0)
        
        # Gain adjustment factors
        kp_factor = 1.0
        ki_factor = 1.0
        kd_factor = 1.0
        
        # Reduce overshoot
        if overshoot > 0.1:  # 10% overshoot threshold
            kp_factor *= 0.9
            kd_factor *= 1.1
            
        # Improve settling time
        if settling_time > target_response_time:
            kp_factor *= 1.1
            ki_factor *= 1.05
            
        # Reduce steady-state error
        if steady_state_error > 0.01:  # 1% error threshold
            ki_factor *= 1.1
            
        # Apply polymer-corrected adjustments
        polymer_factor = self._sinc(np.pi * self.params.mu)
        
        # Update gains
        if self.params.Kp.ndim == 1:
            self.params.Kp *= kp_factor * polymer_factor
            self.params.Ki *= ki_factor * polymer_factor
            self.params.Kd *= kd_factor * polymer_factor
        else:
            self.params.Kp *= kp_factor * polymer_factor
            self.params.Ki *= ki_factor * polymer_factor  
            self.params.Kd *= kd_factor * polymer_factor
            
        self.logger.info(f"Adaptive tuning: Kp×{kp_factor:.3f}, Ki×{ki_factor:.3f}, Kd×{kd_factor:.3f}")
    
    def performance_summary(self) -> Dict[str, any]:
        """
        Generate performance summary for the controller.
        
        Returns:
            Dictionary with performance metrics
        """
        if not self.response_times:
            return {'status': 'No control updates recorded'}
            
        response_times_ms = np.array(self.response_times) * 1000
        
        summary = {
            'mean_response_time_ms': np.mean(response_times_ms),
            'max_response_time_ms': np.max(response_times_ms),
            'min_response_time_ms': np.min(response_times_ms),
            'std_response_time_ms': np.std(response_times_ms),
            'sub_1ms_percentage': np.sum(response_times_ms < 1.0) / len(response_times_ms) * 100,
            'stability_violations': self.stability_violations,
            'total_updates': len(self.response_times),
            'polymer_parameter': self.params.mu,
            'target_dt_ms': self.params.dt * 1000
        }
        
        self.logger.info(f"Performance: {summary['sub_1ms_percentage']:.1f}% sub-1ms updates")
        return summary
