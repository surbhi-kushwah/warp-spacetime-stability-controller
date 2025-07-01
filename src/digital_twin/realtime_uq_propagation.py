"""
Advanced Real-Time UQ Propagation with Enhanced Correlation
Implements polynomial chaos expansion, Sobol' sensitivity, and enhanced correlation matrices

This module provides the mathematical framework for real-time uncertainty quantification
propagation with 5×5 correlation matrices, polynomial chaos expansion, and Sobol' sensitivity analysis.
"""

import numpy as np
import logging
from typing import Dict, Tuple, Optional, List, Callable, Union
from dataclasses import dataclass
from scipy.stats import norm, uniform
from scipy.linalg import cholesky, LinAlgError
from scipy.special import eval_hermite, factorial
import itertools

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class UQPropagationConfig:
    """Configuration for real-time UQ propagation"""
    n_parameters: int = 5
    max_polynomial_degree: int = 4
    n_monte_carlo_samples: int = 50000
    correlation_structure: str = 'enhanced'  # 'enhanced', 'exponential', 'custom'
    diffusion_coefficient: float = 1e-6
    source_term_strength: float = 1e-3
    temporal_correlation_length: float = 1e-3
    convergence_tolerance: float = 1e-6

class AdvancedRealTimeUQPropagation:
    """
    Advanced Real-Time UQ Propagation with Enhanced Correlation
    
    Implements the mathematical framework:
    ∂²P(x,t)/∂t² = ∇·(D(x,ξ)∇P) + S(x,t,ξ) + Correlation(5×5)
    
    Features:
    - Polynomial Chaos Expansion: α₈₄₇(ξ) = Σᵢ αᵢ × Φᵢ(ξ)
    - Sobol' sensitivity analysis: Sᵢ = Var[E[Y|Xᵢ]]/Var[Y]
    - Enhanced 5×5 correlation matrix with Cholesky decomposition
    - Real-time probability density evolution
    """
    
    def __init__(self, config: Optional[UQPropagationConfig] = None):
        self.config = config or UQPropagationConfig()
        
        # UQ parameters
        self.n_params = self.config.n_parameters
        self.max_degree = self.config.max_polynomial_degree
        self.n_mc_samples = self.config.n_monte_carlo_samples
        
        # Initialize correlation matrix
        self.correlation_matrix = self._construct_enhanced_correlation_matrix()
        self.cholesky_factor = self._compute_cholesky_decomposition()
        
        # Polynomial chaos basis
        self.pce_basis = self._construct_polynomial_chaos_basis()
        self.pce_coefficients = {}
        
        # UQ propagation state
        self.probability_field = None
        self.parameter_samples = None
        self.sensitivity_indices = {}
        
        # Evolution history
        self.propagation_history = []
        self.sobol_history = []
        
        logger.info(f"Initialized real-time UQ propagation: {self.n_params} parameters, degree {self.max_degree}")
        logger.info(f"Correlation matrix condition number: {np.linalg.cond(self.correlation_matrix):.2f}")
    
    def _construct_enhanced_correlation_matrix(self) -> np.ndarray:
        """
        Construct enhanced 5×5 correlation matrix
        Σ_params with structured correlation for metamaterial parameters
        """
        if self.config.correlation_structure == 'enhanced':
            # Enhanced correlation structure based on physical coupling
            correlation_matrix = np.array([
                [1.00, 0.35, 0.25, 0.15, 0.10],  # ε' correlations
                [0.35, 1.00, 0.40, 0.20, 0.15],  # μ' correlations
                [0.25, 0.40, 1.00, 0.30, 0.25],  # Geometric correlations
                [0.15, 0.20, 0.30, 1.00, 0.35],  # Thermal correlations
                [0.10, 0.15, 0.25, 0.35, 1.00]   # Control correlations
            ])
        
        elif self.config.correlation_structure == 'exponential':
            # Exponential decay correlation
            correlation_matrix = np.zeros((self.n_params, self.n_params))
            decay_length = 1.5
            
            for i in range(self.n_params):
                for j in range(self.n_params):
                    correlation_matrix[i, j] = np.exp(-abs(i - j) / decay_length)
        
        else:
            # Default identity matrix
            correlation_matrix = np.eye(self.n_params)
        
        # Ensure positive definiteness
        correlation_matrix = self._ensure_positive_definite(correlation_matrix)
        
        return correlation_matrix
    
    def _ensure_positive_definite(self, matrix: np.ndarray) -> np.ndarray:
        """Ensure correlation matrix is positive definite"""
        eigenvals, eigenvecs = np.linalg.eigh(matrix)
        
        # Regularize negative eigenvalues
        min_eigenval = 1e-6
        eigenvals_regularized = np.maximum(eigenvals, min_eigenval)
        
        # Reconstruct matrix
        matrix_pd = eigenvecs @ np.diag(eigenvals_regularized) @ eigenvecs.T
        
        # Normalize diagonal to 1
        diag_sqrt = np.sqrt(np.diag(matrix_pd))
        matrix_normalized = matrix_pd / np.outer(diag_sqrt, diag_sqrt)
        
        return matrix_normalized
    
    def _compute_cholesky_decomposition(self) -> np.ndarray:
        """Compute Cholesky decomposition for correlated sampling"""
        try:
            chol_factor = cholesky(self.correlation_matrix, lower=True)
            return chol_factor
        except LinAlgError:
            logger.warning("Cholesky decomposition failed, using regularized matrix")
            # Add regularization
            regularized_matrix = self.correlation_matrix + 1e-6 * np.eye(self.n_params)
            return cholesky(regularized_matrix, lower=True)
    
    def _construct_polynomial_chaos_basis(self) -> List[Tuple]:
        """Construct polynomial chaos basis functions"""
        # Generate multi-indices for polynomial terms
        basis_indices = []
        
        # All combinations of polynomial degrees up to max_degree
        for total_degree in range(self.max_degree + 1):
            for multi_index in itertools.combinations_with_replacement(range(self.n_params), total_degree):
                # Convert to degree vector
                degree_vector = [0] * self.n_params
                for param_index in multi_index:
                    degree_vector[param_index] += 1
                
                basis_indices.append(tuple(degree_vector))
        
        # Remove duplicates and sort
        basis_indices = list(set(basis_indices))
        basis_indices.sort()
        
        logger.debug(f"Constructed {len(basis_indices)} polynomial chaos basis functions")
        return basis_indices
    
    def generate_correlated_samples(self, n_samples: Optional[int] = None) -> np.ndarray:
        """
        Generate correlated parameter samples using Cholesky decomposition
        
        Returns:
            Array of shape (n_samples, n_parameters) with correlated samples
        """
        if n_samples is None:
            n_samples = self.n_mc_samples
        
        # Generate independent standard normal samples
        independent_samples = np.random.normal(0, 1, (n_samples, self.n_params))
        
        # Apply Cholesky transformation for correlation
        correlated_samples = independent_samples @ self.cholesky_factor.T
        
        # Store for later use
        self.parameter_samples = correlated_samples
        
        logger.debug(f"Generated {n_samples} correlated parameter samples")
        return correlated_samples
    
    def compute_polynomial_chaos_expansion(self, response_function: Callable,
                                         parameter_samples: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        Compute Polynomial Chaos Expansion coefficients
        α₈₄₇(ξ) = Σᵢ αᵢ × Φᵢ(ξ)
        
        Args:
            response_function: Function Y = f(ξ) to expand
            parameter_samples: Parameter samples (if None, generates new ones)
            
        Returns:
            Dictionary containing PCE coefficients and analysis
        """
        if parameter_samples is None:
            parameter_samples = self.generate_correlated_samples()
        
        n_samples = parameter_samples.shape[0]
        n_basis = len(self.pce_basis)
        
        # Evaluate response function at sample points
        response_values = np.array([response_function(sample) for sample in parameter_samples])
        
        # Construct polynomial basis matrix
        basis_matrix = np.zeros((n_samples, n_basis))
        
        for sample_idx, sample in enumerate(parameter_samples):
            for basis_idx, multi_index in enumerate(self.pce_basis):
                # Evaluate multivariate Hermite polynomial
                basis_value = 1.0
                for param_idx, degree in enumerate(multi_index):
                    if degree > 0:
                        hermite_val = eval_hermite(degree, sample[param_idx])
                        normalization = np.sqrt(factorial(degree))
                        basis_value *= hermite_val / normalization
                
                basis_matrix[sample_idx, basis_idx] = basis_value
        
        # Solve for PCE coefficients using least squares
        pce_coefficients, residuals, rank, singular_values = np.linalg.lstsq(
            basis_matrix, response_values, rcond=None
        )
        
        # Compute PCE statistics
        pce_mean = pce_coefficients[0]  # Zeroth-order coefficient
        pce_variance = np.sum(pce_coefficients[1:]**2)  # Sum of squared higher-order coefficients
        pce_std = np.sqrt(pce_variance)
        
        # R² coefficient of determination
        response_mean = np.mean(response_values)
        ss_total = np.sum((response_values - response_mean)**2)
        ss_residual = residuals[0] if len(residuals) > 0 else 0
        r_squared = 1 - ss_residual / ss_total if ss_total > 0 else 0
        
        pce_result = {
            'coefficients': pce_coefficients,
            'basis_functions': self.pce_basis,
            'mean': pce_mean,
            'variance': pce_variance,
            'std_deviation': pce_std,
            'r_squared': r_squared,
            'rank': rank,
            'condition_number': singular_values[0] / singular_values[-1] if len(singular_values) > 0 else 1,
            'response_samples': response_values,
            'parameter_samples': parameter_samples
        }
        
        # Store coefficients
        self.pce_coefficients = pce_result
        
        logger.info(f"PCE computed: R² = {r_squared:.4f}, std = {pce_std:.6f}")
        return pce_result
    
    def compute_sobol_sensitivity_indices(self, pce_result: Optional[Dict] = None) -> Dict[str, np.ndarray]:
        """
        Compute Sobol' sensitivity indices
        Sᵢ = Var[E[Y|Xᵢ]]/Var[Y] and Sᵢⱼ for second-order interactions
        
        Args:
            pce_result: PCE expansion result (uses stored if None)
            
        Returns:
            Dictionary containing first and second-order Sobol' indices
        """
        if pce_result is None:
            pce_result = self.pce_coefficients
        
        if not pce_result:
            raise ValueError("No PCE coefficients available. Run compute_polynomial_chaos_expansion first.")
        
        coefficients = pce_result['coefficients']
        basis_functions = pce_result['basis_functions']
        total_variance = pce_result['variance']
        
        # First-order Sobol' indices
        first_order_indices = np.zeros(self.n_params)
        
        for param_idx in range(self.n_params):
            # Sum coefficients for basis functions involving only parameter param_idx
            param_variance = 0
            
            for coeff_idx, multi_index in enumerate(basis_functions):
                # Check if basis function depends only on parameter param_idx
                if sum(multi_index) > 0:  # Skip constant term
                    non_zero_params = [i for i, degree in enumerate(multi_index) if degree > 0]
                    if len(non_zero_params) == 1 and non_zero_params[0] == param_idx:
                        param_variance += coefficients[coeff_idx]**2
            
            first_order_indices[param_idx] = param_variance / total_variance if total_variance > 0 else 0
        
        # Second-order Sobol' indices
        second_order_indices = np.zeros((self.n_params, self.n_params))
        
        for i in range(self.n_params):
            for j in range(i + 1, self.n_params):
                # Sum coefficients for basis functions involving only parameters i and j
                interaction_variance = 0
                
                for coeff_idx, multi_index in enumerate(basis_functions):
                    if sum(multi_index) > 0:  # Skip constant term
                        non_zero_params = [k for k, degree in enumerate(multi_index) if degree > 0]
                        if (len(non_zero_params) == 2 and 
                            set(non_zero_params) == {i, j}):
                            interaction_variance += coefficients[coeff_idx]**2
                
                interaction_index = interaction_variance / total_variance if total_variance > 0 else 0
                second_order_indices[i, j] = interaction_index
                second_order_indices[j, i] = interaction_index  # Symmetric
        
        # Total-order Sobol' indices
        total_order_indices = np.zeros(self.n_params)
        
        for param_idx in range(self.n_params):
            # Sum all coefficients for basis functions involving parameter param_idx
            total_param_variance = 0
            
            for coeff_idx, multi_index in enumerate(basis_functions):
                if sum(multi_index) > 0 and multi_index[param_idx] > 0:
                    total_param_variance += coefficients[coeff_idx]**2
            
            total_order_indices[param_idx] = total_param_variance / total_variance if total_variance > 0 else 0
        
        sobol_indices = {
            'first_order': first_order_indices,
            'second_order': second_order_indices,
            'total_order': total_order_indices,
            'sum_first_order': np.sum(first_order_indices),
            'sum_total_order': np.sum(total_order_indices),
            'total_variance': total_variance
        }
        
        # Store in sensitivity indices
        self.sensitivity_indices = sobol_indices
        
        # Store in history
        self.sobol_history.append(sobol_indices)
        
        logger.info(f"Sobol' indices computed: Σ S₁ = {np.sum(first_order_indices):.4f}")
        return sobol_indices
    
    def compute_realtime_uq_propagation(self, time_span: Tuple[float, float],
                                      initial_probability: Callable,
                                      diffusion_func: Optional[Callable] = None,
                                      source_func: Optional[Callable] = None,
                                      spatial_domain: Tuple[float, float] = (-1, 1),
                                      n_spatial_points: int = 100,
                                      n_time_points: int = 1000) -> Dict[str, np.ndarray]:
        """
        Compute real-time UQ propagation
        ∂²P(x,t)/∂t² = ∇·(D(x,ξ)∇P) + S(x,t,ξ) + Correlation effects
        
        Args:
            time_span: (t_start, t_end) for evolution
            initial_probability: Initial probability density P(x,0)
            diffusion_func: Diffusion coefficient function D(x,ξ)
            source_func: Source term function S(x,t,ξ)
            spatial_domain: (x_min, x_max) spatial boundaries
            n_spatial_points: Number of spatial discretization points
            n_time_points: Number of temporal discretization points
            
        Returns:
            Dictionary containing UQ propagation results
        """
        # Spatial and temporal grids
        x_min, x_max = spatial_domain
        t_start, t_end = time_span
        
        x_grid = np.linspace(x_min, x_max, n_spatial_points)
        t_grid = np.linspace(t_start, t_end, n_time_points)
        dx = (x_max - x_min) / (n_spatial_points - 1)
        dt = (t_end - t_start) / (n_time_points - 1)
        
        # Initialize probability field
        probability_evolution = np.zeros((n_time_points, n_spatial_points))
        
        # Initial condition
        probability_evolution[0, :] = [initial_probability(x) for x in x_grid]
        
        # Generate parameter samples for UQ
        parameter_samples = self.generate_correlated_samples(1000)  # Smaller sample for real-time
        
        # Default functions if not provided
        if diffusion_func is None:
            diffusion_func = lambda x, xi: self.config.diffusion_coefficient * (1 + 0.1 * np.sum(xi))
        
        if source_func is None:
            source_func = lambda x, t, xi: self.config.source_term_strength * np.sin(2 * np.pi * x) * np.cos(2 * np.pi * t)
        
        # Finite difference evolution
        for t_idx in range(1, n_time_points):
            t_current = t_grid[t_idx]
            
            # Average over parameter uncertainty
            probability_new = np.zeros(n_spatial_points)
            
            for sample in parameter_samples[:100]:  # Limit samples for computational efficiency
                # Diffusion term: ∇·(D∇P)
                diffusion_term = np.zeros(n_spatial_points)
                
                for x_idx in range(1, n_spatial_points - 1):
                    x_current = x_grid[x_idx]
                    
                    # Diffusion coefficient at current point
                    D_current = diffusion_func(x_current, sample)
                    
                    # Second derivative approximation
                    d2P_dx2 = (probability_evolution[t_idx - 1, x_idx + 1] - 
                              2 * probability_evolution[t_idx - 1, x_idx] + 
                              probability_evolution[t_idx - 1, x_idx - 1]) / dx**2
                    
                    diffusion_term[x_idx] = D_current * d2P_dx2
                
                # Source term
                source_term = np.array([source_func(x, t_current, sample) for x in x_grid])
                
                # Correlation enhancement (simplified)
                correlation_enhancement = 1 + 0.01 * np.trace(self.correlation_matrix) / self.n_params
                
                # Combined evolution step
                probability_increment = correlation_enhancement * (diffusion_term + source_term) * dt
                probability_new += (probability_evolution[t_idx - 1, :] + probability_increment) / len(parameter_samples[:100])
            
            # Boundary conditions (Neumann - zero flux)
            probability_new[0] = probability_new[1]
            probability_new[-1] = probability_new[-2]
            
            # Normalization to maintain probability conservation
            total_probability = np.trapz(probability_new, x_grid)
            if total_probability > 1e-15:
                probability_new = probability_new / total_probability
            
            probability_evolution[t_idx, :] = probability_new
        
        # Statistical analysis
        final_statistics = self._analyze_probability_field(probability_evolution, x_grid, t_grid)
        
        # Uncertainty bounds
        uncertainty_bounds = self._compute_uncertainty_bounds(probability_evolution, parameter_samples[:100])
        
        propagation_result = {
            'time_grid': t_grid,
            'spatial_grid': x_grid,
            'probability_evolution': probability_evolution,
            'final_probability': probability_evolution[-1, :],
            'statistics': final_statistics,
            'uncertainty_bounds': uncertainty_bounds,
            'correlation_matrix': self.correlation_matrix,
            'parameter_samples_used': parameter_samples[:100]
        }
        
        # Store in history
        self.propagation_history.append(propagation_result)
        
        logger.info(f"Real-time UQ propagation completed: {final_statistics['final_entropy']:.6f} entropy")
        return propagation_result
    
    def _analyze_probability_field(self, probability_field: np.ndarray,
                                 x_grid: np.ndarray, t_grid: np.ndarray) -> Dict[str, float]:
        """Analyze statistical properties of probability field"""
        # Final time slice
        final_prob = probability_field[-1, :]
        
        # Statistical moments
        mean_position = np.trapz(x_grid * final_prob, x_grid)
        variance_position = np.trapz((x_grid - mean_position)**2 * final_prob, x_grid)
        std_position = np.sqrt(variance_position)
        
        # Entropy
        prob_nonzero = final_prob[final_prob > 1e-15]
        entropy = -np.sum(prob_nonzero * np.log(prob_nonzero)) * (x_grid[1] - x_grid[0])
        
        # Temporal stability
        temporal_variation = np.std(np.sum(probability_field, axis=1))
        
        return {
            'final_mean': mean_position,
            'final_variance': variance_position,
            'final_std': std_position,
            'final_entropy': entropy,
            'temporal_stability': temporal_variation,
            'max_probability': np.max(final_prob),
            'probability_spread': np.trapz(np.abs(final_prob), x_grid)
        }
    
    def _compute_uncertainty_bounds(self, probability_field: np.ndarray,
                                  parameter_samples: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute uncertainty bounds using bootstrap sampling"""
        n_bootstrap = 500
        n_time, n_space = probability_field.shape
        
        # Bootstrap statistics
        bootstrap_means = np.zeros((n_bootstrap, n_time))
        bootstrap_stds = np.zeros((n_bootstrap, n_time))
        
        for bootstrap_idx in range(n_bootstrap):
            # Bootstrap sample indices
            sample_indices = np.random.choice(len(parameter_samples), len(parameter_samples), replace=True)
            
            # Compute statistics for bootstrap sample
            for t_idx in range(n_time):
                prob_slice = probability_field[t_idx, :]
                # Simplified bootstrap - add parameter-dependent noise
                noise_level = 0.01 * np.std(parameter_samples[sample_indices], axis=0)
                prob_noisy = prob_slice * (1 + np.mean(noise_level) * np.random.normal(0, 1, len(prob_slice)))
                
                bootstrap_means[bootstrap_idx, t_idx] = np.mean(prob_noisy)
                bootstrap_stds[bootstrap_idx, t_idx] = np.std(prob_noisy)
        
        # Confidence intervals
        confidence_level = 0.95
        alpha = 1 - confidence_level
        lower_percentile = 100 * alpha / 2
        upper_percentile = 100 * (1 - alpha / 2)
        
        mean_lower = np.percentile(bootstrap_means, lower_percentile, axis=0)
        mean_upper = np.percentile(bootstrap_means, upper_percentile, axis=0)
        std_lower = np.percentile(bootstrap_stds, lower_percentile, axis=0)
        std_upper = np.percentile(bootstrap_stds, upper_percentile, axis=0)
        
        return {
            'mean_confidence_lower': mean_lower,
            'mean_confidence_upper': mean_upper,
            'std_confidence_lower': std_lower,
            'std_confidence_upper': std_upper,
            'bootstrap_means': bootstrap_means,
            'bootstrap_stds': bootstrap_stds,
            'confidence_level': confidence_level
        }
    
    def generate_comprehensive_uq_report(self) -> Dict[str, any]:
        """Generate comprehensive UQ propagation report"""
        if not self.propagation_history or not self.sensitivity_indices:
            return {'status': 'insufficient_data', 'message': 'Insufficient data for comprehensive report'}
        
        # Latest results
        latest_propagation = self.propagation_history[-1]
        latest_sobol = self.sensitivity_indices
        
        # Performance metrics
        correlation_effectiveness = 1 / np.linalg.cond(self.correlation_matrix)
        sobol_coverage = latest_sobol['sum_first_order']
        uncertainty_reduction = 1 - latest_propagation['statistics']['temporal_stability']
        
        # Quality assessment
        if (correlation_effectiveness > 0.8 and sobol_coverage > 0.7 and 
            uncertainty_reduction > 0.5):
            uq_grade = 'Excellent'
        elif (correlation_effectiveness > 0.6 and sobol_coverage > 0.5 and 
              uncertainty_reduction > 0.3):
            uq_grade = 'Good'
        elif (correlation_effectiveness > 0.4 and sobol_coverage > 0.3 and 
              uncertainty_reduction > 0.1):
            uq_grade = 'Acceptable'
        else:
            uq_grade = 'Needs Improvement'
        
        report = {
            'uq_performance': {
                'correlation_effectiveness': correlation_effectiveness,
                'sobol_coverage': sobol_coverage,
                'uncertainty_reduction': uncertainty_reduction,
                'correlation_condition_number': np.linalg.cond(self.correlation_matrix),
                'total_variance_explained': latest_sobol['total_variance']
            },
            'sensitivity_analysis': {
                'first_order_indices': latest_sobol['first_order'],
                'dominant_parameter': np.argmax(latest_sobol['first_order']),
                'interaction_strength': np.max(latest_sobol['second_order']),
                'total_order_sum': latest_sobol['sum_total_order']
            },
            'propagation_quality': latest_propagation['statistics'],
            'uncertainty_bounds': latest_propagation['uncertainty_bounds'],
            'uq_grade': uq_grade,
            'n_parameters': self.n_params,
            'polynomial_degree': self.max_degree,
            'monte_carlo_samples': self.n_mc_samples,
            'recommendations': self._generate_uq_recommendations(correlation_effectiveness, sobol_coverage)
        }
        
        return report
    
    def _generate_uq_recommendations(self, correlation_eff: float, sobol_coverage: float) -> List[str]:
        """Generate UQ optimization recommendations"""
        recommendations = []
        
        if correlation_eff < 0.6:
            recommendations.append("Improve correlation matrix conditioning through regularization")
        
        if sobol_coverage < 0.7:
            recommendations.append("Increase polynomial chaos expansion degree for better coverage")
        
        if self.n_mc_samples < 10000:
            recommendations.append("Increase Monte Carlo sample size for better convergence")
        
        if np.linalg.cond(self.correlation_matrix) > 100:
            recommendations.append("Optimize correlation structure for better numerical stability")
        
        if not recommendations:
            recommendations.append("UQ propagation performance is optimal")
        
        return recommendations

def create_realtime_uq_propagation(n_parameters: int = 5,
                                 polynomial_degree: int = 4,
                                 correlation_structure: str = 'enhanced') -> AdvancedRealTimeUQPropagation:
    """Factory function to create real-time UQ propagation system"""
    config = UQPropagationConfig(
        n_parameters=n_parameters,
        max_polynomial_degree=polynomial_degree,
        correlation_structure=correlation_structure
    )
    return AdvancedRealTimeUQPropagation(config)

# Example usage and validation
if __name__ == "__main__":
    # Create real-time UQ propagation system
    uq_system = create_realtime_uq_propagation(n_parameters=5, polynomial_degree=4)
    
    # Define example response function (847× enhancement model)
    def enhancement_response(parameters):
        epsilon_prime, mu_prime, geometry, thermal, control = parameters
        enhancement = 847 * abs((epsilon_prime * mu_prime - 1) / (epsilon_prime * mu_prime + 1))**2
        enhancement *= (1 + 0.1 * geometry) * (1 + 0.05 * thermal) * (1 + 0.02 * control)
        return enhancement
    
    # Compute Polynomial Chaos Expansion
    pce_result = uq_system.compute_polynomial_chaos_expansion(enhancement_response)
    
    print("Advanced Real-Time UQ Propagation Results:")
    print(f"PCE R²: {pce_result['r_squared']:.4f}")
    print(f"Enhancement mean: {pce_result['mean']:.2f}")
    print(f"Enhancement std: {pce_result['std_deviation']:.2f}")
    
    # Compute Sobol' sensitivity indices
    sobol_indices = uq_system.compute_sobol_sensitivity_indices()
    
    print(f"\nSobol' Sensitivity Analysis:")
    parameter_names = ['ε\'', 'μ\'', 'Geometry', 'Thermal', 'Control']
    for i, (name, index) in enumerate(zip(parameter_names, sobol_indices['first_order'])):
        print(f"{name}: S₁ = {index:.4f}")
    
    print(f"Sum of first-order indices: {sobol_indices['sum_first_order']:.4f}")
    
    # Define initial probability for UQ propagation
    def initial_prob(x):
        return np.exp(-x**2 / 0.1) / np.sqrt(0.1 * np.pi)
    
    # Compute real-time UQ propagation
    propagation_result = uq_system.compute_realtime_uq_propagation(
        time_span=(0, 1e-3),
        initial_probability=initial_prob
    )
    
    print(f"\nUQ Propagation Results:")
    print(f"Final entropy: {propagation_result['statistics']['final_entropy']:.6f}")
    print(f"Final mean position: {propagation_result['statistics']['final_mean']:.6f}")
    print(f"Temporal stability: {propagation_result['statistics']['temporal_stability']:.6f}")
    
    # Generate comprehensive report
    report = uq_system.generate_comprehensive_uq_report()
    print(f"\nUQ Grade: {report['uq_grade']}")
    print(f"Correlation effectiveness: {report['uq_performance']['correlation_effectiveness']:.4f}")
    print(f"Dominant parameter: {parameter_names[report['sensitivity_analysis']['dominant_parameter']]}")
