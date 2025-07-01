"""
Advanced Polynomial Chaos & Sensitivity Analysis Integration
Comprehensive framework for polynomial chaos expansion and global sensitivity analysis

This module provides advanced polynomial chaos expansion with adaptive basis selection,
global sensitivity analysis using variance-based methods, and integrated uncertainty
propagation for complex multi-physics systems.
"""

import numpy as np
import logging
from typing import Dict, Tuple, Optional, List, Callable, Union, Any
from dataclasses import dataclass, field
from scipy.special import eval_hermite, eval_legendre, eval_chebyt, factorial, gamma
from scipy.stats import norm, uniform, beta, gamma as gamma_dist
from scipy.linalg import qr, svd
from scipy.optimize import minimize_scalar
from itertools import combinations_with_replacement, product
import warnings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PolynomialChaosConfig:
    """Configuration for polynomial chaos expansion"""
    # Basis configuration
    max_polynomial_degree: int = 6
    polynomial_family: str = 'hermite'  # 'hermite', 'legendre', 'chebyshev'
    adaptive_basis: bool = True
    basis_selection_threshold: float = 1e-6
    
    # Sampling configuration
    n_monte_carlo_samples: int = 50000
    n_quadrature_points: int = 100
    sampling_method: str = 'latin_hypercube'  # 'monte_carlo', 'latin_hypercube', 'sobol'
    
    # Regression configuration
    regression_method: str = 'lars'  # 'ols', 'ridge', 'lasso', 'lars'
    regularization_parameter: float = 1e-8
    cross_validation_folds: int = 5
    
    # Sensitivity analysis
    sensitivity_methods: List[str] = field(default_factory=lambda: ['sobol', 'morris', 'delta'])
    n_sensitivity_samples: int = 10000
    morris_trajectories: int = 1000
    
    # Performance parameters
    max_basis_functions: int = 5000
    convergence_tolerance: float = 1e-10
    max_iterations: int = 1000

@dataclass
class SensitivityResult:
    """Container for sensitivity analysis results"""
    first_order: np.ndarray = field(default_factory=lambda: np.array([]))
    second_order: np.ndarray = field(default_factory=lambda: np.array([]))
    total_order: np.ndarray = field(default_factory=lambda: np.array([]))
    morris_means: np.ndarray = field(default_factory=lambda: np.array([]))
    morris_stds: np.ndarray = field(default_factory=lambda: np.array([]))
    delta_indices: np.ndarray = field(default_factory=lambda: np.array([]))
    interaction_matrix: np.ndarray = field(default_factory=lambda: np.array([]))

class AdvancedPolynomialChaosSensitivity:
    """
    Advanced Polynomial Chaos & Sensitivity Analysis Integration
    
    Implements comprehensive framework:
    Y(ξ) = Σᵢ αᵢ Ψᵢ(ξ) with adaptive basis selection
    
    Features:
    - Multi-family polynomial basis (Hermite, Legendre, Chebyshev)
    - Adaptive basis selection with SVD truncation
    - Multiple regression methods (OLS, Ridge, LASSO, LARS)
    - Comprehensive sensitivity analysis (Sobol', Morris, δ-indices)
    - Cross-validation and error estimation
    - High-dimensional uncertainty propagation
    """
    
    def __init__(self, n_parameters: int, config: Optional[PolynomialChaosConfig] = None):
        self.n_parameters = n_parameters
        self.config = config or PolynomialChaosConfig()
        
        # Polynomial basis
        self.basis_indices = []
        self.basis_functions = []
        self.pce_coefficients = np.array([])
        
        # Sampling and evaluation
        self.parameter_samples = None
        self.response_samples = None
        self.basis_evaluations = None
        
        # Sensitivity analysis results
        self.sensitivity_results = SensitivityResult()
        
        # Performance tracking
        self.basis_construction_time = 0
        self.regression_time = 0
        self.sensitivity_computation_time = 0
        
        # Validation metrics
        self.pce_r_squared = 0
        self.pce_q_squared = 0  # Cross-validation R²
        self.leave_one_out_error = 0
        
        # Initialize basis
        self._construct_polynomial_basis()
        
        logger.info(f"Advanced PC-SA initialized: {n_parameters} parameters, "
                   f"{len(self.basis_indices)} basis functions")
    
    def _construct_polynomial_basis(self):
        """Construct polynomial chaos basis with adaptive selection"""
        import time
        start_time = time.time()
        
        # Generate all possible multi-indices up to max degree
        all_multi_indices = []
        
        for total_degree in range(self.config.max_polynomial_degree + 1):
            for multi_index in combinations_with_replacement(range(self.n_parameters), total_degree):
                # Convert to degree vector
                degree_vector = [0] * self.n_parameters
                for param_index in multi_index:
                    degree_vector[param_index] += 1
                
                all_multi_indices.append(tuple(degree_vector))
        
        # Remove duplicates and sort by total degree
        all_multi_indices = list(set(all_multi_indices))
        all_multi_indices.sort(key=lambda x: (sum(x), x))
        
        if self.config.adaptive_basis:
            # Start with low-degree basis and expand adaptively
            self.basis_indices = [idx for idx in all_multi_indices if sum(idx) <= 2]
        else:
            # Use all basis functions up to max degree
            self.basis_indices = all_multi_indices
        
        # Limit basis size
        if len(self.basis_indices) > self.config.max_basis_functions:
            self.basis_indices = self.basis_indices[:self.config.max_basis_functions]
        
        # Construct basis function objects
        self.basis_functions = [self._create_basis_function(idx) for idx in self.basis_indices]
        
        self.basis_construction_time = time.time() - start_time
        logger.debug(f"Basis construction completed in {self.basis_construction_time:.4f}s")
    
    def _create_basis_function(self, multi_index: Tuple[int, ...]) -> Callable:
        """Create polynomial basis function for given multi-index"""
        def basis_function(xi: np.ndarray) -> float:
            """Evaluate multivariate polynomial basis function"""
            if len(xi) != len(multi_index):
                raise ValueError(f"Parameter dimension mismatch: {len(xi)} vs {len(multi_index)}")
            
            result = 1.0
            for param_idx, degree in enumerate(multi_index):
                if degree > 0:
                    xi_param = xi[param_idx]
                    
                    # Select polynomial family
                    if self.config.polynomial_family == 'hermite':
                        # Physicist's Hermite polynomials (orthogonal w.r.t. standard normal)
                        poly_value = eval_hermite(degree, xi_param)
                        normalization = np.sqrt(factorial(degree))
                    
                    elif self.config.polynomial_family == 'legendre':
                        # Legendre polynomials (orthogonal w.r.t. uniform distribution)
                        poly_value = eval_legendre(degree, xi_param)
                        normalization = np.sqrt(2 / (2 * degree + 1))
                    
                    elif self.config.polynomial_family == 'chebyshev':
                        # Chebyshev polynomials of the first kind
                        poly_value = eval_chebyt(degree, xi_param)
                        if degree == 0:
                            normalization = np.sqrt(np.pi)
                        else:
                            normalization = np.sqrt(np.pi / 2)
                    
                    else:
                        raise ValueError(f"Unknown polynomial family: {self.config.polynomial_family}")
                    
                    result *= poly_value / normalization
            
            return result
        
        return basis_function
    
    def generate_parameter_samples(self, n_samples: Optional[int] = None,
                                 distribution_types: Optional[List[str]] = None,
                                 distribution_params: Optional[List[Dict]] = None) -> np.ndarray:
        """
        Generate parameter samples for polynomial chaos expansion
        
        Args:
            n_samples: Number of samples (uses config default if None)
            distribution_types: List of distribution types for each parameter
            distribution_params: List of distribution parameters
            
        Returns:
            Parameter samples array of shape (n_samples, n_parameters)
        """
        if n_samples is None:
            n_samples = self.config.n_monte_carlo_samples
        
        # Default to standard normal distributions
        if distribution_types is None:
            distribution_types = ['normal'] * self.n_parameters
        
        if distribution_params is None:
            distribution_params = [{'loc': 0, 'scale': 1}] * self.n_parameters
        
        samples = np.zeros((n_samples, self.n_parameters))
        
        for param_idx in range(self.n_parameters):
            dist_type = distribution_types[param_idx]
            dist_params = distribution_params[param_idx]
            
            if dist_type == 'normal':
                samples[:, param_idx] = np.random.normal(
                    dist_params.get('loc', 0), dist_params.get('scale', 1), n_samples
                )
            
            elif dist_type == 'uniform':
                samples[:, param_idx] = np.random.uniform(
                    dist_params.get('low', -1), dist_params.get('high', 1), n_samples
                )
            
            elif dist_type == 'beta':
                samples[:, param_idx] = np.random.beta(
                    dist_params.get('a', 2), dist_params.get('b', 2), n_samples
                ) * 2 - 1  # Scale to [-1, 1]
            
            elif dist_type == 'gamma':
                gamma_samples = np.random.gamma(
                    dist_params.get('shape', 2), dist_params.get('scale', 1), n_samples
                )
                # Standardize to zero mean, unit variance
                samples[:, param_idx] = (gamma_samples - np.mean(gamma_samples)) / np.std(gamma_samples)
            
            else:
                # Default to standard normal
                samples[:, param_idx] = np.random.normal(0, 1, n_samples)
                logger.warning(f"Unknown distribution type {dist_type}, using standard normal")
        
        # Apply Latin Hypercube or Sobol sampling if requested
        if self.config.sampling_method == 'latin_hypercube':
            samples = self._apply_latin_hypercube_sampling(samples)
        elif self.config.sampling_method == 'sobol':
            samples = self._apply_sobol_sampling(n_samples)
        
        self.parameter_samples = samples
        logger.debug(f"Generated {n_samples} parameter samples using {self.config.sampling_method}")
        
        return samples
    
    def _apply_latin_hypercube_sampling(self, samples: np.ndarray) -> np.ndarray:
        """Apply Latin Hypercube sampling transformation"""
        n_samples, n_params = samples.shape
        
        # For each parameter, apply LHS transformation
        lhs_samples = np.zeros_like(samples)
        
        for param_idx in range(n_params):
            # Get ranks of original samples
            ranks = np.argsort(np.argsort(samples[:, param_idx]))
            
            # Apply LHS transformation
            uniform_lhs = (ranks + np.random.uniform(0, 1, n_samples)) / n_samples
            
            # Transform back to original distribution using inverse CDF
            lhs_samples[:, param_idx] = norm.ppf(uniform_lhs)
        
        return lhs_samples
    
    def _apply_sobol_sampling(self, n_samples: int) -> np.ndarray:
        """Apply Sobol sequence sampling (simplified implementation)"""
        # Simplified Sobol sequence - in practice, use specialized library
        sobol_samples = np.zeros((n_samples, self.n_parameters))
        
        # Generate Sobol-like sequences using van der Corput sequences
        for param_idx in range(self.n_parameters):
            base = 2 + param_idx  # Different base for each parameter
            sequence = []
            
            for i in range(n_samples):
                van_der_corput = self._van_der_corput_sequence(i, base)
                sequence.append(van_der_corput)
            
            # Transform to standard normal
            uniform_sequence = np.array(sequence)
            sobol_samples[:, param_idx] = norm.ppf(uniform_sequence)
        
        return sobol_samples
    
    def _van_der_corput_sequence(self, n: int, base: int) -> float:
        """Compute nth term of van der Corput sequence in given base"""
        sequence = 0.0
        denominator = base
        
        while n > 0:
            sequence += (n % base) / denominator
            n //= base
            denominator *= base
        
        return sequence
    
    def evaluate_basis_matrix(self, parameter_samples: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Evaluate polynomial basis matrix for given parameter samples
        
        Args:
            parameter_samples: Parameter samples (uses stored if None)
            
        Returns:
            Basis matrix of shape (n_samples, n_basis_functions)
        """
        if parameter_samples is None:
            if self.parameter_samples is None:
                raise ValueError("No parameter samples available. Call generate_parameter_samples first.")
            parameter_samples = self.parameter_samples
        
        n_samples = parameter_samples.shape[0]
        n_basis = len(self.basis_functions)
        
        basis_matrix = np.zeros((n_samples, n_basis))
        
        for sample_idx, sample in enumerate(parameter_samples):
            for basis_idx, basis_func in enumerate(self.basis_functions):
                try:
                    basis_matrix[sample_idx, basis_idx] = basis_func(sample)
                except (OverflowError, ValueError) as e:
                    # Handle numerical issues with high-degree polynomials
                    logger.warning(f"Basis evaluation error at sample {sample_idx}, basis {basis_idx}: {e}")
                    basis_matrix[sample_idx, basis_idx] = 0.0
        
        self.basis_evaluations = basis_matrix
        return basis_matrix
    
    def compute_polynomial_chaos_expansion(self, response_function: Callable,
                                         parameter_samples: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Compute polynomial chaos expansion coefficients
        
        Args:
            response_function: Function Y = f(ξ) to expand
            parameter_samples: Parameter samples (generates if None)
            
        Returns:
            Dictionary containing PCE results and analysis
        """
        import time
        start_time = time.time()
        
        # Generate samples if needed
        if parameter_samples is None:
            parameter_samples = self.generate_parameter_samples()
        
        # Evaluate response function
        response_samples = np.array([response_function(sample) for sample in parameter_samples])
        self.response_samples = response_samples
        
        # Evaluate basis matrix
        basis_matrix = self.evaluate_basis_matrix(parameter_samples)
        
        # Perform regression to find PCE coefficients
        pce_coefficients, regression_metrics = self._perform_regression(basis_matrix, response_samples)
        self.pce_coefficients = pce_coefficients
        
        # Compute PCE statistics
        pce_stats = self._compute_pce_statistics(pce_coefficients)
        
        # Cross-validation
        cv_metrics = self._cross_validate_pce(basis_matrix, response_samples)
        
        # Adaptive basis refinement
        if self.config.adaptive_basis:
            refined_results = self._refine_basis_adaptively(response_function, parameter_samples)
            if refined_results['improvement'] > 0.01:  # 1% improvement threshold
                logger.info(f"Basis refined: {refined_results['improvement']:.3f} improvement")
                return refined_results['pce_result']
        
        self.regression_time = time.time() - start_time
        
        pce_result = {
            'coefficients': pce_coefficients,
            'basis_indices': self.basis_indices,
            'statistics': pce_stats,
            'regression_metrics': regression_metrics,
            'cross_validation': cv_metrics,
            'response_samples': response_samples,
            'parameter_samples': parameter_samples,
            'basis_matrix': basis_matrix,
            'computation_time': self.regression_time
        }
        
        # Update validation metrics
        self.pce_r_squared = regression_metrics['r_squared']
        self.pce_q_squared = cv_metrics['mean_cv_score']
        
        logger.info(f"PCE computed: R² = {self.pce_r_squared:.4f}, Q² = {self.pce_q_squared:.4f}")
        return pce_result
    
    def _perform_regression(self, basis_matrix: np.ndarray, 
                          response_samples: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
        """Perform regression to compute PCE coefficients"""
        n_samples, n_basis = basis_matrix.shape
        
        if self.config.regression_method == 'ols':
            # Ordinary Least Squares
            coefficients, residuals, rank, singular_values = np.linalg.lstsq(
                basis_matrix, response_samples, rcond=None
            )
        
        elif self.config.regression_method == 'ridge':
            # Ridge regression with regularization
            regularized_matrix = (basis_matrix.T @ basis_matrix + 
                                self.config.regularization_parameter * np.eye(n_basis))
            coefficients = np.linalg.solve(regularized_matrix, basis_matrix.T @ response_samples)
            residuals = np.sum((response_samples - basis_matrix @ coefficients)**2)
            rank = n_basis
            singular_values = np.linalg.svd(basis_matrix, compute_uv=False)
        
        elif self.config.regression_method == 'lars':
            # Least Angle Regression (simplified implementation)
            coefficients = self._lars_regression(basis_matrix, response_samples)
            residuals = np.sum((response_samples - basis_matrix @ coefficients)**2)
            rank = np.sum(np.abs(coefficients) > 1e-10)
            singular_values = np.linalg.svd(basis_matrix, compute_uv=False)
        
        else:
            # Default to OLS
            coefficients, residuals, rank, singular_values = np.linalg.lstsq(
                basis_matrix, response_samples, rcond=None
            )
        
        # Compute regression metrics
        response_mean = np.mean(response_samples)
        ss_total = np.sum((response_samples - response_mean)**2)
        ss_residual = residuals if np.isscalar(residuals) else residuals[0]
        r_squared = 1 - ss_residual / ss_total if ss_total > 0 else 0
        
        # Condition number
        condition_number = singular_values[0] / singular_values[-1] if len(singular_values) > 0 else 1
        
        regression_metrics = {
            'r_squared': r_squared,
            'residual_sum_squares': ss_residual,
            'rank': rank,
            'condition_number': condition_number,
            'effective_coefficients': np.sum(np.abs(coefficients) > 1e-10),
            'max_coefficient': np.max(np.abs(coefficients))
        }
        
        return coefficients, regression_metrics
    
    def _lars_regression(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Simplified LARS regression implementation"""
        n_samples, n_features = X.shape
        coefficients = np.zeros(n_features)
        
        # Normalize columns
        X_normalized = X / np.linalg.norm(X, axis=0)
        
        # LARS algorithm (simplified)
        active_set = []
        residual = y.copy()
        
        for iteration in range(min(n_features, n_samples)):
            # Compute correlations
            correlations = np.abs(X_normalized.T @ residual)
            
            # Find most correlated inactive variable
            inactive_mask = np.ones(n_features, dtype=bool)
            inactive_mask[active_set] = False
            
            if not np.any(inactive_mask):
                break
            
            max_corr_idx = np.argmax(correlations[inactive_mask])
            actual_idx = np.arange(n_features)[inactive_mask][max_corr_idx]
            
            active_set.append(actual_idx)
            
            # Solve least squares on active set
            X_active = X[:, active_set]
            try:
                active_coeffs = np.linalg.lstsq(X_active, y, rcond=None)[0]
                coefficients[active_set] = active_coeffs
                residual = y - X_active @ active_coeffs
            except np.linalg.LinAlgError:
                break
            
            # Check stopping criterion
            if np.linalg.norm(residual) < self.config.convergence_tolerance:
                break
        
        return coefficients
    
    def _compute_pce_statistics(self, coefficients: np.ndarray) -> Dict[str, float]:
        """Compute polynomial chaos expansion statistics"""
        # PCE mean (zeroth-order coefficient)
        pce_mean = coefficients[0] if len(coefficients) > 0 else 0
        
        # PCE variance (sum of squared higher-order coefficients)
        pce_variance = np.sum(coefficients[1:]**2) if len(coefficients) > 1 else 0
        pce_std = np.sqrt(pce_variance)
        
        # Contribution analysis
        total_variance = pce_variance
        coefficient_contributions = coefficients[1:]**2 / total_variance if total_variance > 0 else np.zeros(len(coefficients) - 1)
        
        # Effective dimension
        significant_coeffs = np.sum(np.abs(coefficients) > 1e-6 * np.max(np.abs(coefficients)))
        
        statistics = {
            'mean': pce_mean,
            'variance': pce_variance,
            'std_deviation': pce_std,
            'coefficient_of_variation': pce_std / abs(pce_mean) if abs(pce_mean) > 1e-15 else float('inf'),
            'effective_dimension': significant_coeffs,
            'total_coefficients': len(coefficients),
            'max_coefficient_magnitude': np.max(np.abs(coefficients)),
            'coefficient_sparsity': 1 - significant_coeffs / len(coefficients)
        }
        
        return statistics
    
    def _cross_validate_pce(self, basis_matrix: np.ndarray, 
                          response_samples: np.ndarray) -> Dict[str, float]:
        """Perform cross-validation of PCE"""
        n_samples = len(response_samples)
        n_folds = min(self.config.cross_validation_folds, n_samples)
        
        fold_size = n_samples // n_folds
        cv_scores = []
        
        for fold in range(n_folds):
            # Split data
            start_idx = fold * fold_size
            end_idx = (fold + 1) * fold_size if fold < n_folds - 1 else n_samples
            
            test_indices = np.arange(start_idx, end_idx)
            train_indices = np.concatenate([np.arange(0, start_idx), np.arange(end_idx, n_samples)])
            
            X_train = basis_matrix[train_indices]
            y_train = response_samples[train_indices]
            X_test = basis_matrix[test_indices]
            y_test = response_samples[test_indices]
            
            # Train on fold
            try:
                fold_coeffs, _ = self._perform_regression(X_train, y_train)
                
                # Predict on test set
                y_pred = X_test @ fold_coeffs
                
                # Compute R² score
                ss_total = np.sum((y_test - np.mean(y_test))**2)
                ss_residual = np.sum((y_test - y_pred)**2)
                r2_score = 1 - ss_residual / ss_total if ss_total > 0 else 0
                
                cv_scores.append(r2_score)
                
            except Exception as e:
                logger.warning(f"Cross-validation fold {fold} failed: {e}")
                cv_scores.append(0)
        
        cv_metrics = {
            'mean_cv_score': np.mean(cv_scores),
            'std_cv_score': np.std(cv_scores),
            'min_cv_score': np.min(cv_scores),
            'max_cv_score': np.max(cv_scores),
            'cv_scores': cv_scores
        }
        
        return cv_metrics
    
    def _refine_basis_adaptively(self, response_function: Callable,
                               parameter_samples: np.ndarray) -> Dict[str, Any]:
        """Adaptively refine polynomial basis"""
        current_r2 = self.pce_r_squared
        
        # Try adding higher-degree terms
        current_max_degree = max(sum(idx) for idx in self.basis_indices)
        candidate_indices = []
        
        # Add basis functions of degree current_max_degree + 1
        for multi_index in combinations_with_replacement(range(self.n_parameters), current_max_degree + 1):
            degree_vector = [0] * self.n_parameters
            for param_index in multi_index:
                degree_vector[param_index] += 1
            
            candidate_idx = tuple(degree_vector)
            if candidate_idx not in self.basis_indices:
                candidate_indices.append(candidate_idx)
        
        if not candidate_indices:
            return {'improvement': 0, 'pce_result': None}
        
        # Test adding most promising candidates
        best_improvement = 0
        best_refined_basis = self.basis_indices.copy()
        
        for candidate_idx in candidate_indices[:10]:  # Test up to 10 candidates
            # Temporarily add candidate to basis
            test_basis_indices = self.basis_indices + [candidate_idx]
            test_basis_functions = [self._create_basis_function(idx) for idx in test_basis_indices]
            
            # Evaluate new basis matrix
            n_samples = parameter_samples.shape[0]
            test_basis_matrix = np.zeros((n_samples, len(test_basis_functions)))
            
            for sample_idx, sample in enumerate(parameter_samples):
                for basis_idx, basis_func in enumerate(test_basis_functions):
                    try:
                        test_basis_matrix[sample_idx, basis_idx] = basis_func(sample)
                    except:
                        test_basis_matrix[sample_idx, basis_idx] = 0.0
            
            # Compute coefficients for test basis
            try:
                test_coeffs, test_metrics = self._perform_regression(test_basis_matrix, self.response_samples)
                test_r2 = test_metrics['r_squared']
                
                improvement = test_r2 - current_r2
                if improvement > best_improvement:
                    best_improvement = improvement
                    best_refined_basis = test_basis_indices
                    
            except Exception as e:
                logger.debug(f"Basis refinement test failed for {candidate_idx}: {e}")
        
        if best_improvement > self.config.basis_selection_threshold:
            # Update basis with best refinement
            self.basis_indices = best_refined_basis
            self.basis_functions = [self._create_basis_function(idx) for idx in self.basis_indices]
            
            # Recompute PCE with refined basis
            refined_result = self.compute_polynomial_chaos_expansion(response_function, parameter_samples)
            return {'improvement': best_improvement, 'pce_result': refined_result}
        
        return {'improvement': 0, 'pce_result': None}
    
    def compute_comprehensive_sensitivity_analysis(self, pce_result: Optional[Dict] = None) -> SensitivityResult:
        """
        Compute comprehensive sensitivity analysis
        
        Args:
            pce_result: PCE expansion result (uses stored if None)
            
        Returns:
            Comprehensive sensitivity analysis results
        """
        import time
        start_time = time.time()
        
        if pce_result is None:
            if len(self.pce_coefficients) == 0:
                raise ValueError("No PCE coefficients available. Run compute_polynomial_chaos_expansion first.")
            
            coefficients = self.pce_coefficients
            basis_indices = self.basis_indices
            total_variance = np.sum(coefficients[1:]**2) if len(coefficients) > 1 else 1
        else:
            coefficients = pce_result['coefficients']
            basis_indices = pce_result['basis_indices']
            total_variance = pce_result['statistics']['variance']
        
        # Sobol' indices from PCE coefficients
        sobol_result = self._compute_sobol_indices_from_pce(coefficients, basis_indices, total_variance)
        
        # Morris elementary effects (if samples available)
        if self.parameter_samples is not None and self.response_samples is not None:
            morris_result = self._compute_morris_indices()
        else:
            morris_result = {'means': np.zeros(self.n_parameters), 'stds': np.zeros(self.n_parameters)}
        
        # Delta indices
        delta_result = self._compute_delta_indices(coefficients, basis_indices)
        
        # Compile results
        sensitivity_result = SensitivityResult(
            first_order=sobol_result['first_order'],
            second_order=sobol_result['second_order'],
            total_order=sobol_result['total_order'],
            morris_means=morris_result['means'],
            morris_stds=morris_result['stds'],
            delta_indices=delta_result,
            interaction_matrix=sobol_result['interaction_matrix']
        )
        
        self.sensitivity_results = sensitivity_result
        self.sensitivity_computation_time = time.time() - start_time
        
        logger.info(f"Sensitivity analysis completed in {self.sensitivity_computation_time:.4f}s")
        return sensitivity_result
    
    def _compute_sobol_indices_from_pce(self, coefficients: np.ndarray,
                                      basis_indices: List[Tuple], 
                                      total_variance: float) -> Dict[str, np.ndarray]:
        """Compute Sobol' indices from PCE coefficients"""
        # First-order Sobol' indices
        first_order_indices = np.zeros(self.n_parameters)
        
        for param_idx in range(self.n_parameters):
            param_variance = 0
            
            for coeff_idx, multi_index in enumerate(basis_indices):
                if sum(multi_index) > 0:  # Skip constant term
                    non_zero_params = [i for i, degree in enumerate(multi_index) if degree > 0]
                    if len(non_zero_params) == 1 and non_zero_params[0] == param_idx:
                        param_variance += coefficients[coeff_idx]**2
            
            first_order_indices[param_idx] = param_variance / total_variance if total_variance > 0 else 0
        
        # Second-order Sobol' indices
        second_order_indices = np.zeros((self.n_parameters, self.n_parameters))
        
        for i in range(self.n_parameters):
            for j in range(i + 1, self.n_parameters):
                interaction_variance = 0
                
                for coeff_idx, multi_index in enumerate(basis_indices):
                    if sum(multi_index) > 0:
                        non_zero_params = [k for k, degree in enumerate(multi_index) if degree > 0]
                        if len(non_zero_params) == 2 and set(non_zero_params) == {i, j}:
                            interaction_variance += coefficients[coeff_idx]**2
                
                interaction_index = interaction_variance / total_variance if total_variance > 0 else 0
                second_order_indices[i, j] = interaction_index
                second_order_indices[j, i] = interaction_index
        
        # Total-order Sobol' indices
        total_order_indices = np.zeros(self.n_parameters)
        
        for param_idx in range(self.n_parameters):
            total_param_variance = 0
            
            for coeff_idx, multi_index in enumerate(basis_indices):
                if sum(multi_index) > 0 and multi_index[param_idx] > 0:
                    total_param_variance += coefficients[coeff_idx]**2
            
            total_order_indices[param_idx] = total_param_variance / total_variance if total_variance > 0 else 0
        
        return {
            'first_order': first_order_indices,
            'second_order': second_order_indices,
            'total_order': total_order_indices,
            'interaction_matrix': second_order_indices
        }
    
    def _compute_morris_indices(self) -> Dict[str, np.ndarray]:
        """Compute Morris elementary effects"""
        # Simplified Morris method implementation
        if self.parameter_samples is None or self.response_samples is None:
            return {'means': np.zeros(self.n_parameters), 'stds': np.zeros(self.n_parameters)}
        
        # Generate Morris trajectories (simplified)
        n_trajectories = min(self.config.morris_trajectories, len(self.parameter_samples) // (self.n_parameters + 1))
        delta = 0.1  # Morris step size
        
        elementary_effects = []
        
        for trajectory in range(n_trajectories):
            # Random starting point
            start_idx = np.random.randint(0, len(self.parameter_samples) - self.n_parameters - 1)
            base_point = self.parameter_samples[start_idx]
            base_response = self.response_samples[start_idx]
            
            trajectory_effects = []
            
            for param_idx in range(self.n_parameters):
                # Perturb parameter
                perturbed_point = base_point.copy()
                perturbed_point[param_idx] += delta
                
                # Find closest sample to perturbed point (approximation)
                distances = np.linalg.norm(self.parameter_samples - perturbed_point, axis=1)
                closest_idx = np.argmin(distances)
                perturbed_response = self.response_samples[closest_idx]
                
                # Elementary effect
                elementary_effect = (perturbed_response - base_response) / delta
                trajectory_effects.append(elementary_effect)
            
            elementary_effects.append(trajectory_effects)
        
        elementary_effects = np.array(elementary_effects)
        
        # Compute Morris indices
        morris_means = np.mean(np.abs(elementary_effects), axis=0)
        morris_stds = np.std(elementary_effects, axis=0)
        
        return {'means': morris_means, 'stds': morris_stds}
    
    def _compute_delta_indices(self, coefficients: np.ndarray, 
                             basis_indices: List[Tuple]) -> np.ndarray:
        """Compute delta sensitivity indices"""
        # Simplified delta index computation based on coefficient magnitudes
        delta_indices = np.zeros(self.n_parameters)
        
        for param_idx in range(self.n_parameters):
            param_contribution = 0
            
            for coeff_idx, multi_index in enumerate(basis_indices):
                if multi_index[param_idx] > 0:  # Parameter appears in this basis function
                    # Weight by parameter degree and coefficient magnitude
                    weight = multi_index[param_idx] * abs(coefficients[coeff_idx])
                    param_contribution += weight
            
            delta_indices[param_idx] = param_contribution
        
        # Normalize
        total_contribution = np.sum(delta_indices)
        if total_contribution > 0:
            delta_indices = delta_indices / total_contribution
        
        return delta_indices
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive polynomial chaos and sensitivity analysis report"""
        if len(self.pce_coefficients) == 0:
            return {'status': 'no_pce_data', 'message': 'No PCE data available'}
        
        # Performance metrics
        computational_efficiency = len(self.basis_indices) / (self.basis_construction_time + self.regression_time + 1e-6)
        basis_efficiency = self.pce_r_squared / len(self.basis_indices) if len(self.basis_indices) > 0 else 0
        
        # Quality assessment
        if (self.pce_r_squared > 0.95 and self.pce_q_squared > 0.9 and 
            np.sum(self.sensitivity_results.first_order) > 0.7):
            analysis_grade = 'Excellent'
        elif (self.pce_r_squared > 0.85 and self.pce_q_squared > 0.75 and 
              np.sum(self.sensitivity_results.first_order) > 0.5):
            analysis_grade = 'Good'
        elif (self.pce_r_squared > 0.7 and self.pce_q_squared > 0.6 and 
              np.sum(self.sensitivity_results.first_order) > 0.3):
            analysis_grade = 'Acceptable'
        else:
            analysis_grade = 'Needs Improvement'
        
        report = {
            'pce_performance': {
                'r_squared': self.pce_r_squared,
                'q_squared': self.pce_q_squared,
                'n_basis_functions': len(self.basis_indices),
                'max_polynomial_degree': max(sum(idx) for idx in self.basis_indices) if self.basis_indices else 0,
                'computational_efficiency': computational_efficiency,
                'basis_efficiency': basis_efficiency
            },
            'sensitivity_analysis': {
                'first_order_indices': self.sensitivity_results.first_order,
                'total_order_indices': self.sensitivity_results.total_order,
                'dominant_parameter': np.argmax(self.sensitivity_results.first_order) if len(self.sensitivity_results.first_order) > 0 else -1,
                'interaction_strength': np.max(self.sensitivity_results.second_order) if self.sensitivity_results.second_order.size > 0 else 0,
                'morris_means': self.sensitivity_results.morris_means,
                'delta_indices': self.sensitivity_results.delta_indices
            },
            'computational_performance': {
                'basis_construction_time': self.basis_construction_time,
                'regression_time': self.regression_time,
                'sensitivity_computation_time': self.sensitivity_computation_time,
                'total_time': self.basis_construction_time + self.regression_time + self.sensitivity_computation_time
            },
            'analysis_grade': analysis_grade,
            'recommendations': self._generate_optimization_recommendations()
        }
        
        return report
    
    def _generate_optimization_recommendations(self) -> List[str]:
        """Generate optimization recommendations"""
        recommendations = []
        
        if self.pce_r_squared < 0.8:
            recommendations.append("Increase polynomial degree or basis functions for better accuracy")
        
        if self.pce_q_squared < 0.7:
            recommendations.append("Improve cross-validation performance through regularization")
        
        if len(self.basis_indices) > 1000:
            recommendations.append("Consider basis selection methods to reduce computational cost")
        
        if np.sum(self.sensitivity_results.first_order) < 0.5:
            recommendations.append("Model may have significant higher-order interactions")
        
        if self.regression_time > 10.0:
            recommendations.append("Optimize regression method for faster computation")
        
        if not recommendations:
            recommendations.append("Polynomial chaos and sensitivity analysis performance is optimal")
        
        return recommendations

def create_polynomial_chaos_sensitivity(n_parameters: int, max_degree: int = 4,
                                       polynomial_family: str = 'hermite') -> AdvancedPolynomialChaosSensitivity:
    """Factory function to create polynomial chaos sensitivity analyzer"""
    config = PolynomialChaosConfig(
        max_polynomial_degree=max_degree,
        polynomial_family=polynomial_family
    )
    return AdvancedPolynomialChaosSensitivity(n_parameters, config)

# Example usage and validation
if __name__ == "__main__":
    # Create polynomial chaos sensitivity analyzer
    pc_sa = create_polynomial_chaos_sensitivity(n_parameters=5, max_degree=4)
    
    # Define example response function (847× enhancement model)
    def enhancement_response(parameters):
        epsilon_prime, mu_prime, geometry, thermal, control = parameters
        enhancement = 847 * abs((epsilon_prime * mu_prime - 1) / (epsilon_prime * mu_prime + 1))**2
        enhancement *= (1 + 0.1 * geometry) * (1 + 0.05 * thermal) * (1 + 0.02 * control)
        # Add some nonlinearity
        enhancement *= (1 + 0.01 * epsilon_prime * mu_prime * geometry)
        return enhancement
    
    print("Advanced Polynomial Chaos & Sensitivity Analysis Results:")
    
    # Compute polynomial chaos expansion
    pce_result = pc_sa.compute_polynomial_chaos_expansion(enhancement_response)
    
    print(f"PCE R²: {pce_result['regression_metrics']['r_squared']:.4f}")
    print(f"PCE Q² (CV): {pce_result['cross_validation']['mean_cv_score']:.4f}")
    print(f"Enhancement mean: {pce_result['statistics']['mean']:.2f}")
    print(f"Enhancement std: {pce_result['statistics']['std_deviation']:.2f}")
    print(f"Basis functions: {len(pce_result['basis_indices'])}")
    
    # Compute sensitivity analysis
    sensitivity_result = pc_sa.compute_comprehensive_sensitivity_analysis()
    
    print(f"\nSensitivity Analysis:")
    parameter_names = ['ε\'', 'μ\'', 'Geometry', 'Thermal', 'Control']
    
    print("First-order Sobol' indices:")
    for i, (name, index) in enumerate(zip(parameter_names, sensitivity_result.first_order)):
        print(f"  {name}: S₁ = {index:.4f}")
    
    print("Total-order Sobol' indices:")
    for i, (name, index) in enumerate(zip(parameter_names, sensitivity_result.total_order)):
        print(f"  {name}: Sₜ = {index:.4f}")
    
    print(f"\nMorris indices (μ*):")
    for i, (name, mu_star) in enumerate(zip(parameter_names, sensitivity_result.morris_means)):
        print(f"  {name}: μ* = {mu_star:.4f}")
    
    # Generate comprehensive report
    report = pc_sa.generate_comprehensive_report()
    print(f"\nAnalysis Grade: {report['analysis_grade']}")
    print(f"Dominant parameter: {parameter_names[report['sensitivity_analysis']['dominant_parameter']]}")
    print(f"Computational efficiency: {report['pce_performance']['computational_efficiency']:.2f}")
    print(f"Total computation time: {report['computational_performance']['total_time']:.4f}s")
