"""
Advanced Quantum-Classical Interface with Lindblad Evolution
Enhanced density matrix evolution with multi-physics coupling matrix

This module implements the advanced quantum-classical interface for digital twin
systems with Lindblad evolution, environmental decoherence, and multi-physics coupling.
"""

import numpy as np
import logging
from typing import Dict, Tuple, Optional, List, Callable
from dataclasses import dataclass
from scipy.constants import hbar, k as k_B
from scipy.linalg import expm, logm
from scipy.integrate import solve_ivp
import scipy.sparse as sp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class QuantumClassicalConfig:
    """Configuration for quantum-classical interface"""
    n_qubits: int = 4
    temperature_classical: float = 300.0  # K
    omega_backaction: float = 2 * np.pi * 1e9  # GHz
    decoherence_suppression_factor: float = 0.1
    coupling_strength_tm: float = 0.1  # Thermal-mechanical
    coupling_strength_te: float = 0.05  # Thermal-electromagnetic
    coupling_strength_me: float = 0.08  # Mechanical-electromagnetic
    coupling_strength_qt: float = 0.02  # Quantum-thermal
    coupling_strength_qe: float = 0.03  # Quantum-electromagnetic
    coupling_strength_qm: float = 0.04  # Quantum-mechanical

class AdvancedQuantumClassicalInterface:
    """
    Advanced Quantum-Classical Interface with Lindblad Evolution
    
    Implements the mathematical framework:
    ρ_digital(t) = e^(-iĤt/ℏ) ρ₀ e^(iĤt/ℏ) + Σₖ LₖρLₖ† - ½{Lₖ†Lₖ, ρ} + γ_qt·𝒟_decoherence[ρ]
    
    Features:
    - Enhanced Lindblad evolution with environmental coupling
    - Multi-physics coupling matrix (4×4)
    - Quantum-classical decoherence control
    - Real-time density matrix evolution
    """
    
    def __init__(self, config: Optional[QuantumClassicalConfig] = None):
        self.config = config or QuantumClassicalConfig()
        
        # Physical constants
        self.hbar = hbar
        self.k_B = k_B
        
        # System dimensions
        self.n_qubits = self.config.n_qubits
        self.hilbert_dim = 2**self.n_qubits
        
        # Initialize quantum system components
        self.hamiltonian = self._construct_effective_hamiltonian()
        self.lindblad_operators = self._construct_lindblad_operators()
        self.coupling_matrix = self._construct_multiphysics_coupling_matrix()
        
        # Density matrix state
        self.rho_current = self._initialize_density_matrix()
        
        # Evolution history
        self.evolution_history = []
        self.decoherence_history = []
        
        # Quantum-classical coupling parameter
        self.gamma_qt = self._compute_quantum_classical_coupling()
        
        logger.info(f"Initialized quantum-classical interface: {self.n_qubits} qubits, dim={self.hilbert_dim}")
        logger.info(f"Quantum-classical coupling: γ_qt = {self.gamma_qt:.6f}")
    
    def _construct_effective_hamiltonian(self) -> np.ndarray:
        """Construct effective Hamiltonian for the quantum system"""
        # Pauli matrices
        sigma_x = np.array([[0, 1], [1, 0]])
        sigma_y = np.array([[0, -1j], [1j, 0]])
        sigma_z = np.array([[1, 0], [0, -1]])
        identity = np.eye(2)
        
        # Single-qubit Hamiltonians
        h_local_freq = 2 * np.pi * 1e9  # GHz frequency
        
        # Multi-qubit Hamiltonian construction
        H_total = np.zeros((self.hilbert_dim, self.hilbert_dim), dtype=complex)
        
        # Single-qubit terms
        for i in range(self.n_qubits):
            # Create operator acting on qubit i
            pauli_list = [identity] * self.n_qubits
            pauli_list[i] = sigma_z
            
            # Tensor product
            h_i = pauli_list[0]
            for j in range(1, self.n_qubits):
                h_i = np.kron(h_i, pauli_list[j])
            
            H_total += h_local_freq * (i + 1) * h_i / self.n_qubits
        
        # Two-qubit coupling terms
        coupling_strength = h_local_freq * 0.1
        for i in range(self.n_qubits - 1):
            # ZZ coupling between adjacent qubits
            pauli_list = [identity] * self.n_qubits
            pauli_list[i] = sigma_z
            pauli_list[i + 1] = sigma_z
            
            # Tensor product
            h_coupling = pauli_list[0]
            for j in range(1, self.n_qubits):
                h_coupling = np.kron(h_coupling, pauli_list[j])
            
            H_total += coupling_strength * h_coupling
        
        return H_total
    
    def _construct_lindblad_operators(self) -> List[np.ndarray]:
        """Construct Lindblad operators for decoherence"""
        # Pauli matrices
        sigma_x = np.array([[0, 1], [1, 0]])
        sigma_y = np.array([[0, -1j], [1j, 0]])
        sigma_z = np.array([[1, 0], [0, -1]])
        identity = np.eye(2)
        
        lindblad_ops = []
        
        # Dephasing operators (σ_z on each qubit)
        for i in range(self.n_qubits):
            pauli_list = [identity] * self.n_qubits
            pauli_list[i] = sigma_z
            
            # Tensor product
            L_dephasing = pauli_list[0]
            for j in range(1, self.n_qubits):
                L_dephasing = np.kron(L_dephasing, pauli_list[j])
            
            lindblad_ops.append(L_dephasing)
        
        # Decay operators (σ_- on each qubit)
        sigma_minus = np.array([[0, 0], [1, 0]])
        for i in range(self.n_qubits):
            pauli_list = [identity] * self.n_qubits
            pauli_list[i] = sigma_minus
            
            # Tensor product
            L_decay = pauli_list[0]
            for j in range(1, self.n_qubits):
                L_decay = np.kron(L_decay, pauli_list[j])
            
            lindblad_ops.append(L_decay)
        
        return lindblad_ops
    
    def _construct_multiphysics_coupling_matrix(self) -> np.ndarray:
        """
        Construct 4×4 multi-physics coupling matrix
        C_enhanced with thermal-mechanical-electromagnetic-quantum coupling
        """
        # Extract coupling parameters
        theta_tm = self.config.coupling_strength_tm
        alpha_te = self.config.coupling_strength_te
        beta_me = self.config.coupling_strength_me
        gamma_qt = self.config.coupling_strength_qt
        sigma_em = self.config.coupling_strength_me  # EM coupling
        phi_qe = self.config.coupling_strength_qe
        
        # Construct 4×4 coupling matrix
        # Order: [Thermal, Mechanical, Electromagnetic, Quantum]
        C_enhanced = np.array([
            [1.0, theta_tm * alpha_te, beta_me * alpha_te, gamma_qt * self.gamma_qt],
            [alpha_te * theta_tm, 1.0, sigma_em * beta_me, phi_qe * gamma_qt],
            [beta_me * alpha_te, beta_me * sigma_em, 1.0, self.gamma_qt * phi_qe],
            [gamma_qt * self.gamma_qt, gamma_qt * phi_qe, phi_qe * self.gamma_qt, 1.0]
        ])
        
        # Ensure matrix is positive definite
        eigenvals = np.linalg.eigvals(C_enhanced)
        min_eigenval = np.min(np.real(eigenvals))
        
        if min_eigenval <= 0:
            # Add regularization to ensure positive definiteness
            regularization = abs(min_eigenval) + 0.01
            C_enhanced += regularization * np.eye(4)
        
        return C_enhanced
    
    def _initialize_density_matrix(self) -> np.ndarray:
        """Initialize density matrix in mixed state"""
        # Start with maximally mixed state
        rho_mixed = np.eye(self.hilbert_dim) / self.hilbert_dim
        
        # Add small coherent component
        coherent_amplitude = 0.1
        psi_coherent = np.zeros(self.hilbert_dim, dtype=complex)
        psi_coherent[0] = 1.0  # Ground state
        psi_coherent[1] = coherent_amplitude  # Small excitation
        
        # Normalize
        psi_coherent = psi_coherent / np.linalg.norm(psi_coherent)
        
        # Pure state density matrix
        rho_pure = np.outer(psi_coherent, np.conj(psi_coherent))
        
        # Mixed initial state
        mixing_parameter = 0.8
        rho_initial = mixing_parameter * rho_mixed + (1 - mixing_parameter) * rho_pure
        
        return rho_initial
    
    def _compute_quantum_classical_coupling(self) -> float:
        """
        Compute quantum-classical coupling parameter
        γ_qt = (ℏω_backaction)/(k_B × T_classical) × decoherence_suppression_factor
        """
        omega_backaction = self.config.omega_backaction
        T_classical = self.config.temperature_classical
        suppression_factor = self.config.decoherence_suppression_factor
        
        gamma_qt = (self.hbar * omega_backaction) / (self.k_B * T_classical) * suppression_factor
        
        return gamma_qt
    
    def compute_enhanced_lindblad_evolution(self, time_span: Tuple[float, float],
                                          external_hamiltonian: Optional[np.ndarray] = None,
                                          time_steps: int = 1000) -> Dict[str, any]:
        """
        Compute enhanced Lindblad evolution with multi-physics coupling
        
        Args:
            time_span: (t_start, t_end) for evolution
            external_hamiltonian: Additional time-dependent Hamiltonian
            time_steps: Number of temporal steps
            
        Returns:
            Dictionary containing evolution results
        """
        t_start, t_end = time_span
        time_array = np.linspace(t_start, t_end, time_steps)
        dt = (t_end - t_start) / time_steps
        
        # Evolution storage
        density_evolution = []
        purity_evolution = []
        entanglement_evolution = []
        decoherence_rates = []
        
        # Initial state
        rho_current = self.rho_current.copy()
        density_evolution.append(rho_current.copy())
        
        for i, t in enumerate(time_array[1:], 1):
            # Time-dependent Hamiltonian
            H_total = self.hamiltonian.copy()
            if external_hamiltonian is not None:
                H_total += external_hamiltonian
            
            # Unitary evolution: -i[H, ρ]/ℏ
            commutator_H = -1j / self.hbar * (H_total @ rho_current - rho_current @ H_total)
            
            # Lindblad decoherence terms: Σₖ LₖρLₖ† - ½{Lₖ†Lₖ, ρ}
            lindblad_term = np.zeros_like(rho_current, dtype=complex)
            
            for L_k in self.lindblad_operators:
                # Decoherence rate (time and coupling-dependent)
                decoherence_rate = self._compute_time_dependent_decoherence_rate(t, L_k)
                
                # Lindblad superoperator
                L_rho_L_dag = L_k @ rho_current @ np.conj(L_k.T)
                L_dag_L = np.conj(L_k.T) @ L_k
                anticommutator = L_dag_L @ rho_current + rho_current @ L_dag_L
                
                lindblad_term += decoherence_rate * (L_rho_L_dag - 0.5 * anticommutator)
            
            # Additional decoherence from multi-physics coupling
            coupling_decoherence = self._compute_coupling_decoherence(rho_current, t)
            
            # Combined evolution equation
            drho_dt = commutator_H + lindblad_term + self.gamma_qt * coupling_decoherence
            
            # Temporal integration (Euler method for stability)
            rho_next = rho_current + drho_dt * dt
            
            # Ensure density matrix properties (Hermiticity, trace, positive semidefinite)
            rho_next = self._enforce_density_matrix_properties(rho_next)
            
            # Store results
            rho_current = rho_next
            density_evolution.append(rho_current.copy())
            
            # Compute quantum measures
            purity = np.real(np.trace(rho_current @ rho_current))
            entanglement = self._compute_entanglement_measure(rho_current)
            decoherence_rate_total = np.real(np.trace(lindblad_term))
            
            purity_evolution.append(purity)
            entanglement_evolution.append(entanglement)
            decoherence_rates.append(decoherence_rate_total)
        
        # Final state analysis
        final_state_analysis = self._analyze_final_quantum_state(rho_current)
        
        # Multi-physics coupling analysis
        coupling_analysis = self._analyze_multiphysics_coupling(density_evolution, time_array)
        
        evolution_result = {
            'time_array': time_array,
            'density_evolution': density_evolution,
            'purity_evolution': np.array(purity_evolution),
            'entanglement_evolution': np.array(entanglement_evolution),
            'decoherence_rates': np.array(decoherence_rates),
            'final_density_matrix': rho_current,
            'final_state_analysis': final_state_analysis,
            'coupling_analysis': coupling_analysis,
            'gamma_qt': self.gamma_qt,
            'coupling_matrix': self.coupling_matrix
        }
        
        # Store in history
        self.evolution_history.append(evolution_result)
        
        logger.info(f"Lindblad evolution completed: final purity = {purity_evolution[-1]:.4f}")
        return evolution_result
    
    def _compute_time_dependent_decoherence_rate(self, t: float, L_k: np.ndarray) -> float:
        """Compute time-dependent decoherence rate"""
        # Base decoherence rate
        base_rate = 1e6  # MHz scale
        
        # Time modulation (environmental fluctuations)
        time_modulation = 1 + 0.1 * np.sin(2 * np.pi * t * 1e3)  # kHz modulation
        
        # Operator-dependent scaling
        operator_norm = np.linalg.norm(L_k)
        operator_scaling = operator_norm / np.sqrt(self.hilbert_dim)
        
        # Multi-physics coupling enhancement
        coupling_enhancement = 1 + 0.05 * np.trace(self.coupling_matrix) / 4
        
        decoherence_rate = base_rate * time_modulation * operator_scaling * coupling_enhancement
        
        return decoherence_rate
    
    def _compute_coupling_decoherence(self, rho: np.ndarray, t: float) -> np.ndarray:
        """
        Compute additional decoherence from multi-physics coupling
        𝒟_decoherence[ρ] based on coupling matrix
        """
        # Extract coupling strengths from matrix
        coupling_trace = np.trace(self.coupling_matrix)
        coupling_off_diagonal = np.sum(np.abs(self.coupling_matrix - np.diag(np.diag(self.coupling_matrix))))
        
        # Time-dependent coupling fluctuations
        temporal_fluctuation = 0.01 * np.cos(2 * np.pi * t * 1e6)  # MHz fluctuations
        
        # Decoherence operator proportional to coupling strength
        coupling_strength = (coupling_trace + coupling_off_diagonal) / 8  # Normalize
        
        # Random Hermitian decoherence operator
        np.random.seed(int(t * 1e9) % 2**32)  # Deterministic randomness based on time
        random_hermitian = np.random.normal(0, 1, (self.hilbert_dim, self.hilbert_dim))
        random_hermitian = (random_hermitian + random_hermitian.T) / 2
        
        # Decoherence superoperator
        decoherence_op = coupling_strength * temporal_fluctuation * random_hermitian
        decoherence_term = decoherence_op @ rho - rho @ decoherence_op
        
        return decoherence_term
    
    def _enforce_density_matrix_properties(self, rho: np.ndarray) -> np.ndarray:
        """Enforce density matrix properties: Hermitian, trace=1, positive semidefinite"""
        # Ensure Hermiticity
        rho_hermitian = (rho + np.conj(rho.T)) / 2
        
        # Ensure trace = 1
        current_trace = np.trace(rho_hermitian)
        if abs(current_trace) > 1e-15:
            rho_normalized = rho_hermitian / current_trace
        else:
            rho_normalized = rho_hermitian
        
        # Ensure positive semidefinite (project onto positive cone)
        eigenvals, eigenvecs = np.linalg.eigh(rho_normalized)
        eigenvals_positive = np.maximum(eigenvals, 0)
        
        # Reconstruct density matrix
        rho_positive = eigenvecs @ np.diag(eigenvals_positive) @ np.conj(eigenvecs.T)
        
        # Renormalize trace
        final_trace = np.trace(rho_positive)
        if abs(final_trace) > 1e-15:
            rho_final = rho_positive / final_trace
        else:
            rho_final = np.eye(self.hilbert_dim) / self.hilbert_dim
        
        return rho_final
    
    def _compute_entanglement_measure(self, rho: np.ndarray) -> float:
        """Compute entanglement measure (simplified von Neumann entropy)"""
        # Compute eigenvalues of density matrix
        eigenvals = np.linalg.eigvals(rho)
        eigenvals_real = np.real(eigenvals)
        
        # Remove zero and negative eigenvalues
        eigenvals_positive = eigenvals_real[eigenvals_real > 1e-15]
        
        if len(eigenvals_positive) == 0:
            return 0.0
        
        # Von Neumann entropy
        entropy = -np.sum(eigenvals_positive * np.log(eigenvals_positive))
        
        return entropy
    
    def _analyze_final_quantum_state(self, rho_final: np.ndarray) -> Dict[str, float]:
        """Analyze final quantum state properties"""
        # Purity
        purity = np.real(np.trace(rho_final @ rho_final))
        
        # Von Neumann entropy
        eigenvals = np.linalg.eigvals(rho_final)
        eigenvals_positive = np.real(eigenvals[eigenvals > 1e-15])
        if len(eigenvals_positive) > 0:
            entropy = -np.sum(eigenvals_positive * np.log(eigenvals_positive))
        else:
            entropy = 0.0
        
        # Trace distance from maximally mixed state
        rho_mixed = np.eye(self.hilbert_dim) / self.hilbert_dim
        trace_distance = 0.5 * np.linalg.norm(rho_final - rho_mixed, ord='nuc')
        
        # Coherence measure (l1 norm of off-diagonal elements)
        rho_diag = np.diag(np.diag(rho_final))
        coherence = np.sum(np.abs(rho_final - rho_diag))
        
        return {
            'purity': purity,
            'entropy': entropy,
            'trace_distance_mixed': trace_distance,
            'coherence_measure': coherence,
            'rank': np.linalg.matrix_rank(rho_final),
            'condition_number': np.linalg.cond(rho_final)
        }
    
    def _analyze_multiphysics_coupling(self, density_evolution: List[np.ndarray],
                                     time_array: np.ndarray) -> Dict[str, any]:
        """Analyze multi-physics coupling effects on quantum evolution"""
        # Compute coupling strength evolution
        coupling_effects = np.zeros(len(time_array))
        
        for i, rho in enumerate(density_evolution):
            # Measure coupling-induced changes
            if i == 0:
                coupling_effects[i] = 0
            else:
                state_change = np.linalg.norm(rho - density_evolution[i-1])
                coupling_effects[i] = state_change
        
        # Coupling matrix analysis
        coupling_eigenvals = np.linalg.eigvals(self.coupling_matrix)
        coupling_condition = np.linalg.cond(self.coupling_matrix)
        
        # Cross-domain correlation analysis
        thermal_quantum_correlation = self.coupling_matrix[0, 3]  # Thermal-Quantum
        mechanical_quantum_correlation = self.coupling_matrix[1, 3]  # Mechanical-Quantum
        electromagnetic_quantum_correlation = self.coupling_matrix[2, 3]  # EM-Quantum
        
        return {
            'coupling_effects_evolution': coupling_effects,
            'average_coupling_effect': np.mean(coupling_effects),
            'max_coupling_effect': np.max(coupling_effects),
            'coupling_eigenvalues': coupling_eigenvals,
            'coupling_condition_number': coupling_condition,
            'thermal_quantum_correlation': thermal_quantum_correlation,
            'mechanical_quantum_correlation': mechanical_quantum_correlation,
            'electromagnetic_quantum_correlation': electromagnetic_quantum_correlation,
            'total_correlation_strength': np.sum(np.abs(self.coupling_matrix - np.eye(4)))
        }
    
    def generate_interface_performance_report(self) -> Dict[str, any]:
        """Generate comprehensive quantum-classical interface performance report"""
        if not self.evolution_history:
            return {'status': 'no_data', 'message': 'No evolution data available'}
        
        latest_evolution = self.evolution_history[-1]
        
        # Performance metrics
        final_purity = latest_evolution['purity_evolution'][-1]
        average_entanglement = np.mean(latest_evolution['entanglement_evolution'])
        decoherence_suppression = 1 - np.mean(latest_evolution['decoherence_rates']) / 1e6  # Relative to MHz scale
        
        # Coupling effectiveness
        coupling_analysis = latest_evolution['coupling_analysis']
        coupling_effectiveness = 1 / (1 + coupling_analysis['coupling_condition_number'])
        
        # Interface quality assessment
        if final_purity > 0.8 and decoherence_suppression > 0.5:
            interface_grade = 'Excellent'
        elif final_purity > 0.6 and decoherence_suppression > 0.3:
            interface_grade = 'Good'
        elif final_purity > 0.4 and decoherence_suppression > 0.1:
            interface_grade = 'Acceptable'
        else:
            interface_grade = 'Needs Improvement'
        
        report = {
            'performance_metrics': {
                'final_purity': final_purity,
                'average_entanglement': average_entanglement,
                'decoherence_suppression': decoherence_suppression,
                'gamma_qt_coupling': self.gamma_qt,
                'coupling_effectiveness': coupling_effectiveness
            },
            'quantum_state_quality': latest_evolution['final_state_analysis'],
            'multi_physics_coupling': coupling_analysis,
            'interface_grade': interface_grade,
            'n_qubits': self.n_qubits,
            'hilbert_dimension': self.hilbert_dim,
            'evolution_count': len(self.evolution_history),
            'recommendations': self._generate_interface_recommendations(final_purity, decoherence_suppression)
        }
        
        return report
    
    def _generate_interface_recommendations(self, purity: float, decoherence_suppression: float) -> List[str]:
        """Generate recommendations for interface optimization"""
        recommendations = []
        
        if purity < 0.8:
            recommendations.append("Increase decoherence suppression factor")
        
        if decoherence_suppression < 0.5:
            recommendations.append("Optimize multi-physics coupling matrix")
        
        if self.gamma_qt < 0.01:
            recommendations.append("Enhance quantum-classical coupling strength")
        
        if np.linalg.cond(self.coupling_matrix) > 10:
            recommendations.append("Improve coupling matrix conditioning")
        
        if not recommendations:
            recommendations.append("Quantum-classical interface performance is optimal")
        
        return recommendations

def create_quantum_classical_interface(n_qubits: int = 4,
                                     temperature: float = 300.0,
                                     decoherence_suppression: float = 0.1) -> AdvancedQuantumClassicalInterface:
    """Factory function to create quantum-classical interface"""
    config = QuantumClassicalConfig(
        n_qubits=n_qubits,
        temperature_classical=temperature,
        decoherence_suppression_factor=decoherence_suppression
    )
    return AdvancedQuantumClassicalInterface(config)

# Example usage and validation
if __name__ == "__main__":
    # Create quantum-classical interface
    interface = create_quantum_classical_interface(n_qubits=4)
    
    # Compute enhanced Lindblad evolution
    time_span = (0, 1e-6)  # 1 μs evolution
    result = interface.compute_enhanced_lindblad_evolution(time_span)
    
    print("Advanced Quantum-Classical Interface Results:")
    print(f"Final purity: {result['purity_evolution'][-1]:.4f}")
    print(f"Final entanglement: {result['entanglement_evolution'][-1]:.4f}")
    print(f"Average decoherence rate: {np.mean(result['decoherence_rates']):.2e} Hz")
    print(f"γ_qt coupling: {result['gamma_qt']:.6f}")
    
    # Multi-physics coupling analysis
    coupling = result['coupling_analysis']
    print(f"\nMulti-Physics Coupling:")
    print(f"Thermal-Quantum correlation: {coupling['thermal_quantum_correlation']:.4f}")
    print(f"Coupling condition number: {coupling['coupling_condition_number']:.2f}")
    print(f"Total correlation strength: {coupling['total_correlation_strength']:.4f}")
    
    # Generate performance report
    report = interface.generate_interface_performance_report()
    print(f"\nInterface Grade: {report['interface_grade']}")
    print(f"Coupling effectiveness: {report['performance_metrics']['coupling_effectiveness']:.4f}")
    print(f"Decoherence suppression: {report['performance_metrics']['decoherence_suppression']:.4f}")
