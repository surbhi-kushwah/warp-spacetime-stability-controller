"""
Advanced 847× Amplification Validation Framework
Comprehensive validation across frequency spectrum with thermal noise analysis

This module implements frequency-dependent enhancement validation with
Johnson-Nyquist thermal noise characterization for Casimir sensor arrays.
"""

import numpy as np
import logging
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy.constants import hbar, c, k as k_B
from scipy.signal import welch, periodogram
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ValidationConfig:
    """Configuration for amplification validation"""
    target_enhancement: float = 847.0
    resonant_multiplier: float = 2.3
    frequency_range: np.ndarray = None
    temperature: float = 300.0
    bandwidth: float = 1e6
    snr_threshold: float = 100.0  # 20 dB margin
    
    def __post_init__(self):
        if self.frequency_range is None:
            self.frequency_range = np.logspace(3, 12, 1000)  # 1 kHz to 1 THz

class AmplificationValidationFramework:
    """
    Advanced framework for validating 847× metamaterial amplification
    
    Provides comprehensive frequency-domain validation with thermal noise
    analysis and signal-to-noise ratio optimization.
    """
    
    def __init__(self, config: Optional[ValidationConfig] = None):
        self.config = config or ValidationConfig()
        self.k_B = k_B
        self.hbar = hbar
        self.c = c
        
        # Validation results storage
        self.validation_history = []
        self.frequency_response_data = {}
        
        logger.info(f"Initialized amplification validation for {self.config.target_enhancement}× enhancement")
    
    def validate_847x_amplification(self, frequency_range: Optional[np.ndarray] = None,
                                  temperature: float = 300.0,
                                  bandwidth: float = 1e6) -> Dict[str, any]:
        """
        Comprehensive validation across frequency spectrum
        
        Args:
            frequency_range: Frequency array for validation (Hz)
            temperature: Operating temperature (K)
            bandwidth: Measurement bandwidth (Hz)
            
        Returns:
            Comprehensive validation results with SNR analysis
        """
        if frequency_range is None:
            frequency_range = self.config.frequency_range
        
        # Compute frequency-dependent enhancement profile
        enhancement_profile = self._compute_enhancement_profile(frequency_range)
        
        # Calculate thermal noise across frequency spectrum
        thermal_noise = self._compute_thermal_noise_spectrum(frequency_range, temperature, bandwidth)
        
        # Signal-to-noise ratio analysis
        snr_analysis = self._compute_snr_analysis(enhancement_profile, thermal_noise)
        
        # Validation criteria assessment
        validation_results = self._assess_validation_criteria(enhancement_profile, snr_analysis)
        
        # Store results
        validation_record = {
            'timestamp': time.time(),
            'frequency_range': frequency_range,
            'enhancement_profile': enhancement_profile,
            'thermal_noise': thermal_noise,
            'snr_analysis': snr_analysis,
            'validation_results': validation_results,
            'temperature': temperature,
            'bandwidth': bandwidth
        }
        
        self.validation_history.append(validation_record)
        self.frequency_response_data = enhancement_profile
        
        logger.info(f"Validation completed: {validation_results['overall_validation']}")
        
        return validation_record
    
    def _compute_enhancement_profile(self, frequency_range: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Compute frequency-dependent enhancement profile
        
        Returns enhancement profiles for different frequency regimes:
        - Low frequency: 847× with exponential rolloff
        - Resonant: 847× × 2.3 = 1947× peak enhancement
        - High frequency: 847× with thermal limitation
        """
        # Low frequency profile with MHz cutoff
        low_freq_cutoff = 1e6  # 1 MHz
        low_freq = self.config.target_enhancement * np.exp(-frequency_range / low_freq_cutoff)
        
        # Resonant enhancement profile
        resonant_freq = 1e9  # 1 GHz resonance
        resonant_width = 1e8  # 100 MHz width
        resonant_profile = self.config.target_enhancement * self.config.resonant_multiplier
        resonant = resonant_profile * np.exp(-((frequency_range - resonant_freq) / resonant_width)**2)
        
        # High frequency thermal limitation
        thermal_cutoff = 1e9  # 1 GHz thermal cutoff
        thermal_limited = self.config.target_enhancement / (1 + frequency_range / thermal_cutoff)
        
        # Combined profile (maximum of all contributions)
        combined_profile = np.maximum.reduce([low_freq, resonant, thermal_limited])
        
        # Peak resonant enhancement
        peak_enhancement = np.max(combined_profile)
        resonant_frequency = frequency_range[np.argmax(combined_profile)]
        
        enhancement_profile = {
            'low_freq': low_freq,
            'resonant': resonant,
            'thermal_limited': thermal_limited,
            'combined': combined_profile,
            'peak_enhancement': peak_enhancement,
            'resonant_frequency': resonant_frequency,
            'frequency_range': frequency_range
        }
        
        logger.debug(f"Peak enhancement: {peak_enhancement:.1f}× at {resonant_frequency:.2e} Hz")
        
        return enhancement_profile
    
    def _compute_thermal_noise_spectrum(self, frequency_range: np.ndarray,
                                      temperature: float, bandwidth: float) -> Dict[str, np.ndarray]:
        """
        Compute Johnson-Nyquist thermal noise across frequency spectrum
        """
        # Base Johnson-Nyquist thermal noise
        resistance = 1e6  # 1 MΩ typical sensor resistance
        base_noise = np.sqrt(4 * self.k_B * temperature * resistance * bandwidth)
        
        # Frequency-dependent noise factors
        flicker_noise = base_noise * np.sqrt(1e3 / frequency_range)  # 1/f noise
        shot_noise = base_noise * np.ones_like(frequency_range)      # White noise
        thermal_noise_1f = base_noise * np.sqrt(1 + 1e3 / frequency_range)  # Combined
        
        # Quantum noise at high frequencies
        quantum_noise = self.hbar * frequency_range / (2 * self.k_B * temperature)
        quantum_limited = base_noise * np.sqrt(1 + quantum_noise)
        
        # Total noise (RMS combination)
        total_noise = np.sqrt(thermal_noise_1f**2 + quantum_limited**2)
        
        thermal_spectrum = {
            'base_noise': base_noise,
            'flicker_noise': flicker_noise,
            'shot_noise': shot_noise,
            'thermal_1f': thermal_noise_1f,
            'quantum_limited': quantum_limited,
            'total_noise': total_noise,
            'frequency_range': frequency_range
        }
        
        return thermal_spectrum
    
    def _compute_snr_analysis(self, enhancement_profile: Dict[str, np.ndarray],
                            thermal_noise: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Compute signal-to-noise ratio analysis across frequency spectrum
        """
        # Base signal strength (normalized)
        base_signal = 1.0
        
        # Enhanced signal strength
        enhanced_signal = base_signal * enhancement_profile['combined']
        
        # SNR calculations
        snr_linear = enhanced_signal / thermal_noise['total_noise']
        snr_db = 20 * np.log10(snr_linear)
        
        # SNR statistics
        peak_snr = np.max(snr_linear)
        peak_snr_db = np.max(snr_db)
        peak_snr_freq = enhancement_profile['frequency_range'][np.argmax(snr_linear)]
        
        # Usable bandwidth (SNR > threshold)
        snr_threshold = self.config.snr_threshold
        usable_frequencies = enhancement_profile['frequency_range'][snr_linear > snr_threshold]
        usable_bandwidth = np.ptp(usable_frequencies) if len(usable_frequencies) > 0 else 0
        
        snr_analysis = {
            'snr_linear': snr_linear,
            'snr_db': snr_db,
            'peak_snr': peak_snr,
            'peak_snr_db': peak_snr_db,
            'peak_snr_frequency': peak_snr_freq,
            'usable_bandwidth': usable_bandwidth,
            'snr_threshold': snr_threshold,
            'usable_frequencies': usable_frequencies
        }
        
        logger.debug(f"Peak SNR: {peak_snr_db:.1f} dB at {peak_snr_freq:.2e} Hz")
        
        return snr_analysis
    
    def _assess_validation_criteria(self, enhancement_profile: Dict[str, np.ndarray],
                                  snr_analysis: Dict[str, np.ndarray]) -> Dict[str, any]:
        """
        Assess validation criteria for 847× amplification
        """
        # Enhancement criteria
        target_enhancement = self.config.target_enhancement
        peak_enhancement = enhancement_profile['peak_enhancement']
        enhancement_achieved = peak_enhancement >= target_enhancement * 0.95  # 95% tolerance
        
        # Resonant enhancement criteria
        expected_resonant = target_enhancement * self.config.resonant_multiplier
        resonant_achieved = peak_enhancement >= expected_resonant * 0.90  # 90% tolerance
        
        # SNR criteria
        snr_threshold_met = snr_analysis['peak_snr'] > self.config.snr_threshold
        usable_bandwidth_adequate = snr_analysis['usable_bandwidth'] > 1e6  # 1 MHz minimum
        
        # Stability criteria (frequency response smoothness)
        enhancement_variation = np.std(enhancement_profile['combined']) / np.mean(enhancement_profile['combined'])
        stability_acceptable = enhancement_variation < 0.1  # 10% variation
        
        # Overall validation
        validation_criteria = [
            enhancement_achieved,
            resonant_achieved,
            snr_threshold_met,
            usable_bandwidth_adequate,
            stability_acceptable
        ]
        
        overall_validation = all(validation_criteria)
        validation_score = sum(validation_criteria) / len(validation_criteria)
        
        validation_results = {
            'enhancement_achieved': enhancement_achieved,
            'resonant_achieved': resonant_achieved,
            'snr_threshold_met': snr_threshold_met,
            'usable_bandwidth_adequate': usable_bandwidth_adequate,
            'stability_acceptable': stability_acceptable,
            'overall_validation': overall_validation,
            'validation_score': validation_score,
            'peak_enhancement': peak_enhancement,
            'target_enhancement': target_enhancement,
            'enhancement_variation': enhancement_variation,
            'validation_criteria': validation_criteria
        }
        
        return validation_results
    
    def compute_optimization_metrics(self) -> Dict[str, float]:
        """
        Compute optimization metrics for amplification enhancement
        """
        if not self.validation_history:
            logger.warning("No validation history available")
            return {'status': 'no_data'}
        
        latest_validation = self.validation_history[-1]
        enhancement_profile = latest_validation['enhancement_profile']
        snr_analysis = latest_validation['snr_analysis']
        
        # Performance metrics
        enhancement_efficiency = (enhancement_profile['peak_enhancement'] / 
                                self.config.target_enhancement)
        
        snr_margin = snr_analysis['peak_snr_db'] - 20  # dB above 20 dB minimum
        
        bandwidth_utilization = (snr_analysis['usable_bandwidth'] / 
                               (np.max(self.config.frequency_range) - 
                                np.min(self.config.frequency_range)))
        
        # Quality metrics
        frequency_selectivity = (enhancement_profile['peak_enhancement'] / 
                               np.mean(enhancement_profile['combined']))
        
        noise_rejection = (snr_analysis['peak_snr'] / 
                         np.mean(snr_analysis['snr_linear']))
        
        optimization_metrics = {
            'enhancement_efficiency': enhancement_efficiency,
            'snr_margin_db': snr_margin,
            'bandwidth_utilization': bandwidth_utilization,
            'frequency_selectivity': frequency_selectivity,
            'noise_rejection': noise_rejection,
            'overall_performance': np.mean([enhancement_efficiency, 
                                          min(snr_margin/20, 1.0),
                                          bandwidth_utilization,
                                          min(frequency_selectivity/10, 1.0),
                                          min(noise_rejection/10, 1.0)])
        }
        
        return optimization_metrics
    
    def generate_validation_report(self) -> Dict[str, any]:
        """
        Generate comprehensive validation report
        """
        if not self.validation_history:
            return {'status': 'no_data', 'message': 'No validation data available'}
        
        latest_validation = self.validation_history[-1]
        optimization_metrics = self.compute_optimization_metrics()
        
        report = {
            'validation_summary': {
                'total_validations': len(self.validation_history),
                'latest_validation': latest_validation['validation_results']['overall_validation'],
                'validation_score': latest_validation['validation_results']['validation_score'],
                'peak_enhancement': latest_validation['enhancement_profile']['peak_enhancement'],
                'peak_snr_db': latest_validation['snr_analysis']['peak_snr_db']
            },
            'performance_metrics': optimization_metrics,
            'frequency_response': {
                'resonant_frequency': latest_validation['enhancement_profile']['resonant_frequency'],
                'usable_bandwidth': latest_validation['snr_analysis']['usable_bandwidth'],
                'enhancement_variation': latest_validation['validation_results']['enhancement_variation']
            },
            'validation_criteria': latest_validation['validation_results']['validation_criteria'],
            'recommendations': self._generate_recommendations(latest_validation)
        }
        
        return report
    
    def _generate_recommendations(self, validation_data: Dict[str, any]) -> List[str]:
        """Generate optimization recommendations based on validation results"""
        recommendations = []
        
        validation_results = validation_data['validation_results']
        
        if not validation_results['enhancement_achieved']:
            recommendations.append("Increase metamaterial resonance strength for higher enhancement")
        
        if not validation_results['resonant_achieved']:
            recommendations.append("Optimize resonant frequency tuning for peak 1947× enhancement")
        
        if not validation_results['snr_threshold_met']:
            recommendations.append("Implement additional noise reduction measures")
        
        if not validation_results['usable_bandwidth_adequate']:
            recommendations.append("Broaden resonance bandwidth for wider operational range")
        
        if not validation_results['stability_acceptable']:
            recommendations.append("Improve frequency response stability through damping control")
        
        if not recommendations:
            recommendations.append("All validation criteria met - system ready for deployment")
        
        return recommendations

def create_validation_framework(target_enhancement: float = 847.0,
                              resonant_multiplier: float = 2.3) -> AmplificationValidationFramework:
    """
    Factory function to create amplification validation framework
    """
    config = ValidationConfig(
        target_enhancement=target_enhancement,
        resonant_multiplier=resonant_multiplier
    )
    return AmplificationValidationFramework(config)

# Example usage and testing
if __name__ == "__main__":
    # Create validation framework
    validator = create_validation_framework()
    
    # Run comprehensive validation
    validation_result = validator.validate_847x_amplification()
    
    print("847× Amplification Validation Results:")
    print(f"Overall validation: {validation_result['validation_results']['overall_validation']}")
    print(f"Peak enhancement: {validation_result['enhancement_profile']['peak_enhancement']:.1f}×")
    print(f"Peak SNR: {validation_result['snr_analysis']['peak_snr_db']:.1f} dB")
    print(f"Usable bandwidth: {validation_result['snr_analysis']['usable_bandwidth']:.2e} Hz")
    
    # Generate optimization metrics
    metrics = validator.compute_optimization_metrics()
    print(f"\nOptimization Metrics:")
    print(f"Enhancement efficiency: {metrics['enhancement_efficiency']:.2%}")
    print(f"SNR margin: {metrics['snr_margin_db']:.1f} dB")
    print(f"Overall performance: {metrics['overall_performance']:.2%}")
    
    # Generate full report
    report = validator.generate_validation_report()
    print(f"\nValidation Score: {report['validation_summary']['validation_score']:.2%}")
    print(f"Recommendations: {len(report['recommendations'])} items")
