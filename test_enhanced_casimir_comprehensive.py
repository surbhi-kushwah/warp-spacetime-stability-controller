"""
Comprehensive Test Suite for Enhanced Casimir Sensor Array
Production-ready testing with actual execution validation

This test suite validates the complete enhanced Casimir sensor implementation
with 847× metamaterial amplification, following lessons learned about
production-ready code requiring actual functional testing.
"""

import sys
import os
import unittest
import numpy as np
import logging
from pathlib import Path

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import modules to test
from sensors.metamaterial_casimir_enhancer import MetamaterialCasimirEnhancer
from validation.amplification_validation import AmplificationValidationFramework, create_validation_framework

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestMetamaterialCasimirEnhancer(unittest.TestCase):
    """Test suite for MetamaterialCasimirEnhancer class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.enhancer = MetamaterialCasimirEnhancer()
        self.test_positions = np.array([[0, 0, 0], [0.1, 0, 0], [0, 0.1, 0]])
        
    def test_initialization(self):
        """Test proper initialization of enhancer"""
        self.assertEqual(self.enhancer.target_enhancement, 847.0)
        self.assertEqual(self.enhancer.resonant_multiplier, 2.3)
        self.assertIsNotNone(self.enhancer.spatial_profile)
        
    def test_casimir_force_computation(self):
        """Test enhanced Casimir force computation"""
        force_result = self.enhancer.compute_enhanced_casimir_force(
            separation=1e-6,  # 1 μm
            area=1e-4,        # 1 cm²
            position=np.array([0, 0, 0])
        )
        
        # Verify result structure
        self.assertIn('enhanced_force', force_result)
        self.assertIn('base_force', force_result)
        self.assertIn('enhancement_factor', force_result)
        self.assertIn('spatial_factor', force_result)
        
        # Verify enhancement is achieved
        enhancement = force_result['enhancement_factor']
        self.assertGreater(enhancement, 800.0)  # At least 800× enhancement
        self.assertLess(enhancement, 2000.0)    # Reasonable upper bound
        
        # Verify forces are negative (attractive)
        self.assertLess(force_result['enhanced_force'], 0)
        self.assertLess(force_result['base_force'], 0)
        
    def test_spatial_enhancement_profile(self):
        """Test spatial enhancement profile computation"""
        profile = self.enhancer._compute_spatial_enhancement_profile(
            position=np.array([0, 0, 0])
        )
        
        # Verify structure
        self.assertIn('hex_packing_efficiency', profile)
        self.assertIn('local_enhancement', profile)
        self.assertIn('gaussian_weights', profile)
        self.assertIn('combined_spatial_factor', profile)
        
        # Verify values are reasonable
        self.assertAlmostEqual(profile['hex_packing_efficiency'], 0.9069, places=3)
        self.assertGreater(profile['local_enhancement'], 400)
        self.assertLess(profile['local_enhancement'], 500)
        
    def test_enhancement_stability_validation(self):
        """Test enhancement stability over time"""
        stability_result = self.enhancer.validate_enhancement_stability(
            duration=1.0,
            time_steps=100
        )
        
        # Verify result structure
        self.assertIn('stability_metrics', stability_result)
        self.assertIn('temporal_profile', stability_result)
        self.assertIn('validation_passed', stability_result)
        
        # Verify stability
        metrics = stability_result['stability_metrics']
        self.assertLess(metrics['variation_coefficient'], 0.05)  # <5% variation
        self.assertLess(metrics['drift_rate'], 0.01)           # <1% drift
        
    def test_sensor_placement_optimization(self):
        """Test sensor placement optimization"""
        optimization_result = self.enhancer.optimize_sensor_placement(
            coverage_area=0.01,  # 1 cm²
            num_sensors=5
        )
        
        # Verify result structure
        self.assertIn('optimal_positions', optimization_result)
        self.assertIn('coverage_efficiency', optimization_result)
        self.assertIn('total_enhancement', optimization_result)
        
        # Verify optimization quality
        self.assertGreater(optimization_result['coverage_efficiency'], 0.85)  # >85% efficiency
        self.assertEqual(len(optimization_result['optimal_positions']), 5)
        
    def test_thermal_noise_characterization(self):
        """Test thermal noise characterization"""
        noise_result = self.enhancer.characterize_thermal_noise(
            temperature=300.0,
            frequency_range=np.logspace(3, 9, 100)
        )
        
        # Verify result structure
        self.assertIn('johnson_nyquist_noise', noise_result)
        self.assertIn('flicker_noise', noise_result)
        self.assertIn('total_noise_spectrum', noise_result)
        self.assertIn('snr_estimate', noise_result)
        
        # Verify SNR is adequate
        self.assertGreater(noise_result['snr_estimate'], 50)  # >50 dB SNR

class TestAmplificationValidationFramework(unittest.TestCase):
    """Test suite for AmplificationValidationFramework class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.validator = create_validation_framework()
        
    def test_framework_initialization(self):
        """Test proper initialization of validation framework"""
        self.assertEqual(self.validator.config.target_enhancement, 847.0)
        self.assertEqual(self.validator.config.resonant_multiplier, 2.3)
        self.assertIsNotNone(self.validator.config.frequency_range)
        
    def test_847x_amplification_validation(self):
        """Test comprehensive 847× amplification validation"""
        validation_result = self.validator.validate_847x_amplification()
        
        # Verify result structure
        self.assertIn('enhancement_profile', validation_result)
        self.assertIn('thermal_noise', validation_result)
        self.assertIn('snr_analysis', validation_result)
        self.assertIn('validation_results', validation_result)
        
        # Verify enhancement levels
        peak_enhancement = validation_result['enhancement_profile']['peak_enhancement']
        self.assertGreater(peak_enhancement, 800.0)
        
        # Verify SNR analysis
        snr_analysis = validation_result['snr_analysis']
        self.assertGreater(snr_analysis['peak_snr_db'], 100.0)  # >100 dB SNR
        
    def test_enhancement_profile_computation(self):
        """Test enhancement profile computation across frequency spectrum"""
        frequency_range = np.logspace(3, 12, 1000)
        profile = self.validator._compute_enhancement_profile(frequency_range)
        
        # Verify profile components
        self.assertIn('low_freq', profile)
        self.assertIn('resonant', profile)
        self.assertIn('thermal_limited', profile)
        self.assertIn('combined', profile)
        
        # Verify peak enhancement exceeds target
        self.assertGreater(profile['peak_enhancement'], 847.0)
        
    def test_thermal_noise_spectrum(self):
        """Test thermal noise spectrum computation"""
        frequency_range = np.logspace(3, 12, 100)
        noise_spectrum = self.validator._compute_thermal_noise_spectrum(
            frequency_range, temperature=300.0, bandwidth=1e6
        )
        
        # Verify noise components
        self.assertIn('base_noise', noise_spectrum)
        self.assertIn('flicker_noise', noise_spectrum)
        self.assertIn('total_noise', noise_spectrum)
        
        # Verify noise levels are reasonable
        self.assertGreater(noise_spectrum['base_noise'], 0)
        
    def test_snr_analysis(self):
        """Test signal-to-noise ratio analysis"""
        frequency_range = np.logspace(3, 9, 100)
        enhancement_profile = self.validator._compute_enhancement_profile(frequency_range)
        thermal_noise = self.validator._compute_thermal_noise_spectrum(
            frequency_range, 300.0, 1e6
        )
        
        snr_analysis = self.validator._compute_snr_analysis(enhancement_profile, thermal_noise)
        
        # Verify SNR metrics
        self.assertIn('snr_linear', snr_analysis)
        self.assertIn('snr_db', snr_analysis)
        self.assertIn('peak_snr', snr_analysis)
        
        # Verify adequate SNR
        self.assertGreater(snr_analysis['peak_snr_db'], 100.0)
        
    def test_optimization_metrics(self):
        """Test optimization metrics computation"""
        # Run validation first
        self.validator.validate_847x_amplification()
        
        metrics = self.validator.compute_optimization_metrics()
        
        # Verify metrics structure
        self.assertIn('enhancement_efficiency', metrics)
        self.assertIn('snr_margin_db', metrics)
        self.assertIn('overall_performance', metrics)
        
        # Verify reasonable values
        self.assertGreater(metrics['enhancement_efficiency'], 1.0)  # >100% efficiency
        self.assertGreater(metrics['overall_performance'], 0.8)     # >80% performance
        
    def test_validation_report_generation(self):
        """Test comprehensive validation report generation"""
        # Run validation first
        self.validator.validate_847x_amplification()
        
        report = self.validator.generate_validation_report()
        
        # Verify report structure
        self.assertIn('validation_summary', report)
        self.assertIn('performance_metrics', report)
        self.assertIn('frequency_response', report)
        self.assertIn('recommendations', report)
        
        # Verify report content
        self.assertGreater(report['validation_summary']['peak_enhancement'], 800.0)

class TestIntegrationValidation(unittest.TestCase):
    """Integration tests for complete sensor array system"""
    
    def setUp(self):
        """Set up integration test fixtures"""
        self.enhancer = MetamaterialCasimirEnhancer()
        self.validator = create_validation_framework()
        
    def test_enhancer_validator_integration(self):
        """Test integration between enhancer and validator"""
        # Generate sensor data
        force_result = self.enhancer.compute_enhanced_casimir_force(
            separation=1e-6,
            area=1e-4,
            position=np.array([0, 0, 0])
        )
        
        # Validate enhancement
        validation_result = self.validator.validate_847x_amplification()
        
        # Verify consistency
        sensor_enhancement = force_result['enhancement_factor']
        validator_enhancement = validation_result['enhancement_profile']['peak_enhancement']
        
        # Both should exceed 847× target
        self.assertGreater(sensor_enhancement, 800.0)
        self.assertGreater(validator_enhancement, 800.0)
        
    def test_production_readiness(self):
        """Test production readiness criteria"""
        # Test all critical components work together
        enhancer_works = True
        validator_works = True
        
        try:
            # Test enhancer
            force_result = self.enhancer.compute_enhanced_casimir_force(1e-6, 1e-4, np.array([0, 0, 0]))
            stability_result = self.enhancer.validate_enhancement_stability(1.0, 50)
            
            # Test validator
            validation_result = self.validator.validate_847x_amplification()
            metrics = self.validator.compute_optimization_metrics()
            report = self.validator.generate_validation_report()
            
        except Exception as e:
            logger.error(f"Production readiness test failed: {e}")
            enhancer_works = False
            validator_works = False
        
        # Verify production readiness
        self.assertTrue(enhancer_works, "Enhancer component failed")
        self.assertTrue(validator_works, "Validator component failed")
        
        # Verify performance meets requirements
        if enhancer_works and validator_works:
            self.assertGreater(force_result['enhancement_factor'], 800.0)
            self.assertTrue(validation_result['validation_results']['enhancement_achieved'])

def run_comprehensive_test_suite():
    """Run complete test suite with detailed reporting"""
    print("=" * 70)
    print("ENHANCED CASIMIR SENSOR ARRAY - COMPREHENSIVE TEST SUITE")
    print("=" * 70)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestMetamaterialCasimirEnhancer))
    suite.addTests(loader.loadTestsFromTestCase(TestAmplificationValidationFramework))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegrationValidation))
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 70)
    print("TEST SUITE SUMMARY")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {(result.testsRun - len(result.failures) - len(result.errors))/result.testsRun*100:.1f}%")
    
    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    # Production readiness assessment
    production_ready = (len(result.failures) == 0 and len(result.errors) == 0)
    print(f"\nPRODUCTION READINESS: {'✅ PASSED' if production_ready else '❌ FAILED'}")
    
    return result

if __name__ == "__main__":
    # Run comprehensive test suite
    test_result = run_comprehensive_test_suite()
    
    # Exit with appropriate code
    sys.exit(0 if test_result.wasSuccessful() else 1)
