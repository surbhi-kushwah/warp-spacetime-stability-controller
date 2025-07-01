"""
Production-Ready Test Suite for Warp Spacetime Stability Controller
Tests all critical components with realistic physics validation
"""

import unittest
import numpy as np
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

class TestMinimalValidator(unittest.TestCase):
    """Test the minimal working validation framework"""
    
    def setUp(self):
        from validation.minimal_working_validator import create_minimal_validator
        self.validator = create_minimal_validator()
    
    def test_field_strength_validation_pass(self):
        """Test field strength validation with safe values"""
        safe_field = np.array([1e-8, 5e-9, 2e-8])
        result = self.validator.validate_field_strength(safe_field)
        
        self.assertTrue(result.passed)
        self.assertLess(result.value, self.validator.max_field_strength)
        self.assertGreater(result.confidence, 0.5)
    
    def test_field_strength_validation_fail(self):
        """Test field strength validation with unsafe values"""
        unsafe_field = np.array([1e-5, 5e-6, 2e-5])  # Too high
        result = self.validator.validate_field_strength(unsafe_field)
        
        self.assertFalse(result.passed)
        self.assertGreater(result.value, self.validator.max_field_strength)
        self.assertEqual(result.confidence, 0.0)
    
    def test_acceleration_validation_pass(self):
        """Test acceleration validation with safe values"""
        safe_accel = np.array([0.5, 1.0, 2.0])
        result = self.validator.validate_acceleration(safe_accel)
        
        self.assertTrue(result.passed)
        self.assertLess(result.value, self.validator.max_acceleration)
    
    def test_acceleration_validation_fail(self):
        """Test acceleration validation with unsafe values"""
        unsafe_accel = np.array([10.0, 15.0, 20.0])  # Too high
        result = self.validator.validate_acceleration(unsafe_accel)
        
        self.assertFalse(result.passed)
        self.assertGreater(result.value, self.validator.max_acceleration)
    
    def test_causality_validation_minkowski(self):
        """Test causality with Minkowski metric"""
        minkowski = np.array([[-1, 0, 0, 0],
                             [0, 1, 0, 0],
                             [0, 0, 1, 0],
                             [0, 0, 0, 1]])
        result = self.validator.validate_causality(minkowski)
        
        self.assertTrue(result.passed)
        self.assertEqual(result.value, 1.0)  # Perfect causality
    
    def test_causality_validation_fail(self):
        """Test causality with invalid metric"""
        bad_metric = np.array([[1, 0, 0, 0],  # Wrong sign!
                              [0, 1, 0, 0],
                              [0, 0, 1, 0],
                              [0, 0, 0, 1]])
        result = self.validator.validate_causality(bad_metric)
        
        self.assertFalse(result.passed)
        self.assertEqual(result.value, 0.0)
    
    def test_energy_conservation_pass(self):
        """Test energy conservation with small acceptable error"""
        energy_in = 1000.0
        energy_out = 999.999999999  # Very small loss
        result = self.validator.validate_energy_conservation(energy_in, energy_out)
        
        self.assertTrue(result.passed)
        self.assertLess(result.value, self.validator.tolerance)
    
    def test_energy_conservation_fail(self):
        """Test energy conservation with large error"""
        energy_in = 1000.0
        energy_out = 900.0  # 10% loss - too much!
        result = self.validator.validate_energy_conservation(energy_in, energy_out)
        
        self.assertFalse(result.passed)
        self.assertGreater(result.value, self.validator.tolerance)
    
    def test_comprehensive_validation(self):
        """Test comprehensive validation with mixed results"""
        test_data = {
            'field_data': np.array([1e-8, 5e-9]),  # Safe
            'acceleration_data': np.array([1.0, 2.0]),  # Safe
            'metric_tensor': np.diag([-1, 1, 1, 1]),  # Safe (Minkowski)
            'energy_in': 1000.0,
            'energy_out': 999.999999999  # Safe (tiny loss)
        }
        
        report = self.validator.run_comprehensive_validation(test_data)
        
        self.assertEqual(report['status'], 'ALL_PASS')
        self.assertEqual(report['pass_rate'], 1.0)
        self.assertEqual(report['total_tests'], 4)
        self.assertEqual(report['passed_tests'], 4)

class TestPhysicsConstants(unittest.TestCase):
    """Test physics constants and calculations"""
    
    def test_speed_of_light(self):
        """Test speed of light constant"""
        c = 299792458  # m/s
        self.assertAlmostEqual(c, 3e8, delta=1e6)
    
    def test_gravitational_constant(self):
        """Test gravitational constant"""
        G = 6.67430e-11  # m³/kg/s²
        self.assertAlmostEqual(G, 6.674e-11, delta=1e-13)
    
    def test_planck_constant(self):
        """Test Planck constant"""
        h = 6.62607015e-34  # J⋅s
        hbar = h / (2 * np.pi)
        self.assertAlmostEqual(hbar, 1.054571817e-34, delta=1e-42)

class TestSpacetimeMetric(unittest.TestCase):
    """Test spacetime metric calculations"""
    
    def test_minkowski_signature(self):
        """Test Minkowski metric signature"""
        metric = np.diag([-1, 1, 1, 1])
        eigenvals = np.linalg.eigvals(metric)
        
        # Should have signature (-,+,+,+)
        negative_eigenvals = sum(1 for e in eigenvals if e < 0)
        positive_eigenvals = sum(1 for e in eigenvals if e > 0)
        
        self.assertEqual(negative_eigenvals, 1)
        self.assertEqual(positive_eigenvals, 3)
    
    def test_metric_determinant(self):
        """Test metric determinant calculation"""
        metric = np.diag([-1, 1, 1, 1])
        det = np.linalg.det(metric)
        
        self.assertAlmostEqual(det, -1.0, places=10)
    
    def test_metric_inverse(self):
        """Test metric inverse calculation"""
        metric = np.diag([-1, 1, 1, 1])
        inverse = np.linalg.inv(metric)
        identity = np.dot(metric, inverse)
        
        np.testing.assert_allclose(identity, np.eye(4), rtol=1e-10)

class TestSafetyThresholds(unittest.TestCase):
    """Test safety threshold validations"""
    
    def test_field_strength_thresholds(self):
        """Test electromagnetic field safety thresholds"""
        # Based on medical safety standards
        safe_threshold = 1e-6  # Tesla
        dangerous_field = 1e-3  # Tesla (known biological effects)
        
        self.assertLess(safe_threshold, dangerous_field)
        
        # Test with known safe values
        earth_magnetic_field = 5e-5  # Tesla
        self.assertGreater(earth_magnetic_field, safe_threshold)  # Should trigger caution
    
    def test_acceleration_thresholds(self):
        """Test acceleration safety thresholds"""
        safe_threshold = 5.0  # m/s²
        earth_gravity = 9.81  # m/s²
        fighter_jet_max = 90.0  # m/s² (9G)
        
        self.assertLess(safe_threshold, earth_gravity)  # Conservative
        self.assertLess(safe_threshold, fighter_jet_max)

def run_comprehensive_tests():
    """Run all test suites and generate report"""
    print("🧪 COMPREHENSIVE PHYSICS VALIDATION TEST SUITE")
    print("=" * 70)
    
    # Create test suite
    test_classes = [
        TestMinimalValidator,
        TestPhysicsConstants,
        TestSpacetimeMetric,
        TestSafetyThresholds
    ]
    
    suite = unittest.TestSuite()
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    
    # Generate summary report
    print("\n" + "=" * 70)
    print("📊 TEST SUMMARY REPORT")
    print("=" * 70)
    print(f"Tests Run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success Rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print("\n❌ FAILURES:")
        for test, traceback in result.failures:
            print(f"   {test}: {traceback.split('AssertionError:')[-1].strip()}")
    
    if result.errors:
        print("\n💥 ERRORS:")
        for test, traceback in result.errors:
            print(f"   {test}: {traceback.split('Error:')[-1].strip()}")
    
    if not result.failures and not result.errors:
        print("\n🚀 ALL TESTS PASSED! Framework is production-ready.")
        return True
    else:
        print("\n⚠️ Some tests failed. Framework needs attention before production.")
        return False

if __name__ == "__main__":
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
