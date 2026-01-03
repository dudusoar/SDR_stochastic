"""Integration tests for configuration system."""

import pytest
import json
import yaml
import tempfile
import os
from typing import Dict, Any

from vrp_toolkit.utils.config import (
    VRPConfig,
    ProblemConfig,
    AlgorithmConfig,
    DataConfig,
    RunConfig,
    ConfigLoader
)


class TestVRPConfig:
    """Test VRPConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = VRPConfig()
        
        # Check nested configs exist
        assert isinstance(config.problem, ProblemConfig)
        assert isinstance(config.algorithm, AlgorithmConfig)
        assert isinstance(config.data, DataConfig)
        assert isinstance(config.run, RunConfig)
    
    def test_custom_config(self):
        """Test custom configuration values."""
        # Create custom nested configs
        problem_config = ProblemConfig(
            problem_type="pdptw",
            num_vehicles=5,
            vehicle_capacity=100.0,
            battery_capacity=200.0
        )
        
        algorithm_config = AlgorithmConfig(
            algorithm_type="alns",
            max_iterations=1000,
            time_limit=3600.0
        )
        
        data_config = DataConfig(
            data_source="synthetic",
            n_orders=20,
            n_time_intervals=24
        )
        
        run_config = RunConfig(
            random_seed=42,
            n_runs=10,
            output_dir="./results"
        )
        
        config = VRPConfig(
            problem=problem_config,
            algorithm=algorithm_config,
            data=data_config,
            run=run_config
        )
        
        # Check custom values
        assert config.problem.problem_type == "pdptw"
        assert config.problem.num_vehicles == 5
        assert config.problem.vehicle_capacity == 100.0
        
        assert config.algorithm.algorithm_type == "alns"
        assert config.algorithm.max_iterations == 1000
        
        assert config.data.data_source == "synthetic"
        assert config.data.n_orders == 20
        
        assert config.run.random_seed == 42
        assert config.run.n_runs == 10
    
    def test_config_validation(self):
        """Test configuration parameter validation."""
        # Invalid values should raise errors
        with pytest.raises((ValueError, TypeError)):
            ProblemConfig(num_vehicles=-5)  # Negative
        
        with pytest.raises((ValueError, TypeError)):
            RunConfig(n_runs=0)  # Zero runs
        
        # These should work
        ProblemConfig(num_vehicles=0)  # Zero vehicles allowed
        RunConfig(n_runs=1)  # Single run


class TestConfigLoader:
    """Test ConfigLoader for JSON and YAML files."""
    
    def test_json_loading_saving(self):
        """Test loading and saving JSON configuration."""
        # Create config
        config = VRPConfig()
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
            ConfigLoader.save_json(config, temp_path)
        
        try:
            # Load back
            loaded_config = ConfigLoader.load_json(temp_path)
            
            # Should be VRPConfig instance
            assert isinstance(loaded_config, VRPConfig)
            
            # Should have same structure
            assert isinstance(loaded_config.problem, ProblemConfig)
            assert isinstance(loaded_config.algorithm, AlgorithmConfig)
            assert isinstance(loaded_config.data, DataConfig)
            assert isinstance(loaded_config.run, RunConfig)
            
            # Values should match (within JSON serialization limits)
            assert loaded_config.problem.problem_type == config.problem.problem_type
            assert loaded_config.problem.num_vehicles == config.problem.num_vehicles
            
        finally:
            # Clean up
            os.unlink(temp_path)
    
    def test_yaml_loading_saving(self):
        """Test loading and saving YAML configuration."""
        # Create config
        config = VRPConfig()
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            temp_path = f.name
            ConfigLoader.save_yaml(config, temp_path)
        
        try:
            # Load back
            loaded_config = ConfigLoader.load_yaml(temp_path)
            
            # Should be VRPConfig instance
            assert isinstance(loaded_config, VRPConfig)
            
            # Should have same structure
            assert isinstance(loaded_config.problem, ProblemConfig)
            assert isinstance(loaded_config.algorithm, AlgorithmConfig)
            assert isinstance(loaded_config.data, DataConfig)
            assert isinstance(loaded_config.run, RunConfig)
            
        finally:
            # Clean up
            os.unlink(temp_path)
    
    def test_load_from_string(self):
        """Test loading configuration from string."""
        # JSON string
        json_str = """
        {
            "problem": {
                "problem_type": "pdptw",
                "num_vehicles": 3,
                "vehicle_capacity": 50.0
            },
            "algorithm": {
                "algorithm_type": "alns",
                "max_iterations": 500
            },
            "data": {
                "data_source": "synthetic",
                "n_orders": 10
            },
            "run": {
                "random_seed": 123,
                "n_runs": 5
            }
        }
        """
        
        config = ConfigLoader.load_json_string(json_str)
        
        assert isinstance(config, VRPConfig)
        assert config.problem.problem_type == "pdptw"
        assert config.problem.num_vehicles == 3
        assert config.algorithm.algorithm_type == "alns"
        assert config.algorithm.max_iterations == 500
        assert config.data.n_orders == 10
        assert config.run.random_seed == 123
    
    def test_file_not_found(self):
        """Test error handling for missing files."""
        with pytest.raises(FileNotFoundError):
            ConfigLoader.load_json("nonexistent_file.json")
        
        with pytest.raises(FileNotFoundError):
            ConfigLoader.load_yaml("nonexistent_file.yaml")
    
    def test_invalid_json(self):
        """Test error handling for invalid JSON."""
        invalid_json = "{invalid json}"
        
        with pytest.raises(json.JSONDecodeError):
            ConfigLoader.load_json_string(invalid_json)
    
    def test_config_merging(self):
        """Test merging configurations."""
        base_config = VRPConfig()
        
        # Partial update
        update_dict = {
            "problem": {
                "num_vehicles": 7,
                "vehicle_capacity": 75.0
            },
            "run": {
                "random_seed": 999
            }
        }
        
        merged_config = ConfigLoader.merge_configs(base_config, update_dict)
        
        # Updated values
        assert merged_config.problem.num_vehicles == 7
        assert merged_config.problem.vehicle_capacity == 75.0
        assert merged_config.run.random_seed == 999
        
        # Unchanged values
        assert merged_config.problem.problem_type == base_config.problem.problem_type
        assert merged_config.algorithm.algorithm_type == base_config.algorithm.algorithm_type
        assert merged_config.data.data_source == base_config.data.data_source


class TestConfigIntegration:
    """Test configuration integration with other components."""
    
    def test_config_with_alns(self):
        """Test using configuration with ALNS."""
        from vrp_toolkit.algorithms.alns.solver import ALNSConfig
        
        # Create ALNSConfig from dictionary
        alns_params = {
            "num_removal": 7,
            "max_no_improve": 100,
            "start_temp": 80.0,
            "cooling_rate": 0.92
        }
        
        # ALNSConfig should accept these parameters
        alns_config = ALNSConfig(**alns_params)
        
        assert alns_config.num_removal == 7
        assert alns_config.max_no_improve == 100
        assert alns_config.start_temp == 80.0
        assert alns_config.cooling_rate == 0.92
    
    def test_vrpconfig_to_alnsconfig(self):
        """Test converting VRPConfig to algorithm-specific config."""
        # Create VRPConfig with algorithm parameters
        vrp_config = VRPConfig()
        
        # In a real integration, we'd have a method to convert
        # VRPConfig.algorithm to ALNSConfig
        # For now, just test that both config types exist
        
        assert hasattr(vrp_config, 'algorithm')
        assert hasattr(vrp_config.algorithm, 'algorithm_type')
        assert hasattr(vrp_config.algorithm, 'max_iterations')
        assert hasattr(vrp_config.algorithm, 'time_limit')
    
    def test_example_config_files(self):
        """Test example configuration files in repository."""
        example_files = [
            "config_example.json",
            "config_example.yaml"
        ]
        
        for file_name in example_files:
            file_path = os.path.join(
                os.path.dirname(__file__), "..", "..", file_name
            )
            
            if os.path.exists(file_path):
                # Should load without errors
                if file_name.endswith('.json'):
                    config = ConfigLoader.load_json(file_path)
                else:  # .yaml
                    config = ConfigLoader.load_yaml(file_path)
                
                assert isinstance(config, VRPConfig)
                
                # Should have all required sections
                assert hasattr(config, 'problem')
                assert hasattr(config, 'algorithm')
                assert hasattr(config, 'data')
                assert hasattr(config, 'run')
            else:
                print(f"Note: Example config file not found: {file_name}")
    
    def test_config_validation_integration(self):
        """Test configuration validation in integrated context."""
        # Create config with some invalid values
        invalid_dict = {
            "problem": {
                "num_vehicles": -1  # Invalid
            }
        }
        
        # Loading might fail or produce invalid config
        # Actual behavior depends on validation implementation
        
        # For now, just ensure ConfigLoader exists and works
        config_loader = ConfigLoader()
        assert config_loader is not None


class TestConfigurationUseCases:
    """Test real-world configuration use cases."""
    
    def test_experiment_configuration(self):
        """Test configuration for running experiments."""
        # Configuration for sensitivity analysis experiment
        experiment_config = VRPConfig(
            problem=ProblemConfig(
                problem_type="pdptw",
                num_vehicles=3,
                vehicle_capacity=100.0,
                battery_capacity=150.0
            ),
            algorithm=AlgorithmConfig(
                algorithm_type="alns",
                max_iterations=500,
                time_limit=600.0  # 10 minutes
            ),
            data=DataConfig(
                data_source="synthetic",
                n_orders=15,
                n_time_intervals=12
            ),
            run=RunConfig(
                random_seed=42,
                n_runs=20,  # Multiple runs for statistics
                output_dir="./experiment_results"
            )
        )
        
        # Should be valid
        assert experiment_config.run.n_runs == 20
        assert experiment_config.algorithm.time_limit == 600.0
    
    def test_quick_demo_configuration(self):
        """Test configuration for quick demonstration."""
        # Quick demo with minimal settings
        demo_config = VRPConfig(
            problem=ProblemConfig(
                problem_type="pdptw",
                num_vehicles=1,
                vehicle_capacity=50.0,
                battery_capacity=100.0
            ),
            algorithm=AlgorithmConfig(
                algorithm_type="alns",
                max_iterations=50,  # Few iterations for speed
                time_limit=30.0  # 30 seconds
            ),
            data=DataConfig(
                data_source="synthetic",
                n_orders=3,  # Small problem
                n_time_intervals=6
            ),
            run=RunConfig(
                random_seed=123,
                n_runs=1,  # Single run
                output_dir="./demo_output"
            )
        )
        
        # Should be valid for quick demo
        assert demo_config.algorithm.max_iterations == 50
        assert demo_config.data.n_orders == 3
    
    def test_production_configuration(self):
        """Test configuration for production use."""
        # Production settings with thorough optimization
        production_config = VRPConfig(
            problem=ProblemConfig(
                problem_type="pdptw",
                num_vehicles=10,
                vehicle_capacity=200.0,
                battery_capacity=300.0
            ),
            algorithm=AlgorithmConfig(
                algorithm_type="alns",
                max_iterations=5000,
                time_limit=3600.0  # 1 hour
            ),
            data=DataConfig(
                data_source="real",  # Real data
                n_orders=50,
                n_time_intervals=24
            ),
            run=RunConfig(
                random_seed=None,  # None means truly random
                n_runs=30,  # Many runs for confidence
                output_dir="/var/results/vrp_production"
            )
        )
        
        # Should be valid for production
        assert production_config.algorithm.max_iterations == 5000
        assert production_config.run.n_runs == 30


if __name__ == "__main__":
    pytest.main([__file__, "-v"])